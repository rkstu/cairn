"""Drug interaction and adverse event checking using OnSIDES (offline SQLite)."""

from __future__ import annotations

import sqlite3
from itertools import combinations
from pathlib import Path

from src.safety.vital_check import SafetyFlag, Severity


class DrugSafetyChecker:
    """Offline drug adverse event checker backed by OnSIDES SQLite database."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            if not self.db_path.exists():
                raise FileNotFoundError(
                    f"OnSIDES database not found at {self.db_path}. "
                    "Run `python scripts/download_onsides.py` to fetch it."
                )
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    # ADEs that genuinely warrant a triage warning (life-threatening or dose-limiting)
    HIGH_SIGNAL_ADES = {
        "hepatotoxicity", "nephrotoxicity", "rhabdomyolysis", "anaphylaxis",
        "stevens-johnson syndrome", "ten", "agranulocytosis", "aplastic anemia",
        "serotonin syndrome", "neuroleptic malignant syndrome", "lactic acidosis",
        "hemorrhage", "bleeding", "haemorrhage", "cardiac arrest", "qt prolongation",
        "torsade de pointes", "respiratory depression", "seizure", "pancreatitis",
        "pulmonary embolism", "thrombocytopenia", "neutropenia", "hyperkalemia",
        "hyponatremia", "hypoglycemia", "adrenal insufficiency", "myocardial infarction",
        "stroke", "dic", "acute kidney injury", "liver failure", "renal failure",
        "angioedema", "bronchospasm", "status epilepticus", "coma",
    }

    def check_adverse_events(self, medication: str) -> list[SafetyFlag]:
        """Check for known serious adverse events for a single medication."""
        flags: list[SafetyFlag] = []
        try:
            cursor = self.conn.execute(
                """
                SELECT DISTINCT adverse_event
                FROM adverse_effects
                WHERE LOWER(ingredient) = LOWER(?)
                LIMIT 200
                """,
                (medication,),
            )
            rows = cursor.fetchall()
            if rows:
                high_signal = [r[0] for r in rows if r[0].lower() in self.HIGH_SIGNAL_ADES]
                if high_signal:
                    flags.append(SafetyFlag(
                        flag_type="DRUG_ADE",
                        severity=Severity.WARNING,
                        detail=f"{medication}: known risks include {', '.join(high_signal[:5])}",
                        action="WARN",
                    ))
        except sqlite3.OperationalError:
            pass
        return flags

    def check_interactions(self, medications: list[str]) -> list[SafetyFlag]:
        """Check for drug-drug interactions between a list of medications."""
        flags: list[SafetyFlag] = []
        if len(medications) < 2:
            return flags

        for med_a, med_b in combinations(medications, 2):
            try:
                cursor = self.conn.execute(
                    """
                    SELECT description, severity
                    FROM drug_interactions
                    WHERE (LOWER(drug_a) = LOWER(?) AND LOWER(drug_b) = LOWER(?))
                       OR (LOWER(drug_a) = LOWER(?) AND LOWER(drug_b) = LOWER(?))
                    """,
                    (med_a, med_b, med_b, med_a),
                )
                rows = cursor.fetchall()
                for row in rows:
                    sev = Severity.CRITICAL if row["severity"] == "high" else Severity.WARNING
                    flags.append(SafetyFlag(
                        flag_type="DRUG_INTERACTION",
                        severity=sev,
                        detail=f"{med_a} + {med_b}: {row['description']}",
                        action="BLOCK" if sev == Severity.CRITICAL else "WARN",
                    ))
            except sqlite3.OperationalError:
                pass  # Table doesn't exist yet

        return flags

    def check_all(self, medications: list[str]) -> list[SafetyFlag]:
        """Run all drug safety checks.

        Prioritizes interactions (actionable) over individual ADEs (informational).
        Individual ADE warnings only appear for high-alert medications to avoid
        alarm fatigue — a CHW entering 5 meds shouldn't see 50 warnings.
        """
        flags: list[SafetyFlag] = []
        flags.extend(self.check_interactions(medications))
        # Only show individual ADEs for high-alert meds (anticoagulants, opioids, etc.)
        high_alert = {"warfarin", "heparin", "enoxaparin", "insulin", "metformin",
                      "digoxin", "lithium", "methotrexate", "cyclosporine",
                      "morphine", "fentanyl", "oxycodone", "hydromorphone"}
        for med in medications:
            if med.lower() in high_alert:
                flags.extend(self.check_adverse_events(med))
        return flags

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
