"""Download and prepare the OnSIDES drug adverse event database for offline use."""

import sqlite3
import csv
import io
import zipfile
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "onsides.db"

# OnSIDES release data URLs (from tatonetti-lab/onsides GitHub releases)
ONSIDES_BASE = "https://github.com/tatonetti-lab/onsides/releases/download"
ONSIDES_VERSION = "v2.1.0"
ADVERSE_EVENTS_URL = f"{ONSIDES_BASE}/{ONSIDES_VERSION}/onsides_v2.1.0_adverse_reactions.csv.zip"


def download_and_extract(url: str) -> str:
    """Download a zip file and return the CSV content."""
    print(f"Downloading {url}...")
    resp = httpx.get(url, follow_redirects=True, timeout=120.0)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        csv_name = zf.namelist()[0]
        print(f"  Extracting {csv_name}...")
        return zf.read(csv_name).decode("utf-8")


def build_db():
    """Build the OnSIDES SQLite database from CSV data."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists():
        print(f"Database already exists at {DB_PATH}")
        print("Delete it to rebuild.")
        return

    print("Building OnSIDES database...")
    conn = sqlite3.connect(str(DB_PATH))

    # Create tables
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS adverse_effects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ingredient TEXT NOT NULL,
            adverse_event TEXT NOT NULL,
            severity TEXT DEFAULT 'unknown',
            source TEXT DEFAULT 'onsides'
        );
        CREATE INDEX IF NOT EXISTS idx_ingredient ON adverse_effects(LOWER(ingredient));

        CREATE TABLE IF NOT EXISTS drug_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            drug_a TEXT NOT NULL,
            drug_b TEXT NOT NULL,
            description TEXT,
            severity TEXT DEFAULT 'unknown'
        );
        CREATE INDEX IF NOT EXISTS idx_drug_a ON drug_interactions(LOWER(drug_a));
        CREATE INDEX IF NOT EXISTS idx_drug_b ON drug_interactions(LOWER(drug_b));
    """)

    # Try downloading OnSIDES data
    try:
        csv_data = download_and_extract(ADVERSE_EVENTS_URL)
        reader = csv.DictReader(io.StringIO(csv_data))
        rows = []
        for row in reader:
            ingredient = row.get("ingredient", row.get("drug_concept_name", "")).strip()
            event = row.get("adverse_reaction", row.get("condition_concept_name", "")).strip()
            if ingredient and event:
                rows.append((ingredient, event, "serious", "onsides"))
        conn.executemany(
            "INSERT INTO adverse_effects (ingredient, adverse_event, severity, source) VALUES (?, ?, ?, ?)",
            rows,
        )
        print(f"  Loaded {len(rows)} adverse event records.")
    except Exception as e:
        print(f"  Warning: Could not download OnSIDES data: {e}")
        print("  Seeding with critical drug interactions manually...")
        _seed_critical_interactions(conn)

    conn.commit()

    # Stats
    count = conn.execute("SELECT COUNT(*) FROM adverse_effects").fetchone()[0]
    print(f"Database ready at {DB_PATH} ({count} adverse event records)")
    conn.close()


def _seed_critical_interactions(conn: sqlite3.Connection):
    """Seed with clinically significant drug-drug interactions.

    Sources: FDA label warnings, UpToDate interaction checker categories,
    ISMP high-alert medication list, WHO Essential Medicines interactions.
    Covers the most commonly prescribed drugs and highest-risk combinations.
    """
    critical = [
        # === ANTICOAGULANT INTERACTIONS (bleeding risk) ===
        ("warfarin", "aspirin", "Increased bleeding risk — major interaction", "high"),
        ("warfarin", "ibuprofen", "Increased bleeding risk — major interaction", "high"),
        ("warfarin", "naproxen", "Increased bleeding risk — NSAID interaction", "high"),
        ("warfarin", "diclofenac", "Increased bleeding risk — NSAID interaction", "high"),
        ("warfarin", "celecoxib", "Increased bleeding risk — COX-2 interaction", "moderate"),
        ("warfarin", "fluconazole", "Increased warfarin effect — bleeding risk", "high"),
        ("warfarin", "metronidazole", "Increased warfarin effect — bleeding risk", "high"),
        ("warfarin", "trimethoprim", "Increased INR — bleeding risk", "high"),
        ("warfarin", "amiodarone", "Markedly increased warfarin effect", "high"),
        ("warfarin", "ciprofloxacin", "Increased INR — bleeding risk", "high"),
        ("heparin", "aspirin", "Increased bleeding risk", "high"),
        ("enoxaparin", "aspirin", "Increased bleeding risk — dual anticoagulation", "high"),
        ("rivaroxaban", "aspirin", "Increased bleeding risk", "high"),
        ("apixaban", "aspirin", "Increased bleeding risk", "high"),
        ("clopidogrel", "aspirin", "Increased bleeding risk — dual antiplatelet", "moderate"),
        ("clopidogrel", "omeprazole", "Reduced clopidogrel efficacy — CYP2C19 inhibition", "high"),
        ("clopidogrel", "esomeprazole", "Reduced clopidogrel efficacy — CYP2C19 inhibition", "high"),

        # === NSAID INTERACTIONS ===
        ("aspirin", "ibuprofen", "Reduced aspirin cardioprotection + increased GI bleeding", "high"),
        ("aspirin", "naproxen", "Increased GI bleeding risk", "moderate"),
        ("ibuprofen", "naproxen", "Duplicate NSAID — no added benefit, increased GI/renal risk", "high"),
        ("ibuprofen", "diclofenac", "Duplicate NSAID — increased toxicity risk", "high"),
        ("ibuprofen", "lithium", "Increased lithium levels — toxicity risk", "high"),
        ("naproxen", "lithium", "Increased lithium levels — toxicity risk", "high"),
        ("ibuprofen", "methotrexate", "Increased methotrexate toxicity — potentially fatal", "high"),
        ("naproxen", "methotrexate", "Increased methotrexate toxicity — potentially fatal", "high"),
        ("ibuprofen", "lisinopril", "Reduced antihypertensive effect + renal risk", "moderate"),
        ("ibuprofen", "enalapril", "Reduced antihypertensive effect + renal risk", "moderate"),
        ("ibuprofen", "losartan", "Reduced antihypertensive effect + renal risk", "moderate"),
        ("ibuprofen", "furosemide", "Reduced diuretic effect + renal risk", "moderate"),
        ("naproxen", "furosemide", "Reduced diuretic effect + renal risk", "moderate"),

        # === CARDIOVASCULAR ===
        ("digoxin", "amiodarone", "Increased digoxin levels — toxicity risk", "high"),
        ("digoxin", "verapamil", "Increased digoxin levels + AV block risk", "high"),
        ("digoxin", "diltiazem", "Increased digoxin levels + bradycardia", "high"),
        ("digoxin", "quinidine", "Doubled digoxin levels — toxicity", "high"),
        ("amiodarone", "simvastatin", "Rhabdomyolysis risk — limit simvastatin to 20mg", "high"),
        ("amiodarone", "warfarin", "Markedly increased warfarin effect", "high"),
        ("metoprolol", "verapamil", "Severe bradycardia + heart block risk", "high"),
        ("atenolol", "verapamil", "Severe bradycardia + heart block risk", "high"),
        ("amlodipine", "simvastatin", "Increased statin levels — limit simvastatin to 20mg", "moderate"),
        ("spironolactone", "lisinopril", "Hyperkalemia risk — monitor potassium", "high"),
        ("spironolactone", "enalapril", "Hyperkalemia risk — monitor potassium", "high"),
        ("spironolactone", "losartan", "Hyperkalemia risk — monitor potassium", "high"),
        ("spironolactone", "potassium", "Severe hyperkalemia risk", "high"),

        # === STATIN INTERACTIONS (rhabdomyolysis) ===
        ("simvastatin", "erythromycin", "Rhabdomyolysis risk — contraindicated", "high"),
        ("simvastatin", "clarithromycin", "Rhabdomyolysis risk — contraindicated", "high"),
        ("simvastatin", "itraconazole", "Rhabdomyolysis risk — contraindicated", "high"),
        ("simvastatin", "ketoconazole", "Rhabdomyolysis risk — contraindicated", "high"),
        ("simvastatin", "cyclosporine", "Rhabdomyolysis risk — contraindicated", "high"),
        ("atorvastatin", "clarithromycin", "Increased statin levels — rhabdomyolysis risk", "high"),
        ("lovastatin", "erythromycin", "Rhabdomyolysis risk — contraindicated", "high"),

        # === SEROTONERGIC (serotonin syndrome) ===
        ("fluoxetine", "tramadol", "Serotonin syndrome risk", "high"),
        ("sertraline", "tramadol", "Serotonin syndrome risk", "high"),
        ("paroxetine", "tramadol", "Serotonin syndrome risk", "high"),
        ("fluoxetine", "sumatriptan", "Serotonin syndrome risk", "high"),
        ("sertraline", "sumatriptan", "Serotonin syndrome risk", "high"),
        ("fluoxetine", "linezolid", "Serotonin syndrome risk — potentially fatal", "high"),
        ("sertraline", "linezolid", "Serotonin syndrome risk — potentially fatal", "high"),
        ("MAOIs", "SSRIs", "Serotonin syndrome — potentially fatal", "high"),
        ("MAOIs", "meperidine", "Serotonin syndrome — potentially fatal", "high"),
        ("MAOIs", "tramadol", "Serotonin syndrome — potentially fatal", "high"),
        ("fluoxetine", "MAOIs", "Serotonin syndrome — 5 week washout required", "high"),

        # === DIABETES ===
        ("metformin", "contrast dye", "Lactic acidosis risk — hold metformin 48hrs", "high"),
        ("insulin", "metformin", "Additive hypoglycemia risk — monitor closely", "moderate"),
        ("glipizide", "fluconazole", "Severe hypoglycemia — CYP2C9 inhibition", "high"),
        ("glyburide", "fluconazole", "Severe hypoglycemia — CYP2C9 inhibition", "high"),
        ("metformin", "alcohol", "Lactic acidosis risk — avoid heavy use", "moderate"),

        # === RENAL/ELECTROLYTE ===
        ("ACE inhibitor", "potassium", "Hyperkalemia risk", "high"),
        ("lisinopril", "potassium", "Hyperkalemia risk", "high"),
        ("enalapril", "potassium", "Hyperkalemia risk", "high"),
        ("losartan", "potassium", "Hyperkalemia risk", "high"),
        ("furosemide", "gentamicin", "Increased ototoxicity and nephrotoxicity", "high"),
        ("furosemide", "lithium", "Increased lithium levels — toxicity risk", "high"),
        ("hydrochlorothiazide", "lithium", "Increased lithium levels — toxicity risk", "high"),

        # === ANTIBIOTICS ===
        ("ciprofloxacin", "tizanidine", "Dramatically increased tizanidine levels — contraindicated", "high"),
        ("ciprofloxacin", "theophylline", "Increased theophylline toxicity — seizure risk", "high"),
        ("metronidazole", "alcohol", "Disulfiram-like reaction — severe nausea/vomiting", "high"),
        ("trimethoprim", "methotrexate", "Bone marrow suppression — potentially fatal", "high"),
        ("erythromycin", "carbamazepine", "Increased carbamazepine toxicity", "high"),
        ("rifampin", "oral contraceptives", "Reduced contraceptive efficacy — pregnancy risk", "high"),
        ("rifampin", "warfarin", "Markedly reduced warfarin effect — clotting risk", "high"),

        # === CNS/SEDATION ===
        ("opioids", "benzodiazepines", "Respiratory depression — FDA black box warning", "high"),
        ("morphine", "diazepam", "Respiratory depression risk", "high"),
        ("oxycodone", "alprazolam", "Respiratory depression risk", "high"),
        ("fentanyl", "midazolam", "Respiratory depression risk", "high"),
        ("gabapentin", "opioids", "Increased CNS/respiratory depression", "high"),
        ("pregabalin", "opioids", "Increased CNS/respiratory depression", "high"),
        ("carbamazepine", "valproic acid", "Reduced valproate levels + carbamazepine toxicity", "high"),
        ("phenytoin", "valproic acid", "Complex interaction — monitor both levels", "high"),

        # === IMMUNOSUPPRESSANTS ===
        ("cyclosporine", "erythromycin", "Increased cyclosporine toxicity — nephrotoxicity", "high"),
        ("cyclosporine", "ketoconazole", "Increased cyclosporine levels — toxicity", "high"),
        ("methotrexate", "trimethoprim", "Bone marrow suppression — potentially fatal", "high"),
        ("methotrexate", "NSAIDs", "Increased methotrexate toxicity", "high"),

        # === QT PROLONGATION ===
        ("amiodarone", "azithromycin", "QT prolongation — torsades de pointes risk", "high"),
        ("amiodarone", "levofloxacin", "QT prolongation — torsades de pointes risk", "high"),
        ("haloperidol", "methadone", "QT prolongation — cardiac arrest risk", "high"),
        ("ondansetron", "methadone", "QT prolongation risk", "moderate"),
        ("azithromycin", "levofloxacin", "Additive QT prolongation", "moderate"),

        # === THYROID ===
        ("levothyroxine", "calcium", "Reduced levothyroxine absorption — separate by 4hrs", "moderate"),
        ("levothyroxine", "iron", "Reduced levothyroxine absorption — separate by 4hrs", "moderate"),
        ("levothyroxine", "omeprazole", "Reduced levothyroxine absorption", "moderate"),
        ("levothyroxine", "warfarin", "Increased warfarin effect — monitor INR", "moderate"),
    ]
    conn.executemany(
        "INSERT INTO drug_interactions (drug_a, drug_b, description, severity) VALUES (?, ?, ?, ?)",
        critical,
    )
    print(f"  Seeded {len(critical)} drug-drug interactions.")


if __name__ == "__main__":
    build_db()
