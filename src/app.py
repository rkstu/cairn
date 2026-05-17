"""Cairn Gradio Web UI — minimal-friction triage interface for clinicians."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from src.agent.executor import ToolExecutor
from src.agent.triage import TriageAgent, TriageCase
from src.config import CairnConfig
from src.sync.store import LocalStore

_config = CairnConfig(onsides_db_path="./data/onsides.db")
_agent = TriageAgent(_config)
_executor = ToolExecutor(_config)
_store = LocalStore("./data/cairn.db")

LEVEL_COLORS = {"RED": "#dc2626", "YELLOW": "#d97706", "GREEN": "#16a34a"}
LEVEL_LABELS = {"RED": "IMMEDIATE", "YELLOW": "URGENT", "GREEN": "NON-URGENT"}
LEVEL_ICONS = {"RED": "🔴", "YELLOW": "🟡", "GREEN": "🟢"}


def run_triage(
    description: str,
    medications: str,
    spo2: int | None,
    heart_rate: int | None,
    systolic_bp: int | None,
    diastolic_bp: int | None,
    temperature: float | None,
    glucose: int | None,
    image: str | None,
) -> tuple[str, str, str]:
    """Run triage — returns (result_html, safety_html, details_md)."""

    if not description or not description.strip():
        empty = (
            "<div style='text-align:center;padding:60px;color:#666;'>"
            "<p style='font-size:20px;font-weight:500;'>Enter patient information to begin triage</p>"
            "<p style='font-size:14px;color:#999;'>Describe symptoms, add vitals and medications below</p></div>"
        )
        return empty, "", ""

    vitals = {}
    if spo2: vitals["spo2"] = int(spo2)
    if heart_rate: vitals["heart_rate"] = int(heart_rate)
    if systolic_bp: vitals["systolic_bp"] = int(systolic_bp)
    if diastolic_bp: vitals["diastolic_bp"] = int(diastolic_bp)
    if temperature: vitals["temperature"] = float(temperature)
    if glucose: vitals["glucose"] = int(glucose)

    meds = [m.strip() for m in medications.split(",") if m.strip()] if medications else []

    result = _agent.run(
        patient_description=description, vitals=vitals or None,
        medications=meds or None, image_path=image,
    )

    fhir = _agent.to_fhir_encounter(result)
    _store.save_case({
        "case_id": result.case_id, "timestamp": result.timestamp,
        "patient_description": result.patient_description,
        "triage_level": result.triage_level, "confidence": result.confidence,
        "assessment": result.cascade_result.get("assessment", "")[:2000],
        "vitals": result.vitals, "medications": result.medications,
        "safety_flags": result.safety_result.get("flags", []),
        "fhir_resource": fhir, "flagged_for_human": result.flagged_for_human,
    })

    result_html = _format_result(result)
    safety_html = _format_safety(result)
    details_md = _format_details(result, fhir)

    return result_html, safety_html, details_md


def _format_result(result: TriageCase) -> str:
    level = result.triage_level
    color = LEVEL_COLORS.get(level, "#666")
    label = LEVEL_LABELS.get(level, level)
    icon = LEVEL_ICONS.get(level, "")
    conf = result.confidence
    model = result.cascade_result.get("model_used", "unknown")

    assessment = result.final_output or result.cascade_result.get("assessment", "")
    assessment_clean = assessment.replace("Triage Level:", "").replace("Confidence:", "")
    for prefix in ["RED\n", "YELLOW\n", "GREEN\n", f"{level}\n"]:
        if assessment_clean.strip().startswith(prefix):
            assessment_clean = assessment_clean.strip()[len(prefix):]
    assessment_html = assessment_clean.replace("\n", "<br>").strip()

    flagged = ""
    if result.flagged_for_human:
        flagged = (
            "<div style='background:#fef3c7;border:2px solid #d97706;border-radius:8px;"
            "padding:12px;margin-top:16px;text-align:center;font-weight:600;'>"
            "⚠️ FLAGGED FOR PHYSICIAN REVIEW</div>"
        )

    return f"""
    <div style="border:3px solid {color};border-radius:16px;overflow:hidden;box-shadow:0 4px 12px rgba(0,0,0,0.1);">
        <div style="background:{color};color:white;padding:20px 24px;display:flex;
                    justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:32px;font-weight:800;">{icon} {level}</span>
                <span style="font-size:18px;margin-left:16px;opacity:0.9;">{label}</span>
            </div>
            <div style="text-align:right;font-size:13px;opacity:0.85;">
                Confidence: {conf:.0%}<br>
                Model: Cairn E4B
            </div>
        </div>
        <div style="padding:20px 24px;font-size:15px;line-height:1.7;">
            {assessment_html}
        </div>
        {flagged}
    </div>
    """


def _format_safety(result: TriageCase) -> str:
    flags = result.safety_result.get("flags", [])
    if not flags:
        return (
            "<div style='background:#f0fdf4;border:2px solid #16a34a;border-radius:10px;"
            "padding:14px 18px;text-align:center;color:#16a34a;font-weight:600;font-size:14px;'>"
            "✓ All safety checks passed — no drug interactions or vital alerts</div>"
        )

    parts = []
    if result.safety_result.get("blocked"):
        parts.append(
            "<div style='background:#fef2f2;border:2px solid #dc2626;border-radius:10px;"
            "padding:14px 18px;margin-bottom:10px;'>"
            "<strong style='color:#dc2626;font-size:15px;'>⛔ SAFETY BLOCK</strong>"
            "<p style='margin:6px 0 0;color:#991b1b;'>Recommendation withheld — physician review required</p></div>"
        )
    if result.safety_result.get("emergency"):
        parts.append(
            "<div style='background:#fef2f2;border:2px solid #dc2626;border-radius:10px;"
            "padding:14px 18px;margin-bottom:10px;'>"
            "<strong style='color:#dc2626;font-size:15px;'>🚨 EMERGENCY VITAL ALERT</strong>"
            "<p style='margin:6px 0 0;color:#991b1b;'>Critical vital sign — immediate intervention required</p></div>"
        )

    for f in flags:
        sev = f["severity"]
        if sev == "EMERGENCY":
            bg, border = "#fef2f2", "#dc2626"
        elif sev == "CRITICAL":
            bg, border = "#fef2f2", "#ef4444"
        else:
            bg, border = "#fffbeb", "#d97706"
        parts.append(
            f"<div style='background:{bg};border:1px solid {border};border-radius:8px;"
            f"padding:10px 14px;margin-bottom:6px;font-size:13px;'>"
            f"<strong>{sev}</strong>: {f['detail']}</div>"
        )

    return "\n".join(parts)


def _format_details(result: TriageCase, fhir: dict) -> str:
    parts = []
    parts.append(f"**Case ID**: `{result.case_id}` | **Time**: {result.timestamp[:19]}")
    parts.append(f"**Model**: {result.cascade_result.get('model_used', 'unknown')} | "
                 f"**Flagged**: {'Yes' if result.flagged_for_human else 'No'}")

    if result.vitals:
        vitals_str = ", ".join(f"{k}: {v}" for k, v in result.vitals.items())
        parts.append(f"**Vitals**: {vitals_str}")

    if result.medications:
        parts.append(f"**Medications**: {', '.join(result.medications)}")

    parts.append(f"\n```json\n{json.dumps(fhir, indent=2)}\n```")

    return "\n\n".join(parts)


CSS = """
* { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important; }
.gradio-container { max-width: 1100px !important; margin: auto !important; }
footer { display: none !important; }
.cairn-title { text-align: center; padding: 8px 0 16px; }
.cairn-title h1 { font-size: 2.2em; margin: 0; font-weight: 700; }
.cairn-title p { color: #555; margin: 4px 0 0; font-size: 1em; }
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Cairn — Offline Medical Triage", css=CSS, theme=gr.themes.Soft()) as app:
        gr.HTML(
            "<div class='cairn-title'>"
            "<h1>Cairn</h1>"
            "<p>Offline AI triage for clinicians — powered by Gemma 4</p>"
            "</div>"
        )

        with gr.Row(equal_height=False):
            # LEFT: Input panel
            with gr.Column(scale=2):
                description = gr.Textbox(
                    label="Patient presentation",
                    placeholder="e.g. 45yo male, crushing chest pain radiating to left arm, diaphoretic, history of hypertension",
                    lines=3, autofocus=True,
                )

                medications = gr.Textbox(
                    label="Current medications (comma-separated)",
                    placeholder="e.g. warfarin, aspirin, metformin",
                    lines=1,
                )

                gr.Markdown("**Vitals** *(leave blank if not measured)*", elem_classes=["vitals-label"])
                with gr.Row():
                    spo2 = gr.Number(label="SpO2 (%)", precision=0, minimum=0, maximum=100)
                    heart_rate = gr.Number(label="Heart rate (bpm)", precision=0, minimum=0, maximum=300)
                    glucose = gr.Number(label="Glucose (mg/dL)", precision=0, minimum=0, maximum=1000)
                with gr.Row():
                    systolic = gr.Number(label="BP systolic (mmHg)", precision=0, minimum=0, maximum=300)
                    diastolic = gr.Number(label="BP diastolic (mmHg)", precision=0, minimum=0, maximum=200)
                    temperature = gr.Number(label="Temp (°C)", precision=1, minimum=30, maximum=45)

                with gr.Accordion("Attach photo (wound, rash, etc.)", open=False):
                    image = gr.Image(label="Photo", type="filepath")

                triage_btn = gr.Button("▶ Run Triage", variant="primary", size="lg")

            # RIGHT: Results panel
            with gr.Column(scale=3):
                result_html = gr.HTML(
                    value="<div style='text-align:center;padding:80px;color:#999;'>"
                    "<p style='font-size:20px;font-weight:500;'>Enter patient information to begin triage</p>"
                    "<p style='font-size:14px;'>Results will appear here</p></div>"
                )
                safety_html = gr.HTML()
                with gr.Accordion("Case details & FHIR record", open=False):
                    details_md = gr.Markdown()

        # Wire up the button and enter key
        inputs = [
            description, medications,
            spo2, heart_rate, systolic, diastolic, temperature, glucose,
            image,
        ]
        outputs = [result_html, safety_html, details_md]

        triage_btn.click(fn=run_triage, inputs=inputs, outputs=outputs)
        description.submit(fn=run_triage, inputs=inputs, outputs=outputs)

        # Demo examples
        gr.Examples(
            examples=[
                ["25-year-old, paper cut on finger. Bleeding stopped.", "", None, None, None, None, None, None, None],
                ["14-year-old skateboard fall. Forearm angulated, fingers pink. Pain 8/10.", "", None, None, None, None, None, None, None],
                ["60-year-old crushing chest pain radiating to jaw. Diaphoretic.", "warfarin, aspirin", 93, 115, 165, 100, None, None, None],
                ["62-year-old diabetic, found confused and sweaty. Slurred speech.", "insulin, metformin", None, None, None, None, None, 28, None],
                ["72-year-old shortness of breath, worsening 2 days.", "", 85, 130, None, None, None, None, None],
            ],
            inputs=inputs,
            label="Example cases (click to load)",
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
