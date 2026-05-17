"""Cairn CLI — entry point for the triage agent."""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel

from src.agent.triage import TriageAgent
from src.config import CairnConfig

app = typer.Typer(name="cairn", help="Cairn: Offline Medical Triage Agent")
console = Console()


@app.command()
def triage(
    description: str = typer.Argument(..., help="Patient description"),
    image: str | None = typer.Option(None, "--image", "-i", help="Path to wound/condition image"),
    spo2: int | None = typer.Option(None, help="SpO2 percentage"),
    heart_rate: int | None = typer.Option(None, "--hr", help="Heart rate bpm"),
    systolic: int | None = typer.Option(None, "--bp-sys", help="Systolic BP mmHg"),
    diastolic: int | None = typer.Option(None, "--bp-dia", help="Diastolic BP mmHg"),
    glucose: int | None = typer.Option(None, help="Blood glucose mg/dL"),
    temperature: float | None = typer.Option(None, "--temp", help="Temperature Celsius"),
    medications: str | None = typer.Option(None, "--meds", help="Comma-separated medication list"),
    ollama_url: str = typer.Option("http://localhost:11434", "--ollama-url"),
):
    """Run a triage assessment on a patient case."""
    config = CairnConfig(ollama_base_url=ollama_url)
    agent = TriageAgent(config)

    vitals = {}
    if spo2 is not None:
        vitals["spo2"] = spo2
    if heart_rate is not None:
        vitals["heart_rate"] = heart_rate
    if systolic is not None:
        vitals["systolic_bp"] = systolic
    if diastolic is not None:
        vitals["diastolic_bp"] = diastolic
    if glucose is not None:
        vitals["glucose"] = glucose
    if temperature is not None:
        vitals["temperature"] = temperature

    meds = [m.strip() for m in medications.split(",")] if medications else []

    console.print(Panel("[bold]Cairn Triage Agent[/bold]", style="blue"))
    console.print(f"Patient: {description[:100]}...")
    if vitals:
        console.print(f"Vitals: {json.dumps(vitals)}")
    if meds:
        console.print(f"Medications: {', '.join(meds)}")

    with console.status("Running triage cascade..."):
        result = agent.run(
            patient_description=description,
            vitals=vitals or None,
            medications=meds or None,
            image_path=image,
        )

    # Color-coded triage level
    color_map = {"RED": "red", "YELLOW": "yellow", "GREEN": "green"}
    level_color = color_map.get(result.triage_level, "white")

    console.print()
    console.print(Panel(
        result.final_output,
        title=f"[bold {level_color}]TRIAGE: {result.triage_level}[/bold {level_color}]",
        subtitle=f"Case {result.case_id} | {result.cascade_result.get('model_used', 'unknown')}",
        border_style=level_color,
    ))

    if result.flagged_for_human:
        console.print("[bold red]This case requires physician review.[/bold red]")


@app.command()
def scan_devices():
    """Scan for nearby BLE medical devices."""
    import asyncio

    from src.devices.ble_reader import scan_medical_devices

    console.print("Scanning for BLE medical devices...")
    devices = asyncio.run(scan_medical_devices())
    if not devices:
        console.print("No medical devices found.")
        return
    for d in devices:
        console.print(f"  {d['type']}: {d['name']} ({d['address']})")


@app.command()
def version():
    """Show Cairn version."""
    console.print("Cairn v0.1.0 — Offline Medical Triage Agent powered by Gemma 4")


if __name__ == "__main__":
    app()
