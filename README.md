# Cairn

**Offline AI Medical Triage — powered by Gemma 4**

*For 5,000 years, cairns have guided travelers through fog and darkness. Now, Cairn guides clinicians when the network goes dark.*

---

## The Problem

When internet fails — in rural clinics, disaster zones, or under-resourced hospitals — every AI tool becomes useless. Clinicians face life-or-death triage decisions with zero support. Studies show AI triage achieves 88%+ accuracy where conventional field triage achieves 33% ([Arslan et al., AJEM 2025](https://doi.org/10.1016/j.ajem.2024.12.001)). The capability exists. The deployment gap is the problem.

## What Cairn Does

Cairn runs **entirely offline** via [Ollama](https://ollama.com/). No internet. No cloud. No API keys. Describe a patient → get a RED/YELLOW/GREEN triage decision in seconds. Medications are checked against **8.5 million drug safety records**. Every case is stored locally in FHIR format for later sync.

### Key Numbers

| Metric | Value |
|--------|-------|
| Overall accuracy | **93%** (42/45 cases) |
| RED (critical) recall | **100%** (13/13) |
| Under-triage | **0** |
| Error direction | All over-triage (safe) |
| Drug interactions DB | 101 critical pairs |
| Adverse effects DB | 8,477,894 entries |
| Inference time | ~10-15s per case |
| Model size | 5.3 GB (GGUF Q4_K_M) |

## Live Demo

**[Try Cairn](https://huggingface.co/spaces/echotruth/cairn-triage)** — Hosted on Hugging Face Spaces (T4 GPU, may take 2-3 min to wake from sleep).

## Quick Start

```bash
git clone https://github.com/rkstu/cairn.git && cd cairn
pip install -e .

# Download fine-tuned model
huggingface-cli download lightmate/cairn-gemma4-e4b-triage-gguf \
  --include "*.gguf" --local-dir ./models/gguf
ollama create cairn-e4b-triage -f models/gguf/CairnModelfile

# Build drug safety database
python scripts/download_onsides.py

# Run
python -m src.app    # Web UI at localhost:7860
```

Turn off WiFi. It still works. That's the point.

## Architecture

```
Patient Input (text + optional vitals + medications + image)
         │
    ┌────▼────────┐
    │  Gemma 4    │  Fine-tuned E4B (93%, 5.3GB GGUF, ~10-15s)
    │  E4B Screen │  Teacher-student distillation via Unsloth QLoRA
    └────┬────────┘
         │ RED or confidence < 0.7
    ┌────▼────────┐
    │  Gemma 4    │  26B-A4B with consistency sampling (3x)
    │  26B Reason │  Graceful fallback if absent → flags for human review
    └────┬────────┘
         │
    ┌────▼──────────────────────┐
    │  DETERMINISTIC SAFETY     │  No LLM in the safety path
    │  GATE                     │
    │  • 7 vital thresholds     │  SpO2, HR, BP, Temp, Glucose
    │  • 101 drug interactions  │  BLOCK on critical combos
    │  • 8.5M adverse effects   │  OnSIDES v3.1.0
    └────┬──────────────────────┘
         │
    ┌────▼────────┐
    │  Output     │  Triage level + assessment + FHIR R4
    │  + Store    │  SQLite local → CouchDB when online
    └─────────────┘
```

### Why Two Models?

Real emergency departments triage in stages: nurse screens → physician reviews. Cairn mirrors this:
- **E4B** handles ~85% of cases autonomously (GREEN + clear YELLOW)
- **26B** provides deep reasoning for RED and uncertain cases
- If 26B is unavailable (fully offline), E4B still works — uncertain cases get **flagged for physician review** when connectivity returns

### Why a Deterministic Safety Gate?

LLMs achieve only **38% accuracy on drug interaction tasks** ([RxSafeBench, 2024](https://arxiv.org/abs/2511.04328)). Cairn never uses LLM reasoning for drug safety. The safety gate is deterministic — hard-coded rules that BLOCK dangerous recommendations regardless of what the model says.

## Fine-Tuning

**Method**: Teacher-student distillation. Gemma 4 26B-A4B generates gold-standard triage assessments for 75 WHO/ESI scenarios → fine-tune E4B with Unsloth QLoRA on 130 examples.

**Training**: 1x A100 40GB SXM4, ~10 min, 165 steps, loss 0.1992.

| Metric | Base E4B | Fine-Tuned |
|--------|----------|------------|
| Overall | 89% | **93%** |
| RED recall | 100% | **100%** |
| YELLOW recall | 64% | **79%** (+15pp) |
| GREEN recall | 100% | **100%** |

### Published Artifacts

| Artifact | Link |
|----------|------|
| LoRA weights | [lightmate/cairn-gemma4-e4b-triage](https://huggingface.co/lightmate/cairn-gemma4-e4b-triage) |
| GGUF (Q4_K_M + Q8) | [lightmate/cairn-gemma4-e4b-triage-gguf](https://huggingface.co/lightmate/cairn-gemma4-e4b-triage-gguf) |
| Training data | [rahulkumar99/cairn-triage-distillation](https://kaggle.com/datasets/rahulkumar99/cairn-triage-distillation) |
| Training notebook | `notebooks/cairn_finetune_A100.ipynb` |

## Gemma 4 Features Used

| Feature | How Cairn Uses It |
|---------|-------------------|
| **Post-training / QLoRA** | Domain-adapted for medical triage (93% accuracy) |
| **Native function calling** | 6 tools — model autonomously reads BLE devices, queries drug DB |
| **Multimodal (vision)** | Image input for wound/rash assessment in cascade |
| **E4B for edge** | 5.3GB GGUF, runs on any laptop without internet |
| **26B for reasoning** | Deep clinical reasoning with consistency sampling |

## Drug Safety

- **101 curated critical drug-drug interactions** (warfarin+NSAIDs, SSRIs+tramadol, amiodarone+statins, etc.)
- **8,477,894 adverse effect records** from OnSIDES v3.1.0 (FDA NLP-extracted)
- **High-signal filter** prevents alarm fatigue — only life-threatening ADEs shown
- **BLOCK action** on critical interactions — recommendation withheld for physician review

## Agentic Tool Calling

Cairn uses Gemma 4's native function calling with 6 tools:

1. `read_pulse_oximeter` — BLE SpO2 + pulse rate
2. `read_blood_pressure` — BLE systolic/diastolic/MAP
3. `read_glucometer` — BLE blood glucose
4. `check_drug_interactions` — Offline OnSIDES query
5. `queue_escalation_report` — FHIR report for remote sync
6. `scan_medical_devices` — BLE device discovery

The model decides which tools to call based on patient context.

## Project Structure

```
cairn/
├── src/
│   ├── config.py           # Models, thresholds, vital bounds
│   ├── app.py              # Gradio UI
│   ├── agent/
│   │   ├── triage.py       # Full triage pipeline
│   │   ├── executor.py     # Agentic tool loop
│   │   └── tools.py        # 6 native FC tool definitions
│   ├── models/
│   │   └── cascade.py      # E4B→26B with consistency sampling
│   ├── safety/
│   │   ├── vital_check.py  # 7 deterministic vital thresholds
│   │   ├── drug_check.py   # 101 interactions + 8.5M ADEs
│   │   └── gate.py         # BLOCK/WARN/EMERGENCY logic
│   └── sync/
│       ├── store.py        # SQLite (thread-safe, FHIR)
│       └── remote.py       # CouchDB sync when online
├── tests/                  # 27 unit tests
├── notebooks/              # Training notebook (executed, all outputs)
├── scripts/                # Data download, distillation generation
├── data/                   # OnSIDES DB (571MB), training examples
└── models/gguf/            # Q4_K_M GGUF + mmproj + Modelfile
```

## Tests

```bash
pytest tests/ -v    # 27 passed in 0.04s
```

Comprehensive E2E testing covers:
- All triage levels (GREEN/YELLOW/RED)
- All vital threshold boundaries (SpO2, HR, BP, Temp, Glucose)
- Drug interaction detection (8 critical pairs verified)
- Edge cases (empty input, non-medical text, contradictory vitals, multilingual)
- Safety gate override (deterministic vitals always override LLM)
- Gradio API integration

## Vision

Cairn is a proof-of-concept demonstrating that **clinically meaningful AI triage can run entirely offline** using open-weight models. The gap between this demo and field deployment is 18-24 months of primarily non-technical work — government permissions, IRB approval, CHW co-design, pilot studies, and regulatory pathway.

The technical architecture (cascade + deterministic safety gate + offline sync) is designed to scale down to smaller models as they mature, and to integrate with existing health information systems via FHIR R4.

## Competition

[Gemma 4 Good Hackathon](https://kaggle.com/competitions/gemma-4-good-hackathon) ($200K prize pool)

Eligible tracks: Main, Global Resilience, Health & Sciences, Safety & Trust, Ollama, Unsloth, llama.cpp

## License

Apache 2.0
