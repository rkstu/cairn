# Cairn — Model Architecture & Technical Reference

**Last updated**: April 23, 2026
**Purpose**: Authoritative technical reference. Replication guide. Every decision, error, and fix with file:line references.

---

## 1. Problem & Evidence

Zero validated edge LLM deployments for medical triage exist. CHWs in disaster zones have no AI support.

| Study | Finding |
|-------|---------|
| Arslan et al. 2025, AJEM | LLMs 87.8% vs nurses 32.7% on critical patients |
| Haim et al. 2024, JCN | LLMs over-triage (conservative) = safer direction |
| Google AMIE 2024 | LLM > 20 physicians on 28/32 diagnostic axes |
| Gao et al. 2026, BMC EM | First meta-analysis of LLM triage accuracy |
| OnSIDES, Tanaka et al. 2025, Med | 8.5M drug-ADE pairs from NLP on drug labels |

---

## 2. Model Selection

| Model | Role | Key Specs |
|-------|------|-----------|
| Gemma 4 E4B | Screening (fine-tuned) | ~8B params, text+image+audio, 128K ctx, Apache 2.0 |
| Gemma 4 26B-A4B | Reasoning (teacher + cascade) | 3.8B active (MoE), LMArena #44, 88.3% AIME, 256K ctx |

Why two: Real EDs triage in stages (nurse → physician). E4B screens ~70% of cases. RED/uncertain escalates to 26B.

---

## 3. Model Cascade

**File**: `src/models/cascade.py`

**Flow** (`ModelCascade.triage()`, line 147):
1. E4B screens every patient → extracts triage level + confidence
2. RED or confidence < 0.7 → escalate to 26B
3. RED cases get consistency sampling (3x via `_consistency_check()`, line 116)
4. 26B unavailable → instant fallback to E4B + `flagged_for_human=True`

**Triage level extraction** (`_extract_triage_level()`, line 47): v2 regex handles negation patterns ("RED - NO, GREEN - YES"). Uses `[^,]*?` to prevent crossing comma boundaries. 7 test patterns pass.

**Confidence extraction** (`_extract_confidence()`, line 22): 5 regex patterns covering `Confidence: 0.95`, `85%`, `(0.9)`, prose. Default 0.5. 4 format tests pass.

**Model availability check** (`_is_model_available()`, line 95): Queries `client.list()` once (cached). Uses `m.model.split(":")[0]` for Ollama's `name:tag` format. Raises `ModelNotAvailableError` instantly if model not loaded. Fixed 575s hang → 19s.

---

## 4. Teacher-Student Distillation

### 4.1 Pipeline

| Step | File → Function | Output |
|------|----------------|--------|
| 75 scenarios from WHO ETAT + ESI | `scripts/generate_distillation_data.py` → `SCENARIOS` (line 53) | 20 RED / 27 YELLOW / 28 GREEN |
| Run through 26B teacher | Same → `generate_teacher_responses()` (line 174) | 75 gold-standard responses |
| Augment with vitals | Same → `augment_with_variations()` (line 236) | 130 total examples |
| Fine-tune E4B | `notebooks/cairn_finetune_A100.ipynb` | LoRA adapters |

Training data: `data/finetune_ready/cairn_distillation_train.jsonl`. Each line: `{"conversations": [system, user, assistant], "expected_level": "GREEN"}`.

GREEN-weighted distribution (28/27/20) because GREEN calibration is where the base model fails.

### 4.2 What Failed First

TimotheeB/triage-medical-dataset: SFT was medical board exams (not triage), DPO was genetics, 60% French, 0% had symptoms populated. Scripts `prepare_finetune.py` and `generate_triage_data.py` are superseded.

---

## 5. Fine-Tuning (A100 Run)

### 5.1 Infrastructure

| Item | Value |
|------|-------|
| Instance | `gpu_1x_a100_sxm4`, Lambda, asia-south-1, $1.99/hr |
| GPU | NVIDIA A100-SXM4-40GB |
| PyTorch | 2.10.0+cu128 (must pin `<2.11.0` — Unsloth compat) |
| Unsloth | 2026.4.6, Transformers 5.5.0, Xformers 0.0.35 |

### 5.2 Install Errors & Fixes

| Error | Fix | Notebook cell |
|-------|-----|---------------|
| `ModuleNotFoundError: typer` | Remove `--no-deps` from huggingface_hub install | `install-deps` |
| `torchvision>=0.25.0 required` | Pin all torch packages from same cu128 index | `install-torch` |
| `_wrap_tensor_autograd` AttributeError | Pin `torch>=2.10.0,<2.11.0` (2.11 breaks Unsloth) | `install-torch` |

Install sequence: Cell 1a (torch stack) → restart → Cell 1b (unsloth + deps) → restart → Cell 1c (verify).

### 5.3 Model Loading

`FastVisionModel.from_pretrained("unsloth/gemma-4-E4B-it", load_in_4bit=False)`

| Decision | Why |
|----------|-----|
| `FastVisionModel` not `FastLanguageModel` | E4B is multimodal. Wrong class caused 8 GGUF export failures on GH200 |
| `load_in_4bit=False` (16-bit) | Required for `save_pretrained_gguf` to merge weights cleanly |
| `ATTN_BACKEND=eager` | FA2 unsupported for Gemma 4 head dim >256 |

GPU after load: 14.9 GB / 40 GB.

### 5.4 LoRA

Rank 32, alpha 64 (2x), all attention + MLP (7 modules), `use_gradient_checkpointing="unsloth"`. Trainable: 84.8M / 8.1B (1.05%). GPU after LoRA: 15.2 GB.

### 5.5 Training

| Parameter | Value | Why (vs v1 GH200 run) |
|-----------|-------|----------------------|
| Batch | 1 | Saves VRAM, Unsloth recommendation (was 2) |
| Grad accumulation | 4 | Effective batch 4 → 32 updates/epoch (was 8 → 8 updates) |
| Epochs | 5 | 165 total steps |
| LR | 2e-4, cosine | Standard |
| Optimizer | adamw_8bit | Saves ~2GB vs full AdamW |
| Loss | 0.1992 | |
| GPU peak | 19.4 / 40 GB | 52% headroom |

### 5.6 Previous Run (GH200 — April 15-17)

8 GGUF export failures before finding `FastVisionModel` + `load_in_4bit=False` pattern. Notebook: `cairn_finetune_gpu_1x_gh200.ipynb`. Key failure modes: wrong model class, broken merge_and_unload, unmapped multimodal tensors, Ollama LoRA TODO, ClippableLinear incompatibility. Documented for posterity, not actionable.

---

## 6. Evaluation

### 6.1 Design

45 held-out cases (none in training). Categories: Clear (25), Boundary (10), Adversarial (10). Distribution: GREEN=18, YELLOW=14, RED=13. Defined in notebook cell `eval-cases`.

### 6.2 Results

| Metric | Base | Fine-Tuned | Delta |
|--------|------|------------|-------|
| **Overall** | 40/45 (89%) | **42/45 (93%)** | +4% |
| RED | 13/13 (100%) | 13/13 (100%) | — |
| YELLOW | 9/14 (64%) | 11/14 (79%) | +15% |
| GREEN | 18/18 (100%) | 18/18 (100%) | — |
| Boundary | 9/10 (90%) | 10/10 (100%) | +10% |
| Adversarial | 8/10 (80%) | 9/10 (90%) | +10% |

### 6.3 Errors

All 3 errors are YELLOW→RED over-triage (safe direction). Zero under-triage.

1. Appendicitis (RLQ, rebound, fever) → RED. Defensible: peritonitis risk.
2. Forearm fracture (angulated) → RED. Over-cautious.
3. Gout (exquisitely tender) → RED. Over-triage.

### 6.4 Live E2E Tests (Ollama, April 22)

19 scenarios via `TriageAgent.run()`: GREEN 4/4, YELLOW 3/3, RED 4/4, Drug BLOCK 3/3, Emergency vitals 2/2, Edge 1/3. **17/19 pass.** 2 failures are conservative defaults on ambiguous input.

### 6.5 Gradio E2E (April 22)

4/4 pass via `gradio_client`: paper cut → GREEN, fracture → YELLOW, chest pain + drugs → RED + BLOCKED, hypoglycemia → RED + EMERGENCY.

---

## 7. GGUF Export & Deployment

Export on A100 after freeing training state (15.2 GB used, 24.3 GB free). `save_pretrained_gguf("cairn-gguf", tokenizer, quantization_method="q4_k_m")`. GPU peak: 15.4 GB. Output in `cairn-gguf_gguf/` (Unsloth appends `_gguf`). OOM recovery cell in notebook for fallback.

### Published Artifacts

| Artifact | Location | Size |
|----------|----------|------|
| LoRA | `huggingface.co/lightmate/cairn-gemma4-e4b-triage` | 354 MB |
| GGUF Q4_K_M | `huggingface.co/lightmate/cairn-gemma4-e4b-triage-gguf` | 5.0 GB |
| GGUF Q8_0 | same repo | 7.7 GB |
| mmproj | same repo | 946 MB |
| Training data | `kaggle.com/datasets/rahulkumar99/cairn-triage-distillation` | 130 examples |

### Ollama Deployment

Modelfile at `models/gguf/CairnModelfile`. Key settings: `num_predict 1024` (Gemma 4 thinking mode fills budget otherwise), `temperature 0.1`, `stop <turn|>`.

HF verification: 3/3 PASS (GREEN, YELLOW, RED) from fresh LoRA load.

---

## 8. Safety Gate

**File**: `src/safety/gate.py` → `SafetyGate.check()` (line 44)

Runs AFTER LLM, BEFORE user. No neural network. Logic: BLOCK (high-severity drug interaction), WARN (moderate interaction or high-alert ADE), EMERGENCY (critical vitals → override to RED).

### 8.1 Vital Bounds

**File**: `src/safety/vital_check.py` → `check_vitals()`. Thresholds in `src/config.py` → `VitalBounds`.

SpO2 < 90 → EMERGENCY. HR > 150 or < 40 → EMERGENCY. Systolic > 180 → EMERGENCY. Diastolic > 120 → EMERGENCY. Temp > 40 or < 35 → CRITICAL. Glucose > 400 or < 54 → EMERGENCY.

### 8.2 Drug Safety

**File**: `src/safety/drug_check.py` → `DrugSafetyChecker`

**101 drug-drug interactions** in `scripts/download_onsides.py` → `_seed_critical_interactions()`. Categories: anticoagulant (17), NSAID (13), cardiovascular (13), statin (7), serotonergic (11), diabetes (5), renal (7), antibiotics (7), CNS (8), immunosuppressant (4), QT (5), thyroid (4). Sources: FDA labels, ISMP, WHO.

**8,477,894 adverse effect records** from OnSIDES v3.1.0 (1,831 drugs, 7,177 MedDRA terms). Downloaded from `github.com/tatonetti-lab/onsides/releases/download/v3.1.0/onsides-v3.1.0.zip` (313 MB). Loaded via custom ETL joining 5 CSV tables (label → product → ingredient → ADE).

**Alarm fatigue prevention**: `HIGH_SIGNAL_ADES` set (35 terms: hemorrhage, rhabdomyolysis, serotonin syndrome, etc.) filters noise. `check_all()` only shows individual ADEs for high-alert meds (warfarin, opioids, insulin, etc.). Drug-drug interactions checked for all med pairs.

### 8.3 Tests

27 total: `test_safety.py` (16 vital bounds), `test_gate.py` (5 integrated), `test_store.py` (6 SQLite).

---

## 9. Agentic Tool Calling

**File**: `src/agent/executor.py` → `ToolExecutor.run_agentic_loop()` (line 57)

6 tools in `src/agent/tools.py`. Model decides which to call. Loop: model → tool_calls → execute → feed back. Max 3 iterations.

| Tool | Source | Status |
|------|--------|--------|
| `read_pulse_oximeter` | BLE GATT 0x1822 | Mock |
| `read_blood_pressure` | BLE GATT 0x1810 | Mock |
| `read_glucometer` | BLE GATT 0x1808 | Mock |
| `check_drug_interactions` | OnSIDES SQLite | **Live** |
| `queue_escalation_report` | Local SQLite | Live |
| `scan_medical_devices` | BLE scan | Mock |

BLE mocks via `register_mock_device()`. Real BLE code in `src/devices/ble_reader.py` (scaffolded, using `bleak` with standard GATT profiles).

---

## 10. UI & User Experience

**File**: `src/app.py` → `build_app()`

### Design Principles

CHW in a disaster zone: type symptoms, get answer. Everything else secondary.

**Default view**: 1 text box ("What's wrong with the patient?") + 1 button ("Triage"). Enter key submits.

**Behind accordions** (closed by default):
- "Add vitals, medications, or photo" — 6 vital fields, medications text, image upload
- "Advanced settings" — agentic toggle, mock devices
- "Details & FHIR" — case metadata, FHIR JSON

**Output**: Color-coded HTML triage banner (RED/YELLOW/GREEN). Safety alerts as styled cards. Assessment rendered as HTML, not raw text. `gr.themes.Soft()` for clean look. Gradio footer hidden.

### Thread Safety Fix

Gradio runs handlers in worker threads. `LocalStore` and `DrugSafetyChecker` open SQLite connections at module load (main thread). Fix: `check_same_thread=False` on both `sqlite3.connect()` calls (`store.py:21`, `drug_check.py:28`).

---

## 11. Offline Architecture

| Component | File |
|-----------|------|
| Inference | `src/models/cascade.py` — local Ollama |
| Drug safety | `data/onsides.db` — 571 MB SQLite |
| Case storage | `src/sync/store.py` — SQLite with sync flags |
| Sync | `src/sync/remote.py` — CouchDB push |
| Interop | `src/agent/triage.py` → `to_fhir_encounter()` — FHIR R4 |

---

## 12. Reproduction

1. Launch `gpu_1x_a100_sxm4` on Lambda
2. Upload `cairn_distillation_train.jsonl` + notebook
3. Run install cells with 2 kernel restarts (pin torch <2.11)
4. Run all cells: base eval → train → ft eval → GGUF export → HF push
5. Locally: `ollama create cairn-e4b-triage -f models/gguf/CairnModelfile`
6. Locally: `python scripts/download_onsides.py`
7. Locally: `.venv/bin/python3 -m src.app` → `localhost:7860`
