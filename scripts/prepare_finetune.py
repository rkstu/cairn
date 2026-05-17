"""Prepare fine-tuning data for Gemma 4 E4B using Unsloth.

Converts TimotheeB/triage-medical-dataset into Gemma chat format
with triage-specific system prompts. Targets GREEN/YELLOW calibration.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset, load_from_disk

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "finetune_ready"

TRIAGE_SYSTEM_PROMPT = (
    "You are Cairn, an offline medical triage assistant for community health workers. "
    "Assess the patient and provide:\n"
    "1. Triage level: RED (immediate life threat), YELLOW (urgent, needs prompt care), "
    "or GREEN (non-urgent, can wait)\n"
    "2. Assessment: brief clinical reasoning\n"
    "3. Recommendations: actionable next steps for a community health worker\n"
    "4. Confidence: 0.0 to 1.0\n"
    "Be calibrated: minor injuries and stable conditions should be GREEN, not YELLOW. "
    "Only use YELLOW for conditions requiring prompt medical attention within hours. "
    "RED is reserved for immediately life-threatening conditions.\n"
    "If uncertain, say so explicitly. Never guess on medication dosages."
)


def prepare_sft_data():
    """Convert SFT triage data to Gemma chat format."""
    sft_path = DATA_DIR / "triage_sft"
    if not sft_path.exists():
        print(f"SFT data not found at {sft_path}. Run dataset download first.")
        return

    ds = load_from_disk(str(sft_path))
    print(f"Loaded {len(ds)} SFT examples")
    print(f"Columns: {ds.column_names}")

    conversations = []
    for row in ds:
        conv = {"conversations": [{"role": "system", "content": TRIAGE_SYSTEM_PROMPT}]}

        if "instruction" in row and row.get("instruction") and "response" in row and row.get("response"):
            # TimotheeB format: instruction/response with metadata
            user_content = row["instruction"]
            # Enrich with structured data if available
            if row.get("symptoms"):
                user_content += f"\nSymptoms: {row['symptoms']}"
            if row.get("history"):
                user_content += f"\nHistory: {row['history']}"
            if row.get("vitals"):
                user_content += f"\nVitals: {row['vitals']}"
            conv["conversations"].append({"role": "user", "content": user_content})
            conv["conversations"].append({"role": "assistant", "content": row["response"]})
        elif "messages" in row and row["messages"]:
            for msg in row["messages"]:
                if msg.get("role") in ("user", "assistant"):
                    conv["conversations"].append({"role": msg["role"], "content": msg["content"]})
        elif "prompt" in row and "chosen" in row:
            conv["conversations"].append({"role": "user", "content": row["prompt"]})
            chosen = row["chosen"]
            if isinstance(chosen, list):
                for msg in chosen:
                    if msg.get("role") == "assistant":
                        conv["conversations"].append({"role": "assistant", "content": msg["content"]})
            elif isinstance(chosen, str):
                conv["conversations"].append({"role": "assistant", "content": chosen})
        elif "question" in row and "answer" in row:
            conv["conversations"].append({"role": "user", "content": row["question"]})
            conv["conversations"].append({"role": "assistant", "content": row["answer"]})

        roles = [c["role"] for c in conv["conversations"]]
        if "user" in roles and "assistant" in roles:
            conversations.append(conv)

    print(f"Prepared {len(conversations)} conversations")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "triage_sft.jsonl"
    with open(output_path, "w") as f:
        for conv in conversations:
            f.write(json.dumps(conv) + "\n")

    ds_out = Dataset.from_list(conversations)
    ds_out.save_to_disk(str(OUTPUT_DIR / "triage_sft_dataset"))
    print(f"Saved {len(conversations)} conversations to {output_path}")
    return conversations


def prepare_dpo_data():
    """Convert DPO triage data for preference alignment."""
    dpo_path = DATA_DIR / "triage_dpo"
    if not dpo_path.exists():
        print(f"DPO data not found at {dpo_path}.")
        return

    ds = load_from_disk(str(dpo_path))
    print(f"Loaded {len(ds)} DPO examples")

    dpo_rows = []
    for row in ds:
        prompt = row.get("prompt", "")
        chosen = row.get("chosen", "")
        rejected = row.get("rejected", "")
        if not prompt or not chosen or not rejected:
            continue

        if isinstance(chosen, list):
            chosen = " ".join(m.get("content", "") for m in chosen if m.get("role") == "assistant")
        if isinstance(rejected, list):
            rejected = " ".join(m.get("content", "") for m in rejected if m.get("role") == "assistant")

        dpo_rows.append({
            "prompt": TRIAGE_SYSTEM_PROMPT + "\n\nPatient: " + prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

    print(f"Prepared {len(dpo_rows)} DPO pairs")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "triage_dpo.jsonl"
    with open(output_path, "w") as f:
        for row in dpo_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(dpo_rows)} DPO pairs to {output_path}")
    return dpo_rows


if __name__ == "__main__":
    print("=== Preparing SFT data ===")
    prepare_sft_data()
    print()
    print("=== Preparing DPO data ===")
    prepare_dpo_data()
