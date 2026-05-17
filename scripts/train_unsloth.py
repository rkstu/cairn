"""Fine-tune Gemma 4 E4B on triage data using Unsloth.

Run on:
- M4 Pro 48GB (MLX/CPU — slower but free)
- Kaggle T4 16GB (fits E4B QLoRA ~12-14GB)
- Nebius L40S 48GB (for 26B-A4B, $1.55/hr)

Usage:
    python scripts/train_unsloth.py --model gemma4-e4b --epochs 3 --lr 2e-4
    python scripts/train_unsloth.py --model gemma4-e4b --dpo  # DPO alignment
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def train_sft(
    model_name: str = "unsloth/gemma-4-4b-it-bnb-4bit",
    data_path: str = "./data/finetune_ready/triage_sft.jsonl",
    output_dir: str = "./models/cairn-e4b-triage",
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 2,
    max_seq_length: int = 2048,
    lora_rank: int = 16,
):
    """Supervised fine-tuning with Unsloth QLoRA."""
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    print(f"Loading model: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Load data
    print(f"Loading data from {data_path}")
    rows = []
    with open(data_path) as f:
        for line in f:
            rows.append(json.loads(line))

    # Format for Unsloth chat template
    def format_conversation(row):
        messages = row["conversations"]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    dataset = Dataset.from_list(rows)
    dataset = dataset.map(format_conversation)
    print(f"Dataset: {len(dataset)} examples")

    # Train
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            seed=42,
        ),
        dataset_text_field="text",
        max_seq_length=max_seq_length,
    )

    print("Starting SFT training...")
    stats = trainer.train()
    print(f"Training complete. Loss: {stats.training_loss:.4f}")

    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    # Export to GGUF for Ollama
    gguf_path = output_dir + "-gguf"
    print(f"Exporting to GGUF at {gguf_path}...")
    model.save_pretrained_gguf(gguf_path, tokenizer, quantization_method="q4_k_m")
    print(f"GGUF exported to {gguf_path}")

    return stats


def train_dpo(
    model_name: str = "unsloth/gemma-4-4b-it-bnb-4bit",
    sft_model_path: str = "./models/cairn-e4b-triage",
    data_path: str = "./data/finetune_ready/triage_dpo.jsonl",
    output_dir: str = "./models/cairn-e4b-triage-dpo",
    epochs: int = 1,
    lr: float = 5e-5,
    max_seq_length: int = 2048,
    lora_rank: int = 16,
):
    """DPO preference alignment after SFT."""
    from unsloth import FastLanguageModel, PatchDPOTrainer
    from trl import DPOTrainer, DPOConfig
    from datasets import Dataset

    PatchDPOTrainer()

    # Load SFT model if available, else base
    load_path = sft_model_path if Path(sft_model_path).exists() else model_name
    print(f"Loading model: {load_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Load DPO data
    print(f"Loading DPO data from {data_path}")
    rows = []
    with open(data_path) as f:
        for line in f:
            rows.append(json.loads(line))
    dataset = Dataset.from_list(rows)
    print(f"DPO dataset: {len(dataset)} pairs")

    # Train DPO
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Unsloth handles this
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=DPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            beta=0.1,
            seed=42,
        ),
    )

    print("Starting DPO training...")
    stats = trainer.train()
    print(f"DPO complete. Loss: {stats.training_loss:.4f}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 for triage")
    parser.add_argument("--model", default="unsloth/gemma-4-4b-it-bnb-4bit")
    parser.add_argument("--dpo", action="store_true", help="Run DPO instead of SFT")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output", default="./models/cairn-e4b-triage")
    args = parser.parse_args()

    if args.dpo:
        train_dpo(model_name=args.model, output_dir=args.output + "-dpo", epochs=args.epochs, lr=args.lr)
    else:
        train_sft(model_name=args.model, output_dir=args.output, epochs=args.epochs, lr=args.lr)
