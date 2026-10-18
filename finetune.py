"""
finetune.py — Fine-tune Phi-3-mini-4k-instruct on LIAR (binary) using LoRA/QLoRA.

Trains a LoRA adapter on top of the frozen Phi-3 base model.
All hyperparameters are read from config/lora_config.yaml.

Usage:
    uv run finetune.py              # standard LoRA (FP16, ~8GB VRAM)
    uv run finetune.py --quantize   # QLoRA (4-bit NF4, ~4GB VRAM)

Requirements:
    - GPU strongly recommended (CUDA); CPU training is very slow
    - Run data/prepare_dataset.py first to generate data/processed/
"""

import argparse

import numpy as np
import polars as pl
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def load_config(config_path: str = "config/lora_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_split(split: str) -> Dataset:
    df = pl.read_csv(f"data/processed/{split}.csv")
    return Dataset.from_dict(df.select("statement", "label_binary").to_dict(as_series=False))


def build_label_mappings(cfg: dict) -> tuple[dict, dict]:
    label2id = cfg["data"]["label2id"]
    id2label = {v: k for k, v in label2id.items()}
    return label2id, id2label


def tokenize_dataset(dataset: Dataset, tokenizer, max_length: int) -> Dataset:
    def tokenize(batch):
        return tokenizer(
            batch["statement"],
            truncation=True,
            max_length=max_length,
        )

    # chisle: dynamic padding via DataCollatorWithPadding — no max_length padding

    return dataset.map(tokenize, batched=True, remove_columns=["statement"])


def encode_labels(dataset: Dataset, label2id: dict) -> Dataset:
    def map_labels(batch):
        return {"labels": [label2id[lb] for lb in batch["label_binary"]]}

    return dataset.map(map_labels, batched=True, remove_columns=["label_binary"])


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "f1_false": f1_score(labels, predictions, pos_label=0, average="binary"),
        "f1_true": f1_score(labels, predictions, pos_label=1, average="binary"),
    }


def main(quantize: bool = False) -> None:
    cfg = load_config()
    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    label2id, id2label = build_label_mappings(cfg)

    print(f"Loading tokenizer: {model_cfg['name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name"],
        trust_remote_code=model_cfg["trust_remote_code"],
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Model loading ──────────────────────────────────────────────────────
    print(f"Loading model: {model_cfg['name']}  (quantize={quantize})")

    model_kwargs: dict = {
        "num_labels": data_cfg["num_labels"],
        "label2id": label2id,
        "id2label": id2label,
        "trust_remote_code": model_cfg["trust_remote_code"],
    }

    if quantize:
        from transformers import BitsAndBytesConfig

        qlora_cfg = cfg["qlora"]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=qlora_cfg["load_in_4bit"],
            bnb_4bit_quant_type=qlora_cfg["bnb_4bit_quant_type"],
            bnb_4bit_compute_dtype=getattr(torch, qlora_cfg["bnb_4bit_compute_dtype"]),
            bnb_4bit_use_double_quant=qlora_cfg["bnb_4bit_use_double_quant"],
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = getattr(torch, model_cfg["torch_dtype"])

    model = AutoModelForSequenceClassification.from_pretrained(model_cfg["name"], **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id

    if quantize:
        model = prepare_model_for_kbit_training(model)

    # ── LoRA adapter ───────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ────────────────────────────────────────────────────────────
    print("Loading and tokenizing dataset...")
    train_ds = encode_labels(
        tokenize_dataset(load_split("train"), tokenizer, data_cfg["max_seq_length"]),
        label2id,
    )
    val_ds = encode_labels(
        tokenize_dataset(load_split("validation"), tokenizer, data_cfg["max_seq_length"]),
        label2id,
    )

    # ── Training ───────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        fp16=train_cfg["fp16"] and torch.cuda.is_available(),
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        seed=train_cfg["seed"],
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving best model to {train_cfg['output_dir']}")
    if quantize:
        # 4-bit base cannot be merged in memory — save the adapter only;
        # load it later on top of the base model (see README).
        model.save_pretrained(train_cfg["output_dir"])
        print("QLoRA run: LoRA adapter saved (not merged with the quantized base).")
    else:
        # Merge adapter into the base weights so the checkpoint loads directly
        # with AutoModelForSequenceClassification.from_pretrained(dir).
        merged = model.merge_and_unload()
        merged.save_pretrained(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Phi-3 on LIAR with LoRA")
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Use QLoRA (4-bit NF4) for memory-constrained environments",
    )
    args = parser.parse_args()
    main(quantize=args.quantize)
