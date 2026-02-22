"""
evaluate.py — Evaluate a fine-tuned Phi-3 LoRA checkpoint on the LIAR test set.

Loads the checkpoint produced by finetune.py, runs inference on the test split,
and prints a full classification report. Saves results to outputs/eval_results.json.

Usage:
    uv run evaluate.py --checkpoint outputs/phi3-lora-liar/
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_config_from_checkpoint(checkpoint_dir: str) -> tuple[dict, dict]:
    """Load label mappings from the saved model config."""
    import json

    config_path = Path(checkpoint_dir) / "config.json"
    with open(config_path) as f:
        cfg = json.load(f)
    return cfg.get("label2id", {"FALSE": 0, "TRUE": 1}), cfg.get(
        "id2label", {"0": "FALSE", "1": "TRUE"}
    )


def predict(
    checkpoint_dir: str,
    statements: list[str],
    batch_size: int = 16,
    max_length: int = 256,
) -> np.ndarray:
    """Run batch inference and return predicted label indices."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_dir,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    all_preds = []
    for i in range(0, len(statements), batch_size):
        batch = statements[i : i + batch_size]
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())

        if (i // batch_size) % 5 == 0:
            print(f"  {min(i + batch_size, len(statements))}/{len(statements)} examples processed")

    return np.array(all_preds)


def main(checkpoint_dir: str) -> None:
    print(f"Loading test split from data/processed/test.csv")
    df = pd.read_csv("data/processed/test.csv")

    label2id, id2label = load_config_from_checkpoint(checkpoint_dir)
    label2id_int = {k: int(v) for k, v in label2id.items()}

    statements = df["statement"].tolist()
    true_labels = df["label_binary"].map(label2id_int).tolist()

    print(f"Running inference on {len(statements)} test examples...")
    pred_labels = predict(checkpoint_dir, statements)

    # ── Metrics ────────────────────────────────────────────────────────────
    acc = accuracy_score(true_labels, pred_labels)
    f1_macro = f1_score(true_labels, pred_labels, average="macro")
    f1_false = f1_score(true_labels, pred_labels, pos_label=0, average="binary")
    f1_true = f1_score(true_labels, pred_labels, pos_label=1, average="binary")
    cm = confusion_matrix(true_labels, pred_labels).tolist()
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=["FALSE", "TRUE"],
        output_dict=True,
    )

    print("\n" + "=" * 50)
    print("Evaluation results (LIAR test set, binary)")
    print("=" * 50)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  F1 macro  : {f1_macro:.4f}")
    print(f"  F1 FALSE  : {f1_false:.4f}")
    print(f"  F1 TRUE   : {f1_true:.4f}")
    print("\nClassification report:")
    print(
        classification_report(true_labels, pred_labels, target_names=["FALSE", "TRUE"])
    )
    print(f"Confusion matrix:\n{confusion_matrix(true_labels, pred_labels)}")

    # ── Save results ───────────────────────────────────────────────────────
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    results = {
        "checkpoint": checkpoint_dir,
        "n_test": len(statements),
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_false": f1_false,
        "f1_true": f1_true,
        "confusion_matrix": cm,
        "classification_report": report,
    }
    out_path = output_dir / "eval_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Phi-3 on LIAR test set")
    parser.add_argument(
        "--checkpoint",
        default="outputs/phi3-lora-liar/",
        help="Path to the fine-tuned model checkpoint",
    )
    args = parser.parse_args()
    main(args.checkpoint)
