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
import polars as pl
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

BASE_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
# chisle: placeholder Hub id — update after uploading the checkpoint (see README)
DEFAULT_CHECKPOINT = "antoinelucasfra/phi3-liar-lora"


def load_config_from_checkpoint(checkpoint_dir: str) -> tuple[dict, dict]:
    """Load label mappings from the saved model config (local dir or Hub id)."""
    cfg = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
    return cfg.label2id, cfg.id2label


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
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
    except OSError as e:
        raise SystemExit(
            f"Checkpoint '{checkpoint_dir}' could not be loaded. Either point --checkpoint "
            "at a local trained checkpoint, publish it to the HF Hub (see README), or "
            "run evaluate.py --zero-shot for the no-checkpoint baseline."
        ) from e
    model.eval()

    all_preds = []
    for i in range(0, len(statements), batch_size):
        batch = statements[i : i + batch_size]
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
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


def parse_true_false(answer: str) -> int:
    """Parse a generated answer into 1 (TRUE) / 0 (FALSE).

    Compares the first word only — substring matching would read "NOT TRUE"
    or "UNTRUE" as TRUE. Ambiguous/empty → FALSE (0).
    """
    words = answer.upper().split()
    first = words[0].strip(".,!\"'") if words else ""
    return 1 if first == "TRUE" else 0


def predict_zero_shot(
    model_name: str,
    statements: list[str],
    batch_size: int = 8,
    max_length: int = 256,
    max_new_tokens: int = 4,
) -> np.ndarray:
    """
    Zero-shot baseline: prompt the raw (non-fine-tuned) model to answer TRUE or
    FALSE. Runs without any checkpoint — usable on CPU for the baseline column
    of the results table. Prompt wording is approximate; scores are not directly
    comparable to the fine-tuned classifier.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    preds = []
    for i in range(0, len(statements), batch_size):
        batch = statements[i : i + batch_size]
        prompts = [
            f'Statement: "{s}"\nQuestion: Is this statement TRUE or FALSE? Answer with one word.'
            for s in batch
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        for seq in output:
            # chisle: parse first word only — see parse_true_false docstring
            answer = tokenizer.decode(seq[inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            preds.append(parse_true_false(answer))
        print(f"  {min(i + batch_size, len(statements))}/{len(statements)} examples processed")

    return np.array(preds)


def main(
    checkpoint_dir: str,
    zero_shot: bool = False,
    model_name: str = BASE_MODEL_NAME,
    limit: int | None = None,
) -> None:
    print("Loading test split from data/processed/test.csv")
    df = pl.read_csv("data/processed/test.csv")
    if limit:
        df = df.head(limit)
        print(f"Limited to first {limit} test examples.")

    if zero_shot:
        label2id_int = {"FALSE": 0, "TRUE": 1}
    else:
        label2id, id2label = load_config_from_checkpoint(checkpoint_dir)
        label2id_int = {k: int(v) for k, v in label2id.items()}

    statements = df["statement"].to_list()
    true_labels = df["label_binary"].replace_strict(label2id_int).to_list()

    print(f"Running inference on {len(statements)} test examples...")
    if zero_shot:
        pred_labels = predict_zero_shot(model_name, statements)
    else:
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
    print(classification_report(true_labels, pred_labels, target_names=["FALSE", "TRUE"]))
    print(f"Confusion matrix:\n{confusion_matrix(true_labels, pred_labels)}")

    # ── Save results ───────────────────────────────────────────────────────
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    results = {
        "checkpoint": checkpoint_dir,
        "mode": "zero_shot" if zero_shot else "fine_tuned",
        "model": model_name if zero_shot else checkpoint_dir,
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
        default=DEFAULT_CHECKPOINT,
        help="Path or Hub id of the fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--zero-shot",
        action="store_true",
        help="Evaluate the non-fine-tuned base model (no checkpoint needed)",
    )
    parser.add_argument(
        "--model",
        default=BASE_MODEL_NAME,
        help="Base model used with --zero-shot",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N test examples (CPU-friendly)",
    )
    args = parser.parse_args()
    main(args.checkpoint, zero_shot=args.zero_shot, model_name=args.model, limit=args.limit)
