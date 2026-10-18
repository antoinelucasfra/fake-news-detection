"""
inference.py — Single-claim inference with a fine-tuned Phi-3 LoRA checkpoint.

Loads the checkpoint produced by finetune.py and classifies a claim as
FALSE or TRUE with a confidence score.

Usage:
    uv run inference.py --claim "The unemployment rate reached a 50-year low."
    uv run inference.py --interactive   # interactive REPL mode
"""

import argparse

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# chisle: placeholder Hub id — update after uploading the checkpoint (see README);
# --checkpoint overrides for local dirs.
DEFAULT_CHECKPOINT = "antoinelucasfra/phi3-liar-lora"
MAX_LENGTH = 256


def load_model(checkpoint_dir: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
            "at a local trained checkpoint, publish the checkpoint to the HF Hub (see "
            "README), or run evaluate.py --zero-shot for the no-checkpoint baseline."
        ) from e
    model.eval()
    return model, tokenizer, device


def classify(claim: str, model, tokenizer, device: str) -> dict:
    """
    Classify a single claim. Returns label, confidence, and raw probabilities.
    """
    inputs = tokenizer(
        claim,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
    pred_id = int(torch.argmax(logits, dim=-1).item())
    id2label = model.config.id2label

    return {
        "label": id2label[pred_id],
        "confidence": round(probs[pred_id], 4),
        "probabilities": {id2label[i]: round(p, 4) for i, p in enumerate(probs)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a claim with fine-tuned Phi-3")
    parser.add_argument("--claim", type=str, default=None, help="Claim text to classify")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive REPL mode",
    )
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model, tokenizer, device = load_model(args.checkpoint)
    print(f"Model loaded (device: {device})\n")

    if args.interactive:
        print("Interactive mode — type a claim and press Enter. Ctrl+C to quit.\n")
        while True:
            try:
                claim = input("Claim: ").strip()
                if not claim:
                    continue
                result = classify(claim, model, tokenizer, device)
                print(f"  Label      : {result['label']}")
                print(f"  Confidence : {result['confidence']:.2%}")
                probs = result["probabilities"]
                print(f"  Probs      : FALSE={probs['FALSE']:.3f}  TRUE={probs['TRUE']:.3f}")
                print()
            except KeyboardInterrupt:
                print("\nExiting.")
                break
    elif args.claim:
        result = classify(args.claim, model, tokenizer, device)
        print(f"Claim      : {args.claim}")
        print(f"Label      : {result['label']}")
        print(f"Confidence : {result['confidence']:.2%}")
        probs = result["probabilities"]
        print(f"Probs      : FALSE={probs['FALSE']:.3f}  TRUE={probs['TRUE']:.3f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
