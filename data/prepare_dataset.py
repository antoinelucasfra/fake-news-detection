"""
data/prepare_dataset.py — Download and preprocess the LIAR dataset.

Downloads LIAR from the HuggingFace Hub, applies binary label mapping,
and saves train/validation/test splits to data/processed/.

Usage:
    uv run data/prepare_dataset.py

Output:
    data/processed/train.csv
    data/processed/validation.csv
    data/processed/test.csv
"""

from pathlib import Path

import polars as pl
from datasets import load_dataset

# Binary label mapping
# FALSE: pants-fire, false, barely-true  (id < 3)
# TRUE:  half-true, mostly-true, true    (id >= 3)
LIAR_LABELS = {
    0: "pants-fire",
    1: "false",
    2: "barely-true",
    3: "half-true",
    4: "mostly-true",
    5: "true",
}

BINARY_MAP = {
    "pants-fire": "FALSE",
    "false": "FALSE",
    "barely-true": "FALSE",
    "half-true": "TRUE",
    "mostly-true": "TRUE",
    "true": "TRUE",
}


def prepare() -> None:
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading LIAR dataset from HuggingFace Hub...")
    dataset = load_dataset("liar")

    for split_name in ["train", "validation", "test"]:
        split = dataset[split_name]
        df = pl.DataFrame(split.to_dict())

        # LIAR label column is integer 0–5; map to string then binary
        df = df.with_columns(
            pl.col("label").replace_strict(LIAR_LABELS).alias("label_str"),
            pl.col("label_str").replace_strict(BINARY_MAP).alias("label_binary"),
        )

        # Keep only the columns we need
        out = df.select("statement", "label_str", "label_binary").drop_nulls(
            ["statement", "label_binary"]
        )

        out_path = output_dir / f"{split_name}.csv"
        out.write_csv(out_path)

        n_false = (out["label_binary"] == "FALSE").sum()
        n_true = (out["label_binary"] == "TRUE").sum()
        print(f"  {split_name}: {len(out)} examples | FALSE={n_false} TRUE={n_true} → {out_path}")

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    prepare()
