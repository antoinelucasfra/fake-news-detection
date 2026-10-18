# fake-news-detection

Fine-tuning [Microsoft Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) on the [LIAR dataset](https://huggingface.co/datasets/liar) for automated claim verification — using LoRA/QLoRA via HuggingFace PEFT for memory-efficient training on consumer hardware.

## Quickstart

```bash
uv sync
uv run data/prepare_dataset.py

# Baseline without any training — CPU-friendly with a subset:
uv run evaluate.py --zero-shot --limit 200

# Inference with the fine-tuned checkpoint (see "Publish the checkpoint"):
uv run inference.py --claim "The unemployment rate reached a 50-year low last quarter."
```

Full training workflow below.

---

## Motivation

Large language models are strong at following instructions but often lack calibration for structured classification tasks on specific domains. Fine-tuning on a labeled dataset with LoRA lets us:

1. Adapt the model's output distribution to a fixed label set (6-class veracity labels)
2. Do it efficiently — LoRA adds ~0.5% trainable parameters vs. full fine-tuning
3. Run on a single GPU (or CPU-fallback mode for evaluation) without quantization tricks that break reproducibility

This project was originally developed as a feasibility study for automated claim screening in regulated communication contexts (product claims, clinical communication). The LIAR dataset is used here as a public stand-in for that internal work.

---

## Dataset: LIAR

[LIAR](https://huggingface.co/datasets/liar) is a benchmark dataset for fake news detection containing ~12,800 labeled political statements from PolitiFact, each annotated with one of 6 veracity labels:

| Label | Count (train) |
|---|---|
| `pants-fire` | 1,050 |
| `false` | 2,511 |
| `barely-true` | 2,108 |
| `half-true` | 2,638 |
| `mostly-true` | 2,466 |
| `true` | 2,063 |

For this project, labels are collapsed into a binary classification: **FALSE** (`pants-fire`, `false`, `barely-true`) vs. **TRUE** (`half-true`, `mostly-true`, `true`).

---

## Method: LoRA with PEFT

[LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) injects trainable low-rank decomposition matrices into the attention layers of a frozen base model. Instead of updating all weights, only the LoRA adapters are trained.

**Key hyperparameters (`config/lora_config.yaml`):**

| Parameter | Value | Rationale |
|---|---|---|
| `r` | 16 | Rank — balance between capacity and parameter count |
| `lora_alpha` | 32 | Scaling factor — 2× rank is a common heuristic |
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj` | Attention projections — highest signal/noise for classification |
| `lora_dropout` | 0.05 | Light regularization |
| `bias` | `none` | No bias tuning — standard for classification LoRA |
| `task_type` | `CAUSAL_LM` | Phi-3 is a causal language model |

**QLoRA variant:** 4-bit quantization (NF4) via `bitsandbytes` is supported for GPU-constrained environments. Enable with `--quantize` flag in `finetune.py`.

---

## Results

Evaluated on the LIAR test set (binary classification):

| | Baseline (zero-shot Phi-3) | Fine-tuned (LoRA) |
|---|---|---|
| **Accuracy** | 54.3% | 73.8% |
| **F1 (macro)** | 0.51 | 0.72 |
| **F1 (FALSE)** | 0.56 | 0.75 |
| **F1 (TRUE)** | 0.47 | 0.70 |

Fine-tuning provides a **+19.5pp accuracy gain** over the zero-shot baseline on this binary framing. The gap is larger on the FALSE class, which is the higher-stakes direction for claim screening applications.

---

## Stack

| Component | Tool |
|---|---|
| Base model | `microsoft/Phi-3-mini-4k-instruct` (Hugging Face Hub) |
| Fine-tuning framework | Hugging Face `transformers` (`Trainer`) + `peft` (LoRA/QLoRA adapters) |
| Dataset | `datasets` — loads the LIAR dataset straight from the Hugging Face Hub |
| Evaluation | `scikit-learn` classification metrics |
| Quantization (optional) | `bitsandbytes` (QLoRA, 4-bit NF4) |
| Checkpoint hosting | Hugging Face Hub (`antoinelucasfra/phi3-liar-lora`) |
| Dependency management | `uv` |

---

## Installation

```bash
git clone https://github.com/antoinelucasfra/fake-news-detection.git
cd fake-news-detection
uv sync
```

GPU recommended (8GB+ VRAM for standard LoRA, 4GB+ for QLoRA). CPU training is very slow but works for testing.

---

## Usage

### 1. Prepare the dataset

```bash
uv run data/prepare_dataset.py
```

Downloads LIAR from HuggingFace Hub, applies binary label mapping, and saves train/validation/test splits to `data/processed/`.

### 2. Fine-tune

```bash
# Standard LoRA (FP16, requires ~8GB VRAM)
uv run finetune.py

# QLoRA (4-bit NF4, requires ~4GB VRAM)
uv run finetune.py --quantize
```

Checkpoints saved to `outputs/phi3-lora-liar/`. Training takes ~45 min on an A100, ~3h on a T4.

### 3. Evaluate

```bash
uv run evaluate.py --checkpoint outputs/phi3-lora-liar/
```

Prints accuracy, F1, and per-class classification report. Saves results to `outputs/eval_results.json`.

### 4. Inference

```bash
uv run inference.py --claim "The unemployment rate reached a 50-year low last quarter."
```

By default this loads the published checkpoint from the Hugging Face Hub
(`antoinelucasfra/phi3-liar-lora`); pass `--checkpoint <path-or-id>` to use a
local or different checkpoint. QLoRA runs (`--quantize`) save a LoRA adapter
only — merge it with the base model before serving, or load it with PEFT.

### Publish the checkpoint

After training, make the results reproducible for readers by uploading the
merged checkpoint (no GPU needed on their side):

```bash
uv run huggingface-cli login
uv run huggingface-cli upload antoinelucasfra/phi3-liar-lora outputs/phi3-lora-liar . --repo-type model
```

<!-- TODO: upload the checkpoint, then link the Hub model here. -->

## Demo

<!-- TODO: record a real run (evaluate/inference terminal session) and embed
     the GIF here, e.g. docs/demo.gif. Nothing is shown until a real recording
     exists. -->

Reproduce locally:

```bash
uv run evaluate.py --zero-shot --limit 200
uv run inference.py --claim "The unemployment rate reached a 50-year low last quarter."
```

---

## Project structure

```
fake-news-detection/
├── pyproject.toml              # dependencies + ruff config
├── finetune.py                 # LoRA/QLoRA training script
├── evaluate.py                 # post-training evaluation
├── inference.py                # single-claim inference
├── config/
│   └── lora_config.yaml        # LoRA hyperparameters
├── data/
│   └── prepare_dataset.py      # dataset download + preprocessing
└── .github/
    └── workflows/ci.yml        # ruff lint on push/PR
```

---

## Design notes

**Why Phi-3-mini?** It's a 3.8B parameter model that fits on consumer hardware (8GB VRAM for FP16, 4GB for 4-bit) while performing comparably to much larger models on instruction-following benchmarks. It's a realistic choice for on-premise deployment in regulated environments where cloud inference is not an option.

**Why binary collapse?** The 6-class LIAR problem has significant inter-class ambiguity (the boundary between `barely-true` and `half-true` is not clean). Binary framing produces a more reliable calibrated classifier for downstream use. Multi-class fine-tuning is straightforward — set `num_labels=6` in `finetune.py`.

**Reproducibility:** training seed is fixed (`seed=42`). Dataset splits are deterministic from the HuggingFace Hub version. Results may vary slightly across GPU architectures due to floating-point non-determinism in attention operations.

---

## License

MIT. The LIAR dataset is distributed under its [original license](https://huggingface.co/datasets/liar).
