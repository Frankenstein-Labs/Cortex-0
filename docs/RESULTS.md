# CORTEX-0 measured results — rewrite branch

This branch (`rewrite/tested-core`) is an **alternative from-scratch implementation**. The
repository's `main` branch contains a pre-existing, more feature-complete implementation that was
imported in parallel and has been restored. See [`BRANCH_NOTICE.md`](../BRANCH_NOTICE.md).

Every number below comes from a run performed in this workspace. Nothing is estimated or copied
from another model.

## Environment

| Item | Value |
| --- | --- |
| CPU | 4 vCPU (container) |
| RAM | ~15 GB |
| GPU | none |
| Disk | ~26 GB |
| OS / Python | Linux, Python 3.13 |
| PyTorch | 2.15.0+cpu |
| Training device | CPU, float32 |

## Data

TinyStories (`roneneldan/TinyStories`, CDLA-Sharing-1.0), prepared by `scripts/prepare_data.py`:

| Split | Stories | Characters | Tokens |
| --- | ---: | ---: | ---: |
| train | 120,000 | 108,003,433 | 26,681,760 |
| validation | 3,000 | 2,622,556 | 644,887 |

Tokenizer: byte-level BPE, 8,192 tokens, trained on 20,000 stories. Characters per token: 4.0483.
The tokenizer file is reproducible but not committed (554 KB); `scripts/prepare_data.py` recreates
it deterministically from the same parquet files and seed.

## Model

| Item | Value |
| --- | --- |
| Parameters | 3,147,456 |
| Layers | 4 |
| `d_model` | 192 |
| Heads / KV heads | 6 / 2 (GQA) |
| FFN | SwiGLU, hidden 512 |
| Context | 512 |
| Positional encoding | RoPE |
| Norm | RMSNorm (pre-norm) |
| Embeddings | tied input/output |
| Dropout | 0.0 |

## Pretraining

Command:

```bash
python scripts/train.py --config configs/cortex-0-nano.json
```

Configuration: batch 32, gradient accumulation 2, sequence length 256 (16,384 tokens/step),
AdamW, lr 2e-3 with 100 warmup steps and cosine decay, weight decay 0.1, grad clip 1.0, seed 1337.

The configured run is 1,200 steps (19.66M tokens). It was **stopped at step 500 (8.192M tokens)**
to keep the CPU budget bounded. The best checkpoint is the step-500 checkpoint.

| Step | Tokens seen | Train loss | Val loss | Val perplexity | Val token accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16,384 | 8.9911 | — | — | — |
| 100 | 1,638,400 | 4.0101 | 3.8061 | 44.97 | 30.90% |
| 200 | 3,276,800 | 3.1365 | 3.0872 | 21.92 | 38.90% |
| 300 | 4,915,200 | 2.8472 | 2.8107 | 16.62 | 42.15% |
| 400 | 6,553,600 | 2.6548 | 2.6543 | 14.22 | 44.06% |
| 500 | 8,192,000 | 2.5306 | 2.5389 | 12.67 | 45.32% |

Throughput: 3,700–5,160 tokens/s on 4 vCPU. Step 500 was reached in 1,742 s of training plus
evaluation time.

Independent evaluation with 40 batches (`python -m cortex.cli ... eval --max-batches 40`):

| Metric | Value |
| --- | ---: |
| Loss | 2.5435 |
| Perplexity | 12.7246 |
| Bits per token | 3.6695 |
| Token accuracy | 45.21% |
| Batches / tokens | 40 / 163,840 |

The 30-batch training-time validation gives loss 2.5389 and perplexity 12.6652; the small
difference is the different evaluation window.

## Samples

Checkpoint: step 500. Prompt: `Once upon a time`. Temperature 0.8, top-k 50, top-p 0.95, seed 42.

> there was a little girl called Lily. She loved to draw and wear her crayons. One day, she found a
> big box in her backyard. She started to put the box in a box. It was a toy box and Lily was really
> happy to have it back. After her mom put her box in the box, she showed it to her mom. "Wow, this
> toy is very cool and she looks like a toy. Can I go outside?"

> , there was a little boy named Timmy. He loved to play outside in the forest. One day, he saw a
> pretty butterfly in the grass. The butterfly flew down and fell down. Timmy's mom saw her and
> said, "Don't worry, Timmy. Can you help me clean up?" Timmy nodded and said, "No, we can't be
> scared. The butterfly might get sick." The butterfly flew away and Timmy was sad.

> there were two friends, Joe and Joe. Joe and Dan and his friends were playing together in the
> woods. Bob had a big, old house. Joe had a big, fluffy blanket and he was very proud of himself.

The text is locally fluent and story-shaped, but it drifts logically and repeats names. This is the
expected quality of a 3.1M-parameter model trained on 8.2M tokens.

## Reasoning benchmark

Procedural arithmetic, 10 problems, 2 samples per problem, 48 new tokens, seed 7/1234:

| Strategy | Correct | Accuracy |
| --- | ---: | ---: |
| direct | 0 / 10 | 0.00 |
| cot | 0 / 10 | 0.00 |
| self_consistency | 0 / 10 | 0.00 |
| verified | 0 / 10 | 0.00 |

This is a real negative result. The model has not learned to answer arithmetic in the required
format, and the reasoning layer cannot create knowledge the weights do not contain. The verifier
itself works (see the mechanism suite); it simply has nothing correct to select.

## Mechanism suite

`python -m cortex.evaluation.suite` → **10 / 10** checks passed in 1.66 s:

* model shapes, causality, KV-cache equivalence (max diff 3.58e-07)
* sampler support and seeded reproducibility
* training decreases loss on a learnable synthetic task (4.207 → 0.184 in 40 steps)
* memory retrieval/facts/preferences
* task resume across a new process with an appended instruction
* initiative gates (disabled, quiet hours, cooldown)
* sandbox path escape and executable allowlist
* arithmetic verifier accepts a correct equation and rejects a wrong one

These are **mechanism** checks, not quality claims.

## Tests

`python -m pytest tests/ -q` → **71 passed** on a fresh clone of the branch.

## Not run / blocked

* No GPU run: `cortex-0-small` (26.7M) and `cortex-0-base` (81.8M) are not trained.
* No SFT run on this checkpoint: the SFT pipeline exists and is unit-tested, but the released
  checkpoint is the base pretrained model.
* No GRPO run on this branch.
* Hugging Face publication: blocked, no `HF_TOKEN` in the environment.
* Weights are not committed: the 13 MB safetensors checkpoint stays local. The repository contains
  the exact recipe and measured metrics; the tokenizer is regenerated by the data script.

## Reproduction

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,serve]"
mkdir -p data/raw
curl -sL https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/train-00000-of-00004-2d5a1467fff1081b.parquet -o data/raw/tinystories_train.parquet
curl -sL https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/validation-00000-of-00001-869c898b519ad725.parquet -o data/raw/tinystories_valid.parquet
python scripts/prepare_data.py
python scripts/train.py --config configs/cortex-0-nano.json --steps 500
python -m cortex.cli --checkpoint checkpoints/cortex-0-nano/best.safetensors eval --max-batches 40
python scripts/generate.py --prompt "Once upon a time" --samples 3
```

Exact hardware and library versions affect throughput and, marginally, floating-point results.

## Relation to `main`

`main` contains a different implementation with a 4,096-token tokenizer, a 2.56M-parameter model and
its own measured validation perplexity of 37.44. **The two perplexities are not directly
comparable** because the tokenizers and evaluation corpora differ. No claim of superiority is made.
This branch is preserved for review and was not merged into `main` without authorization.
