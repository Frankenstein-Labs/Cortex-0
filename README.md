# CORTEX-0

**Small autonomous language model — Frankenstein-Labs**

CORTEX-0 is a decoder-only Transformer written from scratch in PyTorch, trained with next-token
prediction, and surrounded by a real software stack for memory, reasoning, initiative, persistent
missions and confined code execution.

It is **not** a wrapper around an external API and it does **not** pretend to be a general
assistant. The released CPU checkpoint is small (3.15M parameters); its measured quality is
reported honestly in [`docs/RESULTS.md`](docs/RESULTS.md).

```
cortex/
├── model/        RMSNorm, RoPE, grouped-query attention, SwiGLU, LM
├── tokenizer/    byte-level BPE training and chat template
├── training/     pretraining loop, memory-mapped data, SFT
├── generation/   temperature, top-k, top-p, repetition penalty
├── evaluation/   perplexity, reasoning benchmark, mechanism suite
├── reasoning/    multi-sample traces, voting, deterministic verification
├── memory/       working, episodic, semantic, preference, project memory
├── conversation/ proactive companion
├── initiative/   gated autonomous proposals
├── planning/     persistent tasks, resumable plans, checkpoints
├── coding/       confined software-engineering agent
├── execution/    process-level sandbox
├── tools/        explicit tool registry
└── api/          FastAPI service
```

## Status

| Item | Value |
| --- | --- |
| Architecture | decoder-only Transformer, RoPE, RMSNorm, GQA, SwiGLU, tied embeddings |
| Tokenizer | byte-level BPE, 8,192 tokens, trained on the corpus |
| CPU checkpoint | 3.15M parameters, 19.7M training tokens |
| Validation metrics | see [`docs/RESULTS.md`](docs/RESULTS.md) |
| Tests | `python -m pytest tests/ -q` |
| Mechanism suite | `python -m cortex.evaluation.suite` |
| Instruction tuning | short SFT run; the base model is not a chat assistant |
| Hugging Face weights | blocked: no `HF_TOKEN` in the build environment |

## Quickstart

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev,serve]"

# 1. Data (TinyStories, CDLA-Sharing-1.0)
mkdir -p data/raw
curl -sL https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/train-00000-of-00004-2d5a1467fff1081b.parquet -o data/raw/tinystories_train.parquet
curl -sL https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/validation-00000-of-00001-869c898b519ad725.parquet -o data/raw/tinystories_valid.parquet
python scripts/prepare_data.py

# 2. Train the CPU-sized model
python scripts/train.py --config configs/cortex-0-nano.json

# 3. Evaluate and sample
python -m cortex.cli --checkpoint checkpoints/cortex-0-nano/best.safetensors eval
python scripts/generate.py --prompt "Once upon a time" --samples 3

# 4. Run the API
python -m cortex.cli serve --port 8000
```

## Model presets

| Preset | Parameters | Context | Intended hardware |
| --- | ---: | ---: | --- |
| `cortex-0-nano` | 3.15M | 512 | CPU, this repository's measured run |
| `cortex-0-mini` | 6.3M | 512 | CPU (slow) or small GPU |
| `cortex-0-small` | 26.7M | 1024 | GPU (Colab T4) |
| `cortex-0-base` | 81.8M | 2048 | GPU |

GPU instructions: [`notebooks/CORTEX_0_TRAIN_GPU.ipynb`](notebooks/CORTEX_0_TRAIN_GPU.ipynb).
The notebook refuses to run without CUDA.

## API

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/generate \
  -H 'content-type: application/json' \
  -d '{"prompt":"Once upon a time","max_new_tokens":80,"temperature":0.8}'
curl -s -X POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"message":"Bonjour","session_id":"demo"}'
```

Endpoints: `/health`, `/model/info`, `/generate`, `/chat`, `/memory/*`, `/tasks/*`,
`/initiative/tick`, `/initiative/history`, `/tools`.

`uvicorn cortex.api.app:create_app --factory` also works.

## Autonomy

Autonomy is implemented as measurable software behaviour:

* **Initiative engine** — proposes a message only when enabled, outside quiet hours, after cooldown,
  under the daily cap, with sufficient novelty and score. Silence is the default.
* **Persistent tasks** — SQLite tasks and steps survive process restarts; new instructions are
  appended without discarding completed work; checkpoints allow resumption.
* **Coding agent** — reads a workspace, plans, edits files, runs allowlisted commands and tests,
  snapshots and restores. `delete`, `install`, `network` and `deploy` require approval.
* **Memory** — six tiers, TF-IDF retrieval with recency decay, consolidation of repeated
  statements. This is external memory; it never silently updates the weights.

## Tests

```bash
python -m pytest tests/ -q          # unit and integration tests
python -m cortex.evaluation.suite   # mechanism checks, JSON report
```

The suite separates `mechanism` checks from `quality` checks. A mechanism pass is not a quality
claim.

## Documentation

* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
* [`docs/RESULTS.md`](docs/RESULTS.md) — measured numbers and exact commands
* [`docs/ROADMAP.md`](docs/ROADMAP.md)
* [`docs/SECURITY.md`](docs/SECURITY.md)
* [`docs/LANGUAGES.md`](docs/LANGUAGES.md)
* [`MODEL_CARD.md`](MODEL_CARD.md)

## Limitations

* The CPU model produces simple story-like text. It does not reason reliably, follow instructions
  reliably, or know facts.
* No benchmark number is invented. Arithmetic accuracy is reported as measured, including 0%.
* The sandbox is process-level, not a security boundary.
* Soussou, Malinké and Bambara are encodable by the byte-level tokenizer but were not part of
  training. No coverage is claimed.
* The model has no safety alignment.

## License

Code: Apache-2.0. Pretraining data: TinyStories, CDLA-Sharing-1.0. See [`datasets/README.md`](datasets/README.md).
