# AGENTS.md — CORTEX-0

Instructions for agents working in this repository.

## Project

CORTEX-0 is a from-scratch, Llama-compatible small language model plus an
autonomy layer (memory, planning, initiative, companion, coding agent, API).
The trained reference model is 2,557,632 parameters, CPU-only.

## Non-negotiable rules

1. **Never claim a result that did not run.** If a training job, benchmark or
   evaluation was not executed, say so and leave the value absent or `null`.
2. **Never present an untrained checkpoint as a model.** A checkpoint is only
   `trained` after real optimization steps.
3. **Never invent benchmarks or numbers.** Report measured values and the
   command/artifact that produced them (`experiments/*.json`).
4. **Never commit secrets.** No `HF_TOKEN`, `GITHUB_TOKEN`, `.env`, API keys.
5. **Do not delete existing work.** `legacy/bootstrap/` is archived and must
   stay untouched.
6. **Sensitive actions require approval** (production deploys, destructive
   commands, writes to `.env`/secrets/workflows).

## Commands

```bash
python -m pytest -q
python scripts/prepare_data.py --config configs/tiny-cpu.yaml
python scripts/train.py --config configs/tiny-cpu.yaml
python scripts/evaluate.py --model-dir checkpoints/tiny-cpu/best --val-data datasets/processed/val.bin --n 20
python scripts/generate.py --model-dir checkpoints/tiny-cpu --prompt "CORTEX est"
python scripts/serve.py --model-dir checkpoints/tiny-cpu --port 8000
python scripts/export_hf.py --model-dir checkpoints/tiny-cpu --out export/Cortex-0 --verify
```

The unified CLI is `cortex` (`cortex --help`).

## Layout

* `cortex/model` — Llama-compatible decoder, attention, generation.
* `cortex/tokenizer` — byte-level BPE, chat template.
* `cortex/training` — data, trainer, SFT, synthetic corpora.
* `cortex/reasoning` — verifier, best-of-N, GRPO.
* `cortex/evaluation` — perplexity, internal reasoning suite.
* `cortex/memory`, `cortex/planning`, `cortex/initiative`,
  `cortex/conversation` — autonomy layer.
* `cortex/agents`, `cortex/coding`, `cortex/execution` — coding agent and queue.
* `cortex/api` — FastAPI service.
* `legacy/bootstrap` — archived prototype; do not import or modify.

## Conventions

* Python 3.10+, type hints, dataclasses where useful.
* Keep the stack free/open; no paid infrastructure required.
* Add or update tests for every behavior change.
* Keep comments for non-obvious invariants only.
* Update `docs/` and `experiments/` when a measured result changes.

## Verification before claiming done

1. `python -m pytest -q` passes.
2. `python -m compileall -q cortex scripts tests` passes.
3. Any new training/eval number has an artifact under `experiments/`.
4. `git status` shows no secret or generated binary.
