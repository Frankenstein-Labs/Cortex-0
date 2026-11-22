# CORTEX-0 roadmap and status

Status values: **done** (implemented and tested), **measured** (a real run produced numbers),
**partial** (scaffolding exists, quality not demonstrated), **blocked** (missing resource).

| Phase | Scope | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Scientific design | done | `docs/ARCHITECTURE.md` |
| 2 | GitHub repository | done | `Frankenstein-Labs/Cortex-0` |
| 2b | Hugging Face repository | blocked | no `HF_TOKEN` in this environment |
| 3 | Transformer architecture | done | `cortex/model`, model tests |
| 4 | Tokenizer + autoregressive prediction | done | trained 8,192-token BPE, sampling tests |
| 5 | Pretraining pipeline | done + measured | CPU run, `checkpoints/cortex-0-nano` |
| 6 | First tests and benchmarks | done | `python -m cortex.evaluation.suite` |
| 7 | Fine-tuning and reasoning | partial | SFT pipeline + reasoning engine; short SFT run |
| 8 | Memory and conversation | done | SQLite tiers, companion, tests |
| 9 | Autonomy and missions | done | initiative engine, persistent tasks, tests |
| 10 | Specialised skills | partial | arithmetic verifier; general skills need a larger model |
| 11 | API and experimental interface | done | FastAPI, CLI, Docker |
| 12 | Global evaluation, docs, publication | done + measured | README, model card, results |

## What is deliberately not claimed

* The 3.15M-parameter CPU model is not an assistant and does not reason reliably.
* No benchmark score is invented. Arithmetic accuracy is reported as measured, including 0%.
* Autonomy is software behaviour with gates and logs, not consciousness.
* General skills (science, law, business, African languages) require data and compute that are not
  available in this environment. The tokenizer is language-agnostic; coverage is not claimed.

## Next steps (GPU required)

1. Train `cortex-0-small` (26.7M) or `cortex-0-base` (81.8M) with the Colab notebook.
2. Train a larger tokenizer (16k–32k) on a multilingual corpus including Soussou, Malinké and Bambara.
3. Run supervised fine-tuning on curated reasoning traces.
4. Implement GRPO on verifiable arithmetic/code tasks; compare against the SFT baseline.
5. Publish weights and tokenizer to Hugging Face with a model card.
6. Add an embedding-based memory index once a sentence encoder is available.
