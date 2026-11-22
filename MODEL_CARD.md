# Model card — CORTEX-0 nano (rewrite branch)

## Model details

* **Name:** CORTEX-0 nano
* **Organization:** Frankenstein-Labs
* **Type:** decoder-only Transformer language model, trained from scratch
* **Parameters:** 3,147,456 (tied input/output embeddings)
* **Layers:** 4, `d_model` 192, 6 query heads, 2 KV heads, SwiGLU hidden 512
* **Context:** 512 tokens
* **Tokenizer:** byte-level BPE, 8,192 tokens, trained on the corpus
* **Checkpoint:** step 500 of a planned 1,200-step CPU run
* **License:** Apache-2.0 (code and weights); data is CDLA-Sharing-1.0

## Intended use

Research on small language models, training pipelines, sampling, memory, reasoning scaffolding and
autonomous software behaviour. It can continue a short story prompt and demonstrate that the full
pipeline (tokenizer, pretraining, checkpointing, evaluation, API) works end to end.

## Out-of-scope use

* General-purpose assistant or chatbot.
* Factual question answering.
* Arithmetic, mathematics, code generation or logical reasoning.
* Safety-critical, medical, legal or financial decisions.
* Any claim of human-like consciousness or general intelligence.

## Training data

TinyStories (`roneneldan/TinyStories`), CDLA-Sharing-1.0. 120,000 training stories and 3,000
validation stories were prepared. The model saw 8.192M tokens (500 steps × 16,384 tokens), not the
full 26.68M-token training split.

## Evaluation

Measured on 40 validation batches (163,840 tokens):

| Metric | Value |
| --- | ---: |
| Loss | 2.5435 |
| Perplexity | 12.7246 |
| Bits per token | 3.6695 |
| Token accuracy | 45.21% |

Procedural arithmetic benchmark: 0/10 correct for direct, chain-of-thought, self-consistency and
verified strategies. This is a real negative result and is reported as such.

Mechanism suite: 10/10 checks passed (shapes, causality, KV cache, sampler, synthetic training,
memory, task resume, initiative gates, sandbox, verifier). Mechanism checks are not quality claims.

## Limitations

* Produces locally fluent story text that drifts logically; names and objects are inconsistent.
* No instruction tuning on this checkpoint; it is a base next-token model.
* No safety alignment. It can generate inappropriate text.
* Trained only on English children's stories.
* Not evaluated on any external benchmark. No MMLU, GSM8K or similar score is claimed.
* The reasoning layer is software around the model; it cannot compensate for missing model
  knowledge.

## Ethical considerations

The model is a research artifact. Do not deploy it to users without content filtering, monitoring
and a clear statement of its limits. The initiative engine can send unprompted messages; it is
disabled by default in tests and must be configured with quiet hours, cooldown and a daily cap.

## Reproduction

See [`docs/RESULTS.md`](docs/RESULTS.md) for exact commands, hardware and metrics. The checkpoint
file is not committed; the training recipe and tokenizer pipeline are.

## Citation

```
@misc{cortex0_2026,
  title  = {CORTEX-0: a small from-scratch autonomous language model},
  author = {Frankenstein-Labs},
  year   = {2026},
  url    = {https://github.com/Frankenstein-Labs/Cortex-0}
}
```
