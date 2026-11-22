# Bootstrap scaffold (archived)

This directory preserves the first `CORTEX-0` commit
(`33616f7 feat: bootstrap from-scratch causal language model`) unchanged.

It is a minimal decoder-only prototype with:

- `CortexLM` / `CortexConfig`;
- a UTF-8 byte tokenizer;
- a causal text dataset and a simple training loop;
- three smoke tests.

It is **superseded** by the package at the repository root, which provides the
Llama-compatible model, BPE tokenizer, pretraining/SFT pipelines, evaluation,
memory, reasoning, autonomy and API layers. The scaffold is kept for history
and comparison only. Its modules are not imported by the active package.
