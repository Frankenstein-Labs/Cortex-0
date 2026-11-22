# Branch notice

This branch, `rewrite/tested-core`, is an **alternative implementation** of CORTEX-0.

While this rewrite was being prepared, the repository's `main` branch received a separate, more
feature-complete implementation (agents, missions, scheduler, GRPO, Hugging Face export, API
streaming). That work was imported in parallel and was not visible in the initial workspace
snapshot.

To avoid destroying it, `main` was restored exactly to the pre-existing tree (commit `942c0e0`)
after this branch was created. This branch is preserved for review and comparison. It was **not
merged into `main`**, because replacing an existing implementation requires the owner's
authorization.

What this branch contains:

* a tested from-scratch core: Transformer, BPE tokenizer, pretraining, sampling, evaluation;
* memory, reasoning, initiative, persistent tasks, coding agent, API, CLI, Docker, CI;
* a real 3.15M-parameter CPU pretraining run: 500 steps, 8.192M tokens, validation perplexity
  12.72 on its own 8,192-token tokenizer;
* 71 passing tests and a 10/10 mechanism suite.

What `main` contains:

* the pre-existing full implementation, 113 files, 89 tests, its own measured results;
* a 4,096-token tokenizer and a 2.56M-parameter model with validation perplexity 37.44.

The two perplexities are **not directly comparable** (different tokenizers and corpora).

See [`docs/RESULTS.md`](docs/RESULTS.md) and [`MODEL_CARD.md`](MODEL_CARD.md) for this branch's
measured numbers. To merge, review both implementations and authorize the merge explicitly.
