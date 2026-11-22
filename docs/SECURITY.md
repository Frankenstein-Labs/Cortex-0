# Security model

## Secrets

* No token, key or credential is written to the repository.
* The API and engine read configuration from environment variables.
* The GPU notebook reads the Hugging Face token from Colab secrets at runtime.

## Execution sandbox

`cortex/execution/sandbox.py` provides:

* workspace-confined path resolution (`..` and absolute escapes rejected);
* no shell: commands are `argv` lists;
* executable allowlist (`python`, `pytest`, `git`, read-only file utilities);
* timeout, output cap and a scrubbed environment;
* blocked argument patterns for destructive commands.

**Limitation:** this is a process-level guard. It does not isolate CPU, memory, filesystem
permissions or network at kernel level. Do not run untrusted code with it. Use a container or VM
for that.

## Sensitive actions

The coding agent requires explicit approval for `delete`, `install`, `network` and `deploy`.
`ApprovalPolicy(auto_approve=...)` is the only way to allow them, and it is opt-in per action.

## Model safety

* The model is trained on children's stories. It can produce grammatical but nonsensical or
  inappropriate text; it has no safety alignment.
* Do not deploy it as a user-facing assistant without additional filtering and evaluation.
* The initiative engine can be disabled with `CORTEX_INITIATIVE_ENABLED=0`; quiet hours, cooldown
  and daily caps are enforced in code.

## Data

* Dataset provenance and license are recorded in `datasets/README.md` and `data/processed/meta.json`.
* Raw datasets are not committed.
