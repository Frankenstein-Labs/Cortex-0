#!/usr/bin/env python3
"""Export a CORTEX-0 checkpoint to Hugging Face ``transformers`` format.

Tensor names already match ``LlamaForCausalLM``; this script writes the HF
config, a ``PreTrainedTokenizerFast`` and a safetensors file, then optionally
verifies numerical equivalence against the original implementation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from safetensors.torch import save_file

from cortex.training.trainer import load_pretrained


def hf_config(config) -> dict:
    import transformers

    return {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "hidden_act": config.hidden_act,
        "max_position_embeddings": config.max_position_embeddings,
        "initializer_range": config.initializer_range,
        "rms_norm_eps": config.rms_norm_eps,
        "use_cache": True,
        "pad_token_id": config.pad_token_id,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "tie_word_embeddings": config.tie_word_embeddings,
        "rope_theta": config.rope_theta,
        "attention_bias": False,
        "attention_dropout": config.attention_dropout,
        "torch_dtype": "float32",
        "transformers_version": transformers.__version__,
    }


def export_tokenizer(model_dir: Path, out_dir: Path) -> None:
    from transformers import PreTrainedTokenizerFast

    source = model_dir / "tokenizer" / "tokenizer.json"
    if not source.exists():
        raise FileNotFoundError(source)
    fast = PreTrainedTokenizerFast(
        tokenizer_file=str(source),
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        additional_special_tokens=["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"],
    )
    fast.save_pretrained(str(out_dir))


def model_card(config, args) -> str:
    return f"""---
language:
  - fr
  - en
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
  - cortex-0
  - frankenstein-labs
  - from-scratch
---

# CORTEX-0

CORTEX-0 is a from-scratch, Llama-compatible decoder-only language model built by
Frankenstein-Labs. This repository contains exported weights for a checkpoint
trained on CPU. It is an early research model, not an instruction-following
assistant and not a replacement for larger models.

## Architecture

| Field | Value |
| --- | --- |
| Parameters | {config.num_parameters:,} |
| Layers | {config.num_hidden_layers} |
| Hidden size | {config.hidden_size} |
| Attention heads | {config.num_attention_heads} |
| KV heads | {config.num_key_value_heads} |
| Context | {config.max_position_embeddings} |
| Vocab | {config.vocab_size} |
| Norm | RMSNorm |
| Activation | SwiGLU ({config.hidden_act}) |
| Positional | RoPE (theta={config.rope_theta}) |
| Tied embeddings | {config.tie_word_embeddings} |

## Usage

```python
from transformers import AutoTokenizer, LlamaForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{args.repo_id or 'Frankenstein-Labs/Cortex-0'}")
model = LlamaForCausalLM.from_pretrained("{args.repo_id or 'Frankenstein-Labs/Cortex-0'}")
inputs = tokenizer("CORTEX est", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.8, top_p=0.95)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## Training data and limitations

The model was pretrained on a small public-domain / permissively licensed corpus
(French and English). It has limited knowledge, weak reasoning, and may produce
incorrect or repetitive text. The internal evaluation suite is not MMLU, GSM8K or
any external benchmark; measured numbers are published in the source repository
with the exact commit and checkpoint.

## Source

Code, training scripts, evaluation and safety notes:
`https://github.com/Frankenstein-Labs/Cortex-0`

Built by Frankenstein-Labs. CORTEX-0 — building autonomous intelligence.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="checkpoints/tiny-cpu")
    parser.add_argument("--out", default="export/Cortex-0")
    parser.add_argument("--repo-id", default="Frankenstein-Labs/Cortex-0")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, config = load_pretrained(model_dir)

    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    if config.tie_word_embeddings:
        state["lm_head.weight"] = (
            model.model.embed_tokens.weight.detach().cpu().contiguous().clone()
        )
    save_file(state, str(out_dir / "model.safetensors"))
    (out_dir / "config.json").write_text(
        json.dumps(hf_config(config), indent=2) + "\n", encoding="utf-8"
    )
    export_tokenizer(model_dir, out_dir)
    (out_dir / "README.md").write_text(model_card(config, args), encoding="utf-8")
    for name in ("generation_config.json",):
        source = model_dir / name
        if source.exists():
            shutil.copy2(source, out_dir / name)
    print(json.dumps({"status": "exported", "out": str(out_dir), "tensors": len(state)}))

    if args.verify:
        from transformers import AutoTokenizer, LlamaForCausalLM

        hf_model = LlamaForCausalLM.from_pretrained(str(out_dir), attn_implementation="eager")
        hf_tokenizer = AutoTokenizer.from_pretrained(str(out_dir))
        ids = tokenizer.encode("CORTEX est une intelligence artificielle")
        with torch.no_grad():
            ours = model(input_ids=torch.tensor([ids]))["logits"]
            theirs = hf_model(input_ids=torch.tensor([ids])).logits
        diff = float((ours - theirs).abs().max())
        token_match = hf_tokenizer.encode("Bonjour le monde", add_special_tokens=False) == tokenizer.encode("Bonjour le monde")
        print(json.dumps({"max_logit_diff": diff, "tokenizer_match": token_match}))
        if diff > 1e-3 or not token_match:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
