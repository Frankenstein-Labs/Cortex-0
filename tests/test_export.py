from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from cortex.training.trainer import save_pretrained

ROOT = Path(__file__).resolve().parents[1]


def test_export_is_numerically_equivalent(tmp_path, tiny_model, tokenizer):
    checkpoint = tmp_path / "ckpt"
    save_pretrained(tiny_model, tokenizer, tiny_model.config, checkpoint)
    export_dir = tmp_path / "hf"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_hf.py"),
            "--model-dir",
            str(checkpoint),
            "--out",
            str(export_dir),
            "--verify",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.strip().splitlines() if line.startswith("{")]
    verification = json.loads(lines[-1])
    assert verification["max_logit_diff"] < 1e-3
    assert verification["tokenizer_match"] is True
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "README.md"):
        assert (export_dir / name).exists()
    config = json.loads((export_dir / "config.json").read_text())
    assert config["architectures"] == ["LlamaForCausalLM"]
    assert config["model_type"] == "llama"
