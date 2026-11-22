from __future__ import annotations

import pytest
import torch

from cortex.config import CortexConfig
from cortex.model.lm import CortexLM, load_model, save_checkpoint


def test_config_validation():
    with pytest.raises(ValueError):
        CortexConfig(d_model=65, n_heads=4)
    with pytest.raises(ValueError):
        CortexConfig(n_heads=6, n_kv_heads=4)
    cfg = CortexConfig()
    assert cfg.ffn_hidden % 64 == 0


def test_parameter_estimate_matches_reality(tiny_cfg):
    model = CortexLM(tiny_cfg)
    actual = model.num_parameters()
    estimate = tiny_cfg.parameter_estimate()
    assert abs(actual - estimate) / actual < 0.05


def test_forward_shapes_and_loss(tiny_model):
    x = torch.randint(0, 320, (3, 16))
    y = torch.randint(0, 320, (3, 16))
    out = tiny_model(x, targets=y)
    assert out["logits"].shape == (3, 16, 320)
    assert torch.isfinite(out["loss"])
    assert 0.0 <= float(out["token_accuracy"]) <= 1.0


def test_sequence_longer_than_context_rejected(tiny_model):
    with pytest.raises(ValueError):
        tiny_model(torch.randint(0, 320, (1, 65)))


def test_causality(tiny_model):
    tiny_model.eval()
    x = torch.randint(0, 320, (1, 12))
    with torch.no_grad():
        a = tiny_model(x)["logits"]
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 7) % 320
        b = tiny_model(x2)["logits"]
    assert torch.allclose(a[0, :-1], b[0, :-1], atol=1e-5)


def test_kv_cache_matches_full_forward(tiny_model):
    tiny_model.eval()
    x = torch.randint(0, 320, (1, 20))
    with torch.no_grad():
        full = tiny_model(x)["logits"]
        past = None
        parts = []
        for i in range(x.shape[1]):
            out = tiny_model(x[:, i : i + 1], past_kv=past, use_cache=True)
            past = out["past_kv"]
            parts.append(out["logits"])
        inc = torch.cat(parts, dim=1)
    assert torch.allclose(full, inc, atol=1e-4)


def test_checkpoint_roundtrip(tiny_model, tmp_path):
    path = tmp_path / "model.safetensors"
    save_checkpoint(tiny_model, path, extra={"note": "test"})
    loaded = load_model(path)
    assert loaded.num_parameters() == tiny_model.num_parameters()
    x = torch.randint(0, 320, (1, 8))
    tiny_model.eval()
    with torch.no_grad():
        assert torch.allclose(tiny_model(x)["logits"], loaded(x)["logits"], atol=1e-6)


def test_gqa_head_configuration(tiny_tokenizer):
    cfg = CortexConfig(
        vocab_size=tiny_tokenizer.vocab_size,
        d_model=64,
        n_layers=1,
        n_heads=8,
        n_kv_heads=1,
        max_seq_len=32,
    )
    model = CortexLM(cfg)
    out = model(torch.randint(0, cfg.vocab_size, (2, 10)))
    assert out["logits"].shape[-1] == cfg.vocab_size
