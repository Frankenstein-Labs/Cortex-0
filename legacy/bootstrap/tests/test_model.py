import torch

from cortex.model import CortexConfig, CortexLM
from cortex.tokenizer import ByteTokenizer


def test_forward_returns_logits_and_finite_loss():
    torch.manual_seed(0)
    config = CortexConfig(vocab_size=259, context_size=16, d_model=32, n_layers=2, n_heads=4, d_ff=64)
    model = CortexLM(config)
    ids = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(ids, ids)
    assert logits.shape == (2, 16, config.vocab_size)
    assert loss is not None and torch.isfinite(loss)


def test_generation_extends_sequence():
    torch.manual_seed(0)
    config = CortexConfig(vocab_size=259, context_size=8, d_model=32, n_layers=1, n_heads=4, d_ff=64)
    model = CortexLM(config)
    result = model.generate(torch.tensor([[1, 2]]), max_new_tokens=3, temperature=1.0, top_k=10)
    assert result.shape == (1, 5)


def test_byte_tokenizer_round_trip():
    tokenizer = ByteTokenizer()
    text = "Bonjour, CORTEX-0 !"
    assert tokenizer.decode(tokenizer.encode(text)) == text
