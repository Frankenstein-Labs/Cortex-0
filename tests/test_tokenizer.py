from __future__ import annotations

import pytest

from cortex.tokenizer.bpe import SPECIAL_TOKEN_IDS, SPECIAL_TOKENS, CortexTokenizer


@pytest.mark.parametrize(
    "text",
    [
        "Bonjour, je m'appelle CORTEX et j'habite à Paris.",
        "Les accents : é è ê à ç ù û î ï ô œ.",
        "Bambara: i ni ce, Malinké: i ni ke, Soussou: i nu wali.",
        "Emoji: 🧠⚡ and code: def f(x): return x**2",
    ],
)
def test_roundtrip_is_byte_exact(tokenizer, text):
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert tokenizer.token_to_id("<|unk|>") not in ids


def test_special_token_ids(tokenizer):
    for name in SPECIAL_TOKENS:
        assert tokenizer.token_to_id(name) == SPECIAL_TOKEN_IDS[name]
    assert tokenizer.vocab_size >= len(SPECIAL_TOKENS)


def test_chat_template_structure(tokenizer):
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Tu es CORTEX"},
            {"role": "user", "content": "Bonjour"},
        ],
        add_generation_prompt=True,
    )
    assert prompt.startswith("<|bos|><|system|>Tu es CORTEX<|end|>")
    assert "<|user|>Bonjour<|end|><|assistant|>" in prompt
    ids = tokenizer.encode(prompt)
    assert ids[-1] == SPECIAL_TOKEN_IDS["<|assistant|>"]


def test_chat_template_rejects_unknown_role(tokenizer):
    with pytest.raises(ValueError):
        tokenizer.apply_chat_template([{"role": "wizard", "content": "x"}])


def test_encode_supervised_masks_non_assistant_tokens(tokenizer):
    messages = [
        {"role": "system", "content": "Tu es CORTEX"},
        {"role": "user", "content": "Combien font 2+2 ?"},
        {"role": "assistant", "content": "Réponse : 4"},
    ]
    ids, labels = tokenizer.encode_supervised(messages)
    assert len(ids) == len(labels)
    supervised = [label for label in labels if label != -100]
    assert supervised
    assert tokenizer.decode(supervised, skip_special_tokens=False) == "Réponse : 4<|end|>"
    # System and user content are never trained on.
    assert labels[1] == -100


def test_save_and_reload(tmp_path, tokenizer):
    tokenizer.save(tmp_path / "tok")
    reloaded = CortexTokenizer.from_pretrained(tmp_path / "tok")
    text = "La mémoire et le raisonnement progressent."
    assert reloaded.encode(text) == tokenizer.encode(text)
    assert reloaded.decode(reloaded.encode(text)) == text
