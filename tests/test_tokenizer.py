from __future__ import annotations

from cortex.tokenizer.bpe import CortexTokenizer


def test_roundtrip_ascii_and_unicode(tiny_tokenizer):
    for text in [
        "Once upon a time, there was a little robot.",
        "Le petit garçon mange une pomme à côté de la rivière.",
        "N'importe quoi — accents: éèêëàâäîïôöùûüç",
        "Numbers 12345 and symbols !?.,;:()[]",
    ]:
        ids = tiny_tokenizer.encode(text)
        assert ids
        assert tiny_tokenizer.decode(ids) == text


def test_special_token_ids(tiny_tokenizer):
    assert tiny_tokenizer.pad_id == 0
    assert tiny_tokenizer.bos_id == 1
    assert tiny_tokenizer.eos_id == 2
    assert tiny_tokenizer.user_id == 3
    assert tiny_tokenizer.assistant_id == 4
    assert tiny_tokenizer.end_id == 5


def test_chat_prompt_format(tiny_tokenizer):
    prompt = tiny_tokenizer.chat_prompt(
        [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]
    )
    assert prompt.startswith("<|bos|><|user|>Hello<|end|><|assistant|>Hi<|end|>")
    assert prompt.endswith("<|assistant|>")


def test_save_and_load(tiny_tokenizer, tmp_path):
    path = tmp_path / "tok.json"
    tiny_tokenizer.save_with_meta(path, {"vocab_size": tiny_tokenizer.vocab_size})
    loaded, meta = CortexTokenizer.from_meta(path)
    assert loaded.vocab_size == tiny_tokenizer.vocab_size
    assert meta["vocab_size"] == tiny_tokenizer.vocab_size
    text = "A small bird lived in a tall tree."
    assert loaded.encode(text) == tiny_tokenizer.encode(text)
