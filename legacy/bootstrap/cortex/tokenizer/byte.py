"""Dependency-free UTF-8 byte tokenizer used by the first baseline."""

BOS_ID = 256
EOS_ID = 257
PAD_ID = 258
VOCAB_SIZE = 259


class ByteTokenizer:
    vocab_size = VOCAB_SIZE

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        tokens = list(text.encode("utf-8"))
        if add_bos:
            tokens.insert(0, BOS_ID)
        if add_eos:
            tokens.append(EOS_ID)
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        data = bytes(token for token in token_ids if 0 <= token < 256)
        return data.decode("utf-8", errors="replace")
