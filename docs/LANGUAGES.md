# Languages and tokenizer extensibility

The tokenizer is a **byte-level BPE**. It can encode any UTF-8 text without an unknown token,
including Soussou, Malinké and Bambara. That is an encoding property, not a language capability.

Current state:

* Tokenizer trained on English TinyStories (8,192 merges).
* French text is encodable but fragmented into more tokens and not represented in pretraining.
* No Soussou, Malinké or Bambara corpus has been used. No quality claim is made.

To extend coverage:

1. Collect licensed text for each language (books, Wikipedia dumps, community corpora with
   explicit permission).
2. Measure fertility (tokens per word) per language before and after retraining.
3. Train a new tokenizer (16k–32k merges) on a balanced multilingual sample.
4. Continue pretraining from the existing checkpoint with a new embedding matrix, or train from
   scratch if the vocabulary changes substantially.
5. Evaluate per language on held-out perplexity and on native-speaker-reviewed prompts.

The scripts accept any text corpus; `scripts/prepare_data.py` only assumes a parquet file with a
`text` column.
