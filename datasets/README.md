# Datasets

## TinyStories (pretraining)

* Source: `roneneldan/TinyStories` on Hugging Face.
* License: CDLA-Sharing-1.0.
* Files used: one training shard and the validation shard.
* Prepared subset: 120,000 training stories and 3,000 validation stories.

Measured by `scripts/prepare_data.py`:

| Split | Stories | Characters | Tokens |
| --- | ---: | ---: | ---: |
| train | 120,000 | 108,003,433 | 26,681,760 |
| validation | 3,000 | 2,622,556 | 644,887 |

Characters per token: 4.0483. Vocabulary: 8,192.

Raw parquet files and token streams are **not committed** (they are large and reproducible).
Run:

```bash
mkdir -p data/raw
curl -sL https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/train-00000-of-00004-2d5a1467fff1081b.parquet -o data/raw/tinystories_train.parquet
curl -sL https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/data/validation-00000-of-00001-869c898b519ad725.parquet -o data/raw/tinystories_valid.parquet
python scripts/prepare_data.py
```

## Synthetic data

Tests and the CI smoke run generate their own token streams. No dataset download is required to
run the test suite.

## Planned corpora

Soussou, Malinké and Bambara corpora are **not included**. See `docs/LANGUAGES.md` for the
collection and evaluation plan. Do not add a corpus without a clear license and provenance.
