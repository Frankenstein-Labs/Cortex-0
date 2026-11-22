.PHONY: install test data train eval generate serve export docker

install:
	python -m pip install -e ".[dev,train]"

test:
	python -m pytest -q

data:
	python scripts/prepare_data.py --config configs/tiny-cpu.yaml

train:
	python scripts/train.py --config configs/tiny-cpu.yaml

eval:
	python scripts/evaluate.py --model-dir checkpoints/tiny-cpu/best \
		--val-data datasets/processed/val.bin --n 20 \
		--output experiments/eval-tiny-cpu.json

generate:
	python scripts/generate.py --model-dir checkpoints/tiny-cpu \
		--prompt "CORTEX est" --max-new-tokens 60

serve:
	python scripts/serve.py --model-dir checkpoints/tiny-cpu --port 8000

export:
	python scripts/export_hf.py --model-dir checkpoints/tiny-cpu \
		--out export/Cortex-0 --verify

docker:
	docker build -t cortex-0 .
