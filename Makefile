.PHONY: check smoke train infer infer-4b merge gguf

check:
	python -m src.check_gpu

smoke:
	python -m src.train --config configs/smoke.yaml

train:
	python -m src.train --config configs/qwen3-4b-qlora.yaml

infer:
	python -m src.infer --config configs/smoke.yaml

infer-4b:
	python -m src.infer --config configs/qwen3-4b-qlora.yaml

merge:
	python -m src.merge --config configs/smoke.yaml

gguf:
	bash scripts/convert_gguf.sh --config configs/smoke.yaml
