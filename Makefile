.PHONY: setup test lint run-health run-image docker

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

test:
	pytest -q

lint:
	ruff src

run-health:
	python -m src.cli --config configs/healthcare_ctgan_dp.yaml

run-image:
	python -m src.cli --config configs/image_vae.yaml

docker:
	docker build -t synth-hc:latest -f docker/Dockerfile .
