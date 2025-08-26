# Synthetic Data Lab — Healthcare (CTGAN + DP + Membership Inference)

A Python framework to generate realistic **tabular healthcare** data (and small images) with:
- **CTGAN-like conditional generator** to control label and category balance
- Optional **Differential Privacy** (noisy gradients)
- **Profiling** & **Evaluation** (distribution similarity, correlation gap, downstream parity)
- **Membership-Inference Attack Harness** for privacy risk assessment

## Quickstart

```bash
# Create env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run healthcare CTGAN config
python -m src.cli --config configs/healthcare_ctgan_dp.yaml

# (Optional) Run simple VAE image demo
python -m src.cli --config configs/image_vae.yaml
```

Artifacts land under `data/synthetic/` and `reports/` (including `eval.json` and `membership.json` when enabled).

## Healthcare demo dataset
Included: `data/raw/healthcare_demo.csv` with columns:
- Numerical: `age, bmi, systolic_bp, diastolic_bp, cholesterol`
- Categorical/Binary: `sex {F,M}`, `smoker {0,1}`
- Target: `diagnosis_diabetes {0,1}`

## Features
- CTGAN-like **conditional tabular GAN** (PyTorch)
- Differential privacy via **noisy gradient sanitization**
- **Config-driven CLI**, **Dockerfile**, **unit tests**
- **Membership inference** (loss-threshold attack on a downstream classifier) to estimate leakage risk

## Evaluate & attack
- `reports/eval.json`: KS/Wasserstein per-feature, correlation structure gap, AUC parity (train real vs train synth → test real)
- `reports/membership.json`: Attack AUC/accuracy using loss-threshold membership inference

## License
MIT
