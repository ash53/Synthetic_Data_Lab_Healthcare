# Synthetic EHR Data Generator (CTGAN + DP + Membership Inference)

> **What this is :** Generate **realistic, synthetic Electronic Health Record (EHR) like tabular data** with a CTGAN-style conditional generator (optional **differential privacy**) and a **membership-inference** harness to check privacy risk—plus a Streamlit demo.

---

## Features

- **CTGAN-style conditional generator** (control label / category mix)
- **Optional Differential Privacy** (noisy gradients)
- **Evaluation suite:** per-feature KS/Wasserstein, correlation gap, downstream AUC (train real vs. train synth → test real)
- **Membership-Inference Attack** (loss-threshold on downstream classifier)
- **Streamlit demo** with friendly KPIs (“Predictive score”, “Privacy risk”)

---

## Project Layout

```
.
├── demo/
│   └── streamlit_app.py         # Streamlit demo (entry point)
├── src/                         # Library code (ctgan, eval, privacy, utils)
├── data/
│   ├── raw/healthcare_demo.csv  # Tiny EHR-like seed (auto-created if missing)
│   └── synthetic/               # Generated CSVs (outputs)
├── reports/                     # eval.json, membership.json
├── requirements.txt
├── README.md
└── (optional) Dockerfile        # For Hugging Face Spaces (Docker SDK)
```

---

## Quickstart (CLI)

```bash
# Create env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train the healthcare CTGAN config
python -m src.cli --config configs/healthcare_ctgan_dp.yaml

# (Optional) Simple VAE image demo
python -m src.cli --config configs/image_vae.yaml
```

- Artifacts land in `data/synthetic/` and `reports/` (`eval.json`, `membership.json`).
- Included demo dataset columns:
  - **Numerical:** `age, bmi, systolic_bp, diastolic_bp, cholesterol`
  - **Categorical/Binary:** `sex {F,M}`, `smoker {0,1}`
  - **Target:** `diagnosis_diabetes {0,1}`

---

## 🖥️ Run the Streamlit Demo (Local)

```bash
# from repo root (venv active)
streamlit run demo/streamlit_app.py
```

The demo lets you:

- choose a positive-class prevalence (e.g., diabetes rate),
- train / generate,
- see **Predictive score** (AUC; 0.5 ≈ guessing, 1.0 = perfect),
- check **Privacy risk** (membership AUC; closer to 0.5 = safer),
- compare real vs. synthetic distributions & correlations,
- download the generated CSV + eval reports.

---

## Deploy the Demo

### Option A — **Hugging Face Spaces (Docker SDK)** _(recommended for PyTorch reliability)_

1. Put this **Dockerfile** at repo root:

```dockerfile
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=7860 STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
CMD ["streamlit","run","demo/streamlit_app.py","--server.address=0.0.0.0","--server.port=${PORT}"]
```

2. At the top of `README.md`, add this YAML block (Spaces reads it):

```yaml
---
title: Healthcare Synthetic Data Demo
emoji: 🩺
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
python_version: "3.11"
license: mit
---
```

3. Create a new **Space** → SDK **Docker** → push this repo.

### Option B — **Streamlit Community Cloud** (fastest to set up)

- Make sure `requirements.txt` includes `streamlit, pandas, numpy, plotly, scikit-learn, torch`.
- In Streamlit Cloud: **Create app** → repo path: `demo/streamlit_app.py` → Deploy.

---

## Outputs

- **`reports/eval.json`** — KS/Wasserstein per numeric feature, correlation gap (abs mean diff), downstream AUC:
  - `train_real` (Train Real → Test Real)
  - `train_synth_test_real` (Train Synth → Test Real)
- **`reports/membership.json`** — loss-threshold attack metrics (AUC/accuracy).

---

## Tech Stack

- Python, PyTorch, NumPy, Pandas, scikit-learn, Streamlit, Plotly

---

## Privacy & Safety Notes

- The included dataset is **synthetic/toy**; the project is for research & demo use.
- When enabling **DP**, expect some utility drop; tune ε & training to balance risk vs. accuracy.
- Membership inference is a **signal**, not a guarantee; pair with broader privacy reviews.

---

## 📜 License

MIT
