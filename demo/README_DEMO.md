# Streamlit Demo for Privacy-Preserving Healthcare Synthetic Data Generator

This folder adds a **visual, interactive demo** (Streamlit) to the repo.

## Local run

```bash
# from repo root (where src/ exists)
pip install -r demo/requirements-demo.txt
streamlit run demo/streamlit_app.py
```

## What it shows
- **Generate** synthetic healthcare data with exact label control (diabetes ratio slider)
- **Metrics**: AUC (real→real vs synth→real), membership-attack AUC
- **Visuals**: label distribution, hist overlays for numeric columns, correlation heatmaps
- **Downloads**: synthetic CSV, eval.json, membership.json

## Deploy
- **Streamlit Community Cloud**: point to `demo/streamlit_app.py`, add `demo/requirements-demo.txt`.
- **Hugging Face Spaces (Streamlit)**: App file `demo/streamlit_app.py`.
