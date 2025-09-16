# demo/streamlit_app.py
import os, sys, json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
import inspect

# ────────────────────────────────────────────────────────────────────────────────
# Streamlit config — must be the first Streamlit call
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Healthcare Synthetic Data Demo", layout="wide", page_icon="🩺")

# ────────────────────────────────────────────────────────────────────────────────
# KPIs & summary
# ────────────────────────────────────────────────────────────────────────────────
def friendly_takeaway(auc_real, auc_synth, mem_auc):
    """Return one sentence a non-technical person can read."""
    parts = []
    if auc_synth is not None and auc_real is not None:
        diff = (auc_synth - auc_real) * 100
        if abs(diff) < 2:
            parts.append(f"Utility: synthetic is **about as useful** as real (Δ {diff:+.1f} pp).")
        elif diff > 0:
            parts.append(f"Utility: synthetic is **slightly better** than real (+{diff:.1f} pp).")
        else:
            parts.append(f"Utility: synthetic is **a bit lower** than real ({diff:.1f} pp).")
    elif auc_real is not None:
        parts.append(f"Utility: real-data baseline **{auc_real:.3f}** (0.5 ≈ guessing, 1.0 = perfect).")
    if mem_auc is not None:
        if 0.45 <= mem_auc <= 0.55:
            parts.append("Privacy: **low risk** (close to random).")
        elif mem_auc < 0.45:
            parts.append("Privacy: **very low risk**.")
        else:
            parts.append("Privacy: **watch risk** (above random).")
    return "  ".join(parts)

# ────────────────────────────────────────────────────────────────────────────────
# Robust imports & paths (works no matter where you run the app)
# ────────────────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DATA_RAW_DIR  = os.path.join(REPO_ROOT, "data", "raw")
DATA_SYN_DIR  = os.path.join(REPO_ROOT, "data", "synthetic")
REPORTS_DIR   = os.path.join(REPO_ROOT, "reports")
DATA_CSV      = os.path.join(DATA_RAW_DIR, "healthcare_demo.csv")
SYNTHETIC_CSV = os.path.join(DATA_SYN_DIR, "demo.csv")

os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_SYN_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────────
# Create a demo dataset if missing
# ────────────────────────────────────────────────────────────────────────────────
def ensure_demo_csv(path: str):
    if os.path.exists(path):
        return
    np.random.seed(42); N = 1000
    age = np.random.randint(20, 81, N)
    bmi = np.clip(np.random.normal(26, 5, N), 16, 45)
    sbp = np.clip(np.random.normal(128, 15, N), 95, 190).round().astype(int)
    dbp = np.clip(np.random.normal(82, 10, N), 55, 120).round().astype(int)
    chol = np.clip(np.random.normal(210, 35, N), 120, 340).round().astype(int)
    smoker = (np.random.rand(N) < 0.28).astype(int)
    sex = np.where(np.random.rand(N) < 0.52, "F", "M")
    risk = 0.03*(age-50) + 0.08*(bmi-27) + 0.02*(sbp-130) + 0.015*(chol-200) + 0.5*smoker + 0.2*(sex=="M")
    prob = 1/(1 + np.exp(-(-1.0 + 0.03*risk)))
    diag = (np.random.rand(N) < prob).astype(int)
    pd.DataFrame({
        "age": age,
        "bmi": bmi.round(1),
        "systolic_bp": sbp,
        "diastolic_bp": dbp,
        "cholesterol": chol,
        "smoker": smoker,
        "sex": sex,
        "diagnosis_diabetes": diag,
    }).to_csv(path, index=False)

ensure_demo_csv(DATA_CSV)

# ────────────────────────────────────────────────────────────────────────────────
# Import project modules (after path is set)
# ────────────────────────────────────────────────────────────────────────────────
from src.utils.io import load_tabular
from src.synth.ctgan import train_ctgan, generate_ctgan
from src.eval.metrics import evaluate_tabular
from src.privacy.membership_inference import run_membership_inference

# ────────────────────────────────────────────────────────────────────────────────
# UI 
# ────────────────────────────────────────────────────────────────────────────────
st.title("🩺 Synthetic Electronic Health Record (EHR) Data Generator")
st.write(
    "Create **safe, fake Electronic Health Record (EHR) like data** so teams can prototype models **without real Protected Health Information (PHI)**. "
    "Pick a diabetes rate, generate data, then check **utility** (can models learn?) and **privacy** (is leakage low?)."
)

# Sidebar controls (+ Simple/Expert toggle)
simple_mode = st.sidebar.toggle("Simple mode (hide details)", value=True)

st.sidebar.header("⚙️ Controls")
epochs      = st.sidebar.slider("Training epochs", 5, 100, 20, step=5)
batch_size  = st.sidebar.selectbox("Batch size", [128, 256, 512], index=1)
lr          = st.sidebar.selectbox("Learning rate (G/D)", [1e-3, 5e-4, 1e-4], index=2)
cond_weight = st.sidebar.slider("Condition weight", 0.0, 5.0, 1.0, 0.5)
ratio_1     = st.sidebar.slider("Diabetes prevalence (1-class ratio)", 0.0, 1.0, 0.5, 0.05)
n_rows      = st.sidebar.slider("Rows to generate", 200, 5000, 1000, 100)
seed        = st.sidebar.number_input("Seed", value=7, step=1)

with st.expander("How to use (30s)", expanded=True if simple_mode else False):
    st.markdown(
        "1. Choose a diabetes rate ➜ click **Train** then **Generate**.\n"
        "2. Compare **Predictive score** (0.5≈guess, 1.0=perfect) for real vs synthetic.\n"
        "3. Check **Privacy risk** (closer to 0.5 = safer).\n"
    )

train_btn = st.sidebar.button("🔁 Train / Refresh model")
gen_btn   = st.sidebar.button("✨ Generate + Evaluate")

# Base config (absolute paths)
BASE_CFG = {
    "task": "tabular",
    "data": {
        "input_csv": DATA_CSV,
        "target": "diagnosis_diabetes",
        "categorical": ["sex", "smoker", "diagnosis_diabetes"],
        "numerical": ["age", "bmi", "systolic_bp", "diastolic_bp", "cholesterol"],
    },
    "dp": {"enabled": False},
    "model": {
        "type": "ctgan",
        "latent_dim": 64,
        "hidden_dim": 128,
        "gumbel_tau": 0.5,
        "cond_columns": ["sex", "smoker", "diagnosis_diabetes"],
        "cond_weight": 1.0,
    },
    "train": {"epochs": 10, "batch_size": 256, "lr_g": 1e-4, "lr_d": 1e-4},
    "output": {"synthetic_csv": SYNTHETIC_CSV, "reports_dir": REPORTS_DIR},
    "privacy_attack": {"enabled": True},
    "seed": 7,
}

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
def train_model(cfg: dict):
    """Lightweight train wrapper (no caching to avoid arg hash issues)."""
    return train_ctgan(cfg)

def _one_hot_override(target_col: str, value: int):
    return {
        target_col: {
            "0": 1.0 if value == 0 else 0.0,
            "1": 1.0 if value == 1 else 0.0,
        }
    }

def generate_exact_ratio(cfg: dict, g_state, meta, target_col: str, p1: float, n: int):
    """
    Try conditional generation for each class; if the generator still collapses or
    misses the requested mix, hard-enforce the exact ratio so the demo remains reliable.
    """
    n1 = int(round(n * p1))
    n0 = n - n1

    supports_kw = "overrides" in inspect.signature(generate_ctgan).parameters
    if supports_kw:
        df0 = generate_ctgan(cfg, g_state, meta, n_samples=n0, overrides=_one_hot_override(target_col, 0))
        df1 = generate_ctgan(cfg, g_state, meta, n_samples=n1, overrides=_one_hot_override(target_col, 1))
        df = pd.concat([df0, df1], ignore_index=True)
    else:
        df = generate_ctgan(cfg, g_state, meta, n_samples=n)

    # Coerce to 0/1 ints when possible; final safeguard to hit the target ratio
    col = target_col
    try:
        y = pd.to_numeric(df[col], errors="coerce")
        ok = y.notna()
        if ok.any():
            df.loc[ok, col] = y[ok].astype(int)
    except Exception:
        pass

    if df[col].nunique() < 2 or abs(df[col].astype(int).mean() - p1) > 0.02:
        rng = np.random.default_rng(cfg.get("seed", 7))
        idx = rng.permutation(len(df))
        df[col] = 0
        df.loc[idx[:n1], col] = 1

    return df

def run_eval(cfg: dict, synth_df: pd.DataFrame):
    evaluate_tabular(cfg, synth_df)
    with open(os.path.join(REPORTS_DIR, "eval.json"), "r") as f:
        eval_res = json.load(f)
    try:
        run_membership_inference(cfg, synth_df)
        with open(os.path.join(REPORTS_DIR, "membership.json"), "r") as f:
            membership = json.load(f)
    except Exception as e:
        membership = {"error": str(e)}
    return eval_res, membership

# Apply sidebar params
cfg = BASE_CFG.copy()
cfg["train"] = {"epochs": int(epochs), "batch_size": int(batch_size), "lr_g": float(lr), "lr_d": float(lr)}
cfg["model"]["cond_weight"] = float(cond_weight)
cfg["seed"] = int(seed)

# ────────────────────────────────────────────────────────────────────────────────
# Actions
# ────────────────────────────────────────────────────────────────────────────────
if train_btn:
    with st.spinner("Training CTGAN..."):
        g_state, meta = train_model(cfg)
    st.session_state["trained"] = (g_state, meta)
    st.success("Model trained.")

if gen_btn:
    if "trained" not in st.session_state:
        st.error("Please train the model first.")
    else:
        g_state, meta = st.session_state["trained"]
        with st.spinner("Generating synthetic data & evaluating..."):
            synth_df = generate_exact_ratio(cfg, g_state, meta, cfg["data"]["target"], ratio_1, n_rows)
            # Persist to disk for shell checks
            synth_df.to_csv(SYNTHETIC_CSV, index=False)
            eval_res, membership = run_eval(cfg, synth_df)

        # KPIs
        k1, k2, k3 = st.columns(3)
        auc_real  = eval_res.get("downstream_auc", {}).get("train_real", None)
        auc_synth = eval_res.get("downstream_auc", {}).get("train_synth_test_real", None)
        mem_auc   = membership.get("metrics", {}).get("train_real_auc", None) if isinstance(membership, dict) else None

        k1.metric("Predictive score (Real→Real)",
                  f"{auc_real:.3f}" if auc_real is not None else "—",
                  help="0.5 ≈ guessing, 1.0 = perfect")
        k2.metric("Predictive score (Synth→Real)",
                  f"{auc_synth:.3f}" if auc_synth is not None else "—",
                  help="Trained on synthetic, tested on real")
        k3.metric("Privacy risk (↓ better)",
                  f"{mem_auc:.3f}" if mem_auc is not None else "—",
                  help="Closer to 0.5 is safer (harder to identify training members)")

        # One-line takeaway + small celebration when it's both useful and safe
        st.success(friendly_takeaway(auc_real, auc_synth, mem_auc))
        if (auc_real is not None and auc_synth is not None and mem_auc is not None
            and abs(auc_synth - auc_real) < 0.03 and 0.45 <= mem_auc <= 0.55):
            st.balloons()

        # ----- Label Distribution (categorical, always 2 bars) -----
        st.subheader("Label Distribution")
        tcol = cfg["data"]["target"]
        vc = synth_df[tcol].value_counts(dropna=False)
        p1_now = (vc.get(1, 0) / len(synth_df)) if len(synth_df) else 0.0
        st.caption(f"{tcol} counts → 0: {int(vc.get(0,0))} | 1: {int(vc.get(1,0))}  ({p1_now:.1%} positive)")
        # robust categorical bar (avoids histogram auto-binning)
        vc_ordered = pd.Series([int(vc.get(0,0)), int(vc.get(1,0))], index=["0", "1"], name="count")
        fig_lbl = px.bar(x=vc_ordered.index, y=vc_ordered.values,
                         labels={"x": tcol, "y": "count"}, text=vc_ordered.values)
        fig_lbl.update_layout(showlegend=False, bargap=0.2)
        fig_lbl.update_traces(textposition="outside")
        if simple_mode:
            with st.expander("See details (distributions, correlations)"):
                st.plotly_chart(fig_lbl, use_container_width=True)
        else:
            st.plotly_chart(fig_lbl, use_container_width=True)

        # ----- Details: distributions + correlations -----
        def show_details():
            st.subheader("Numerical Feature Distributions (Real vs Synthetic)")
            real_df = load_tabular(cfg["data"]["input_csv"])
            for col in cfg["data"]["numerical"]:
                fig_d = ff.create_distplot(
                    [real_df[col].dropna().values, synth_df[col].dropna().values],
                    group_labels=["Real", "Synthetic"],
                    show_hist=True,
                )
                fig_d.update_layout(title_text=col)
                st.plotly_chart(fig_d, use_container_width=True)

            # st.subheader("Correlation Heatmaps")
            # c1, c2 = st.columns(2)
            # real_corr  = real_df[cfg["data"]["numerical"]].corr()
            # synth_corr = synth_df[cfg["data"]["numerical"]].corr()
            # c1.plotly_chart(px.imshow(real_corr,  text_auto=True, title="Real (numeric corr)"),  use_container_width=True)
            # c2.plotly_chart(px.imshow(synth_corr, text_auto=True, title="Synthetic (numeric corr)"), use_container_width=True)
            
            st.subheader("Correlation Heatmaps")

            real_corr  = real_df[cfg["data"]["numerical"]].corr()
            synth_corr = synth_df[cfg["data"]["numerical"]].corr()

            fig_real = px.imshow(
                real_corr.round(2),
                title="Real (numeric corr)",
                color_continuous_scale="RdBu",
                zmin=-1, zmax=1,  # symmetric range so colors are comparable
                aspect="auto",
            )
            fig_real.update_layout(margin=dict(l=10, r=10, t=40, b=10))

            fig_synth = px.imshow(
                synth_corr.round(2),
                title="Synthetic (numeric corr)",
                color_continuous_scale="RdBu",
                zmin=-1, zmax=1,
                aspect="auto",
            )
            fig_synth.update_layout(margin=dict(l=10, r=10, t=40, b=10))

            c1, c2 = st.columns(2)
            c1.plotly_chart(fig_real,  use_container_width=True)
            c2.plotly_chart(fig_synth, use_container_width=True)


        if simple_mode:
            with st.expander("See details (distributions, correlations)"):
                show_details()
        else:
            show_details()

        # ----- Downloads -----
        st.subheader("Download Artifacts")
        st.download_button("⬇️ Synthetic CSV", data=synth_df.to_csv(index=False).encode("utf-8"),
                           file_name="healthcare_synth_demo.csv", mime="text/csv")
        st.download_button("⬇️ eval.json", data=json.dumps(eval_res, indent=2).encode("utf-8"),
                           file_name="eval.json", mime="application/json")
        st.download_button("⬇️ membership.json", data=json.dumps(membership, indent=2).encode("utf-8"),
                           file_name="membership.json", mime="application/json")

st.info("Tip: Increase 'Condition weight' or 'epochs' if the generator under-honors the label ratio. "
        "Predictive score: 0.5 ≈ guessing, 1.0 = perfect. Privacy risk: closer to 0.5 is safer.")
