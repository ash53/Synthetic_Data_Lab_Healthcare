# src/eval/metrics.py
"""Evaluation utilities: holdout AUCs + simple distribution similarity."""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from scipy.stats import ks_2samp, wasserstein_distance  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

from src.utils.io import load_tabular


def _make_dummies(df: pd.DataFrame, categorical: List[str], numerical: List[str]) -> pd.DataFrame:
    cols = (categorical or []) + (numerical or [])
    return pd.get_dummies(df[cols], columns=categorical or [], drop_first=False)


def _align_like(X_ref: pd.DataFrame, X_new: pd.DataFrame) -> pd.DataFrame:
    """Add missing cols as 0 and order columns to match reference; drop extras."""
    return X_new.reindex(columns=X_ref.columns, fill_value=0)


def evaluate_tabular(cfg: Dict, synth_df: pd.DataFrame) -> None:
    """
    Writes eval.json to cfg['output']['reports_dir'] with:
      - downstream_auc.train_real:  train on REAL (70%) → test on REAL (30% holdout)
      - downstream_auc.train_synth_test_real: train on SYNTH → test on REAL holdout
      - simple numeric distribution & correlation deltas
    """
    real = load_tabular(cfg["data"]["input_csv"]).copy()

    target = cfg["data"]["target"]
    # 🔒 IMPORTANT: Never include the target in features
    cats   = [c for c in (cfg["data"]["categorical"] or []) if c != target]
    nums   = [c for c in (cfg["data"]["numerical"]   or []) if c != target]

    # ---------- Real holdout (stratified 70/30) ----------
    y_all = real[target].astype(int).values
    idx_tr, idx_te, y_tr, y_te = train_test_split(
        np.arange(len(real)), y_all, test_size=0.30, random_state=42, stratify=y_all
    )
    Xr_tr = _make_dummies(real.iloc[idx_tr], cats, nums)
    Xr_te = _make_dummies(real.iloc[idx_te], cats, nums)
    Xr_te = _align_like(Xr_tr, Xr_te)

    # ---------- Train on REAL → Test on REAL holdout ----------
    clf_r = LogisticRegression(max_iter=1000, solver="lbfgs")
    clf_r.fit(Xr_tr, y_tr)
    auc_rr = float(roc_auc_score(y_te, clf_r.predict_proba(Xr_te)[:, 1]))

    # ---------- Train on SYNTH → Test on REAL holdout ----------
    Xs = _make_dummies(synth_df, cats, nums)
    ys = synth_df[target].astype(int).values
    Xs = _align_like(Xr_tr, Xs)         # force synth features to real-train schema

    auc_sr = None
    if np.unique(ys).size >= 2:
        clf_s = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf_s.fit(Xs, ys)
        auc_sr = float(roc_auc_score(y_te, clf_s.predict_proba(Xr_te)[:, 1]))

    # ---------- Numeric distribution similarity ----------
    dist = {}
    for col in nums:
        a = real[col].dropna().values
        b = synth_df[col].dropna().values
        if len(a) == 0 or len(b) == 0:
            continue
        if _HAS_SCIPY:
            ks = float(ks_2samp(a, b).statistic)
            ws = float(wasserstein_distance(a, b))
        else:
            a_s, b_s = np.sort(a), np.sort(b)
            grid = np.linspace(min(a_s[0], b_s[0]), max(a_s[-1], b_s[-1]), num=512)
            def ecdf(x, g): return np.searchsorted(np.sort(x), g, side="right") / len(x)
            ks = float(np.max(np.abs(ecdf(a, grid) - ecdf(b, grid))))
            m = min(len(a_s), len(b_s))
            ws = float(np.mean(np.abs(a_s[:m] - b_s[:m])))
        dist[col] = {"ks": round(ks, 3), "wasserstein": ws}

    # ---------- Correlation delta ----------
    corr_diff = None
    try:
        rc = real[nums].corr().values
        sc = synth_df[nums].corr().values
        corr_diff = float(np.nanmean(np.abs(rc - sc)))
    except Exception:
        pass

    out = {
        "dist_similarity_numeric": dist,
        "corr_diff_abs_mean": corr_diff,
        "downstream_auc": {
            "train_real": auc_rr,
            "train_synth_test_real": auc_sr,
        },
    }

    os.makedirs(cfg["output"]["reports_dir"], exist_ok=True)
    with open(os.path.join(cfg["output"]["reports_dir"], "eval.json"), "w") as f:
        json.dump(out, f, indent=2)
