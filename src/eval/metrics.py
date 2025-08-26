import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from src.utils.io import load_tabular


def _coerce_binary(y_series: pd.Series) -> np.ndarray:
    """
    Robustly coerce a label column to {0,1} integers.

    - Try numeric conversion.
    - If non-numeric, use categorical codes.
    - Finally, threshold at 0.5 to ensure {0,1}.
    """
    y = pd.to_numeric(y_series, errors="coerce")
    if y.isna().any():
        y = y_series.astype("category").cat.codes.astype(float)
    y = (y >= 0.5).astype(int)
    return y.values


def _prep_features(df: pd.DataFrame, feature_cols: list, ref_columns: pd.Index | None = None) -> pd.DataFrame:
    """
    One-hot encode features and align to reference columns (if provided).
    Fill NaNs with 0 to keep LogisticRegression happy.
    """
    X = pd.get_dummies(df[feature_cols], drop_first=False)
    if ref_columns is not None:
        X = X.reindex(columns=ref_columns, fill_value=0)
    X = X.astype(float).fillna(0.0)
    return X


def evaluate_tabular(cfg: dict, synth_df: pd.DataFrame) -> None:
    """
    Evaluate synthetic vs real tabular data:
      - Numeric distribution similarity per feature (KS/Wasserstein)
      - Numeric correlation structure gap (mean abs diff)
      - Downstream parity via LogisticRegression AUC:
          * train on real → test on real (baseline)
          * train on synthetic → test on real (parity)
      - Gracefully skip synthetic AUC if synthetic labels are single-class
    Saves results to {reports_dir}/eval.json.
    """
    real = load_tabular(cfg["data"]["input_csv"])
    y_col = cfg["data"]["target"]
    num_cols = cfg["data"]["numerical"]
    cat_cols = cfg["data"]["categorical"]

    # Exclude the target from features to avoid leakage
    cat_feats = [c for c in cat_cols if c != y_col]
    feature_cols = num_cols + cat_feats

    out: dict[str, object] = {}

    # -----------------------------
    # Distribution similarity (numeric)
    # -----------------------------
    feat_scores = {}
    for c in num_cols:
        r = real[c].astype(float).to_numpy(copy=False)
        s = synth_df[c].astype(float).to_numpy(copy=False)
        r = r[~np.isnan(r)]
        s = s[~np.isnan(s)]
        if r.size == 0 or s.size == 0:
            feat_scores[c] = {"ks": None, "wasserstein": None, "note": "empty after NaN filtering"}
        else:
            ks = ks_2samp(r, s).statistic
            wd = wasserstein_distance(r, s)
            feat_scores[c] = {"ks": float(ks), "wasserstein": float(wd)}
    out["dist_similarity_numeric"] = feat_scores

    # -----------------------------
    # Correlation structure gap (numeric only)
    # -----------------------------
    if len(num_cols) >= 2:
        rc = real[num_cols].corr().to_numpy()
        sc = synth_df[num_cols].corr().to_numpy()
        corr_diff = float(np.mean(np.abs(rc - sc)))
    else:
        corr_diff = None
    out["corr_diff_abs_mean"] = corr_diff

    # -----------------------------
    # Downstream AUC: real→real (baseline)
    # -----------------------------
    y_real = _coerce_binary(real[y_col])
    X_real_df = _prep_features(real, feature_cols)
    # Use stratify only if real labels have both classes
    stratifier = y_real if len(np.unique(y_real)) >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(
        X_real_df.values, y_real, test_size=0.3, random_state=42, stratify=stratifier
    )

    auc_real = None
    if len(np.unique(ytr)) >= 2 and len(np.unique(yte)) >= 2:
        clf_real = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        auc_real = float(roc_auc_score(yte, clf_real.predict_proba(Xte)[:, 1]))
    else:
        out["note_real"] = "Real dataset had single-class split; baseline AUC omitted."

    # -----------------------------
    # Downstream AUC: synth→real (parity)
    # -----------------------------
    y_s = _coerce_binary(synth_df[y_col])
    X_s_df = _prep_features(synth_df, feature_cols, ref_columns=X_real_df.columns)

    if len(np.unique(y_s)) >= 2:
        clf_synth = LogisticRegression(max_iter=1000).fit(X_s_df.values, y_s)
        # Always evaluate on the same real test set (Xte, yte)
        if auc_real is None:
            # still safe to compute synthetic AUC if yte has 2 classes
            if len(np.unique(yte)) >= 2:
                auc_synth = float(roc_auc_score(yte, clf_synth.predict_proba(Xte)[:, 1]))
                out["downstream_auc"] = {
                    "train_real": None,
                    "train_synth_test_real": auc_synth,
                }
            else:
                out["downstream_auc"] = {
                    "train_real": None,
                    "train_synth_test_real": None,
                    "note": "Real test labels were single-class; synthetic AUC omitted.",
                }
        else:
            # Baseline AUC exists; compute synthetic AUC if yte has 2 classes
            if len(np.unique(yte)) >= 2:
                auc_synth = float(roc_auc_score(yte, clf_synth.predict_proba(Xte)[:, 1]))
                out["downstream_auc"] = {
                    "train_real": auc_real,
                    "train_synth_test_real": auc_synth,
                }
            else:
                out["downstream_auc"] = {
                    "train_real": auc_real,
                    "train_synth_test_real": None,
                    "note": "Real test labels were single-class; synthetic AUC omitted.",
                }
    else:
        # Graceful skip for single-class synthetic labels
        out["downstream_auc"] = {
            "train_real": auc_real,
            "train_synth_test_real": None,
            "note": "Synthetic labels were single-class; skipped synthetic AUC.",
        }

    # -----------------------------
    # Persist
    # -----------------------------
    reports_dir = cfg["output"]["reports_dir"]
    with open(f"{reports_dir}/eval.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[EVAL]", out)


def evaluate_images(cfg: dict, tensor) -> None:
    """
    Minimal image evaluation: report global mean/std and save to eval.json.
    """
    stats = {"mean": float(tensor.mean().item()), "std": float(tensor.std().item())}
    reports_dir = cfg["output"]["reports_dir"]
    with open(f"{reports_dir}/eval.json", "w") as f:
        json.dump({"image_stats": stats}, f, indent=2)
    print("[EVAL]", {"image_stats": stats})
