import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

from src.utils.io import load_tabular


def _coerce_binary(y: pd.Series) -> np.ndarray:
    """
    Robustly coerce label column to {0,1} ints.
    - Try numeric; if it fails, use categorical codes.
    - Threshold at 0.5 to ensure {0,1}.
    """
    y_num = pd.to_numeric(y, errors="coerce")
    if y_num.isna().any():
        y_num = y.astype("category").cat.codes.astype(float)
    return (y_num >= 0.5).astype(int).values


def _prep_features(df: pd.DataFrame, feature_cols: list, ref_columns: pd.Index | None = None) -> pd.DataFrame:
    """
    One-hot encode and align to reference columns (if provided).
    """
    X = pd.get_dummies(df[feature_cols], drop_first=False)
    if ref_columns is not None:
        X = X.reindex(columns=ref_columns, fill_value=0)
    return X.astype(float).fillna(0.0)


def run_membership_inference(cfg: dict, synth_df: pd.DataFrame) -> None:
    """
    Yeom-style loss-threshold membership inference on a downstream classifier.

    Steps:
      1) Train a target LogisticRegression model on REAL train split.
      2) Compute per-sample loss on REAL train members and REAL holdout non-members.
      3) Threshold at midpoint between mean(train_loss) and mean(test_loss).
      4) Report attack ACC/AUC.
      5) Repeat with a model trained on SYNTHETIC (optional; skipped if synthetic labels are single-class).

    Saves JSON to {reports_dir}/membership.json
    """
    reports_dir = cfg["output"]["reports_dir"]
    y_col = cfg["data"]["target"]
    num_cols = cfg["data"]["numerical"]
    cat_cols = cfg["data"]["categorical"]

    # Exclude target from features
    cat_feats = [c for c in cat_cols if c != y_col]
    feature_cols = num_cols + cat_feats

    # ---- Load + prep real ----
    real = load_tabular(cfg["data"]["input_csv"])
    y_real = _coerce_binary(real[y_col])
    X_real_df = _prep_features(real, feature_cols)

    # Stratify only if both classes exist
    stratifier = y_real if len(np.unique(y_real)) >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_real_df.values, y_real, test_size=0.5, random_state=1337, stratify=stratifier
    )

    out = {"attack": "loss-threshold (Yeom) on logistic regression", "metrics": {}}

    # If the real training labels collapse to a single class, we cannot run the attack meaningfully
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        out["metrics"]["note"] = "Real split had single-class labels; membership attack not applicable."
        with open(f"{reports_dir}/membership.json", "w") as f:
            json.dump(out, f, indent=2)
        print("[MEMBERSHIP]", out)
        return

    # ---- Train target on REAL ----
    model_r = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)

    # Probabilities on train/test
    p_tr = model_r.predict_proba(X_tr)[:, 1]
    p_te = model_r.predict_proba(X_te)[:, 1]

    # Per-sample log loss
    eps = 1e-12
    loss_tr = -(y_tr * np.log(p_tr + eps) + (1 - y_tr) * np.log(1 - p_tr + eps))
    loss_te = -(y_te * np.log(p_te + eps) + (1 - y_te) * np.log(1 - p_te + eps))

    # Threshold at midpoint between means
    t = 0.5 * (loss_tr.mean() + loss_te.mean())

    # Labels: 1 = member (train), 0 = non-member (test)
    losses = np.concatenate([loss_tr, loss_te], axis=0)
    labels = np.concatenate([np.ones_like(loss_tr), np.zeros_like(loss_te)], axis=0)
    preds = (losses < t).astype(int)

    acc_r = float(accuracy_score(labels, preds))
    auc_r = float(roc_auc_score(labels, -losses))  # lower loss => more likely member

    out["metrics"]["train_real_acc"] = acc_r
    out["metrics"]["train_real_auc"] = auc_r

    # ---- Train target on SYNTH (optional) ----
    y_s = _coerce_binary(synth_df[y_col])
    if len(np.unique(y_s)) < 2:
        # Skip cleanly if synthetic labels are single-class
        out["metrics"]["train_synth_acc"] = None
        out["metrics"]["train_synth_auc"] = None
        out["metrics"]["note_synth"] = "Synthetic labels were single-class; skipped synthetic model attack."
    else:
        X_s_df = _prep_features(synth_df, feature_cols, ref_columns=X_real_df.columns)
        model_s = LogisticRegression(max_iter=1000).fit(X_s_df.values, y_s)

        # Attack still evaluated on REAL members vs REAL non-members
        p_tr_s = model_s.predict_proba(X_tr)[:, 1]
        loss_tr_s = -(y_tr * np.log(p_tr_s + eps) + (1 - y_tr) * np.log(1 - p_tr_s + eps))
        t_s = 0.5 * (loss_tr_s.mean() + loss_te.mean())

        losses_s = np.concatenate([loss_tr_s, loss_te], axis=0)
        preds_s = (losses_s < t_s).astype(int)

        acc_s = float(accuracy_score(labels, preds_s))
        auc_s = float(roc_auc_score(labels, -losses_s))

        out["metrics"]["train_synth_acc"] = acc_s
        out["metrics"]["train_synth_auc"] = auc_s

    with open(f"{reports_dir}/membership.json", "w") as f:
        json.dump(out, f, indent=2)
    print("[MEMBERSHIP]", out)
