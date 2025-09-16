"""
CTGAN-like conditional tabular generator with:
  - Numeric standardization + categorical one-hot encoding
  - Optional Differential Privacy gradient sanitization
  - Condition sampling with per-column probability overrides
  - Condition-consistency loss so G matches the sampled condition
  - Generation API that accepts `overrides` to force class ratios (e.g., 50/50)

Assumes you have:
  src/models/generator.py -> class MLPGenerator(latent_dim, out_dim, hidden_dim, cond_dim=0)
  src/models/discriminator.py -> class MLPDiscriminator(in_dim, hidden_dim)
  src/synth/dp.py -> sanitize_gradients(params)  # no-op if DP disabled
  src/utils/io.py -> load_tabular(path) -> pd.DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.generator import MLPGenerator
from src.models.discriminator import MLPDiscriminator
from src.synth.dp import sanitize_gradients
from src.utils.io import load_tabular


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _one_hot(values: np.ndarray, uniques: np.ndarray) -> np.ndarray:
    """Simple one-hot encoder for a 1D array using a fixed `uniques` ordering."""
    idx = np.array([np.where(uniques == v)[0][0] for v in values])
    oh = np.zeros((len(values), len(uniques)), dtype=np.float32)
    oh[np.arange(len(values)), idx] = 1.0
    return oh


def _override_probs_for_column(
    uniques: np.ndarray,
    base_probs: np.ndarray,
    override_dict: dict | None,
) -> np.ndarray:
    """
    Map override weights (keys can be string/int/actual value) onto the `uniques` order.
    Falls back to base_probs if overrides are invalid or all-zero.
    """
    if override_dict is None:
        return base_probs
    w = np.array(
        [
            float(override_dict.get(str(v), override_dict.get(int(v), override_dict.get(v, 0.0))))
            for v in uniques
        ],
        dtype=np.float32,
    )
    s = w.sum()
    return (w / s) if s > 0 else base_probs


def _base_priors(meta: dict, col: str) -> np.ndarray:
    """
    Get categorical priors for a column.
    Order matches `meta["cat_uniques"][col]`.
    Priority: cached priors -> frequencies in meta["df"] -> uniform.
    """
    if "priors" in meta and col in meta["priors"]:
        return meta["priors"][col]
    u = meta["cat_uniques"][col]
    df = meta.get("df", None)
    if df is not None:
        p = np.array([(df[col].values == v).mean() for v in u], dtype=np.float32)
        s = p.sum()
        return (p / s) if s > 0 else np.ones(len(u), dtype=np.float32) / len(u)
    return np.ones(len(u), dtype=np.float32) / len(u)


def _sample_condition(meta: dict, batch: int, overrides: dict | None = None) -> np.ndarray:
    """
    Sample a full condition vector by sampling each categorical column independently.
    If `overrides` contains a column, those probabilities are used for that column.
    Returns a 2D array of shape [batch, cond_dim] (concatenated one-hots).
    """
    blocks = []
    for c in meta["cat_cols"]:
        u = meta["cat_uniques"][c]
        p = _base_priors(meta, c)
        if overrides and c in overrides:
            p = _override_probs_for_column(u, p, overrides[c])

        idx = np.random.choice(len(u), size=batch, p=p)
        oh = np.zeros((batch, len(u)), dtype=np.float32)
        oh[np.arange(batch), idx] = 1.0
        blocks.append(oh)

    if not blocks:
        return np.zeros((batch, 0), dtype=np.float32)
    return np.hstack(blocks).astype(np.float32)


def _split_gen_output(meta: dict, g_out: torch.Tensor) -> dict:
    """
    Split generator raw output into numerical block and a list of categorical blocks.

    Returns:
      {
        "num": tensor [B, num_dim],
        "cat_segments": [tensor [B, K_c] per categorical column]
      }
    """
    num_dim = meta["num_dim"]
    pos = num_dim
    outs = {"num": g_out[:, :num_dim], "cat_segments": []}
    for c in meta["cat_cols"]:
        k = len(meta["cat_uniques"][c])
        outs["cat_segments"].append(g_out[:, pos : pos + k])
        pos += k
    return outs


def _prep_ctgan(cfg: dict) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """
    Prepare model metadata and the real feature matrix for D:
      - standardize numeric -> X_num
      - one-hot categoricals -> X_cat
      - X_real = [X_num, X_cat]
      - meta dict with scalers, uniques, dims, priors, etc.
    """
    df = load_tabular(cfg["data"]["input_csv"])
    num_cols: list[str] = cfg["data"]["numerical"]
    cat_cols: list[str] = cfg["data"]["categorical"]

    meta: dict = {
        "df": df,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_uniques": {},
        "scaler": {},
    }

    # Standardize numerics
    X_num = []
    for c in num_cols:
        x = df[c].values.astype(np.float32)
        m, s = float(np.mean(x)), float(np.std(x) + 1e-6)
        meta["scaler"][c] = (m, s)
        X_num.append(((x - m) / s)[:, None])
    X_num = np.hstack(X_num) if X_num else np.zeros((len(df), 0), dtype=np.float32)

    # One-hot categoricals (and cache uniques)
    one_hots = []
    for c in cat_cols:
        u = np.unique(df[c].values)
        meta["cat_uniques"][c] = u
        one_hots.append(_one_hot(df[c].values, u))
    X_cat = np.hstack(one_hots) if one_hots else np.zeros((len(df), 0), dtype=np.float32)

    # Dimensions
    meta["num_dim"] = X_num.shape[1]
    meta["cond_dim"] = X_cat.shape[1]          # condition vector is the concatenated one-hots
    meta["gen_out_dim"] = meta["num_dim"] + sum(len(u) for u in meta["cat_uniques"].values())

    # Priors per categorical (by frequency in real df)
    priors = {}
    for c in cat_cols:
        u = meta["cat_uniques"][c]
        p = np.array([(df[c].values == v).mean() for v in u], dtype=np.float32)
        priors[c] = p / (p.sum() if p.sum() > 0 else 1.0)
    meta["priors"] = priors

    X_real = np.hstack([X_num, X_cat]).astype(np.float32)
    return df, X_real, meta


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_ctgan(cfg: dict) -> tuple[dict, dict]:
    """
    Train a conditional GAN:
      - D sees [features, condition]
      - G gets noise z + condition and is trained with:
          * Adversarial loss (BCE)
          * Condition-consistency loss (CE) on selected categorical columns
    Returns:
      (generator_state_dict, meta_dict)
    """
    df, X_real, meta = _prep_ctgan(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    G = MLPGenerator(
        cfg["model"]["latent_dim"],
        meta["gen_out_dim"],
        cfg["model"]["hidden_dim"],
        cond_dim=meta["cond_dim"],
    ).to(device)
    D = MLPDiscriminator(meta["gen_out_dim"] + meta["cond_dim"], cfg["model"]["hidden_dim"]).to(device)

    # Hyper-params
    lr_g = float(cfg["train"]["lr_g"])
    lr_d = float(cfg["train"]["lr_d"])
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    dp_on = cfg.get("dp", {}).get("enabled", False)

    cond_columns = set(cfg["model"].get("cond_columns", []))   # enforce on these
    cond_weight = float(cfg["model"].get("cond_weight", 1.0))  # CE loss weight
    overrides = cfg.get("sampling", {}).get("overrides", None) # optional biasing at training-time

    g_opt = torch.optim.Adam(G.parameters(), lr=lr_g)
    d_opt = torch.optim.Adam(D.parameters(), lr=lr_d)
    bce = nn.BCEWithLogitsLoss()

    dataset = TensorDataset(torch.tensor(X_real, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        for (x_real,) in loader:
            bs = x_real.size(0)
            x_real = x_real.to(device)
            c_real = x_real[:, meta["num_dim"] :]  # real condition is the one-hot part already in x_real

            # -----------------------
            # Train Discriminator
            # -----------------------
            c_fake_np = _sample_condition(meta, bs, overrides)
            c_fake = torch.tensor(c_fake_np, dtype=torch.float32, device=device)

            z = torch.randn(bs, cfg["model"]["latent_dim"], device=device)
            g_raw = G(z, c_fake)

            d_real = D(torch.cat([x_real, c_real], dim=1))
            d_fake = D(torch.cat([g_raw.detach(), c_fake], dim=1))

            d_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            d_opt.zero_grad()
            d_loss.backward()
            if dp_on:
                sanitize_gradients(D.parameters())
            d_opt.step()

            # -----------------------
            # Train Generator
            # -----------------------
            z = torch.randn(bs, cfg["model"]["latent_dim"], device=device)
            g_raw = G(z, c_fake)

            # Adversarial objective: fool D
            adv_loss = bce(D(torch.cat([g_raw, c_fake], dim=1)), torch.ones(bs, 1, device=device))

            # Condition-consistency: categorical outputs should match sampled condition one-hots
            outs = _split_gen_output(meta, g_raw)
            ce_sum = 0.0
            offset = 0
            for j, cname in enumerate(meta["cat_cols"]):
                k = len(meta["cat_uniques"][cname])
                if cname in cond_columns and k > 1:
                    logits = outs["cat_segments"][j]                      # [B, K]
                    target = c_fake[:, offset : offset + k].argmax(dim=1)  # [B]
                    ce_sum = ce_sum + nn.functional.cross_entropy(logits, target)
                offset += k

            g_loss = adv_loss + (cond_weight * ce_sum if cond_columns else 0.0)

            g_opt.zero_grad()
            g_loss.backward()
            if dp_on:
                sanitize_gradients(G.parameters())
            g_opt.step()

        if (epoch + 1) % 5 == 0:
            print(f"[CTGAN] epoch {epoch + 1}/{epochs} d={d_loss.item():.3f} g={g_loss.item():.3f}")

    return G.state_dict(), meta


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_ctgan(
    cfg: dict,
    g_state: dict,
    meta: dict,
    n_samples: int | None = None,
    overrides: dict | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic rows honoring (optional) sampling `overrides`.

    Args:
      cfg: config dict (paths + model sizes)
      g_state: trained generator state dict
      meta: metadata produced by `train_ctgan`
      n_samples: number of rows (defaults to len(real_df))
      overrides: dict like:
         {
           "diagnosis_diabetes": {"0": 0.5, "1": 0.5},
           "sex": {"F": 0.6, "M": 0.4}
         }
         Values can be strings/ints; keys must match categorical column names.

    Returns:
      pd.DataFrame with same columns + order as the real data.
    """
    from torch.nn import functional as F  # local import to keep top import minimal

    # Allow passing through or reading from cfg
    if overrides is None:
        overrides = cfg.get("sampling", {}).get("overrides", None)

    df_real = load_tabular(cfg["data"]["input_csv"])
    meta["df"] = df_real  # make sure priors align to this df
    n = n_samples or len(df_real)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = MLPGenerator(
        cfg["model"]["latent_dim"],
        meta["gen_out_dim"],
        cfg["model"]["hidden_dim"],
        cond_dim=meta["cond_dim"],
    ).to(device)
    G.load_state_dict(g_state)
    G.eval()

    data_frames = []
    with torch.no_grad():
        bs = 512
        total = 0
        while total < n:
            cur = min(bs, n - total)
            # Sample a condition vector (optionally biased by overrides)
            c_np = _sample_condition(meta, cur, overrides)
            c = torch.tensor(c_np, dtype=torch.float32, device=device)

            # Generate raw
            z = torch.randn(cur, cfg["model"]["latent_dim"], device=device)
            g_raw = G(z, c)

            outs = _split_gen_output(meta, g_raw)

            # Restore numerics (de-standardize)
            num_vals = outs["num"].cpu().numpy()
            num_cols = meta["num_cols"]
            restored_num = []
            for i, cname in enumerate(num_cols):
                m, s = meta["scaler"][cname]
                restored_num.append((num_vals[:, i] * s + m)[:, None])
            restored_num = np.hstack(restored_num) if restored_num else np.zeros((cur, 0), dtype=np.float32)

            # Decode categoricals: argmax over softmax(logits)
            cats = []
            for j, cname in enumerate(meta["cat_cols"]):
                logits = outs["cat_segments"][j]
                idx = torch.argmax(F.softmax(logits, dim=1), dim=1).cpu().numpy()
                vals = meta["cat_uniques"][cname][idx]
                cats.append(vals[:, None])
            cats = np.hstack(cats) if cats else np.zeros((cur, 0), dtype=object)

            # Assemble batch in the original column order
            row = {}
            for i, cname in enumerate(num_cols):
                row[cname] = restored_num[:, i]
            for j, cname in enumerate(meta["cat_cols"]):
                row[cname] = cats[:, j]

            batch_df = pd.DataFrame(row)[df_real.columns]  # enforce column order
            data_frames.append(batch_df)
            total += cur

    synth_df = pd.concat(data_frames, ignore_index=True)
    return synth_df
