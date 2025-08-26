import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.generator import MLPGenerator
from src.models.discriminator import MLPDiscriminator
from src.synth.dp import sanitize_gradients
from src.utils.io import load_tabular


def _one_hot(values, uniques):
    idx = np.array([np.where(uniques == v)[0][0] for v in values])
    oh = np.zeros((len(values), len(uniques)), dtype=np.float32)
    oh[np.arange(len(values)), idx] = 1.0
    return oh


def _build_cond_slices(meta):
    """
    Build slice indices for each categorical column inside the full condition vector.
    Returns dict: {col_name: (start, end)}
    """
    slices = {}
    pos = 0
    for c in meta["cat_cols"]:
        k = len(meta["cat_uniques"][c])
        slices[c] = (pos, pos + k)
        pos += k
    return slices


def _prep_ctgan(cfg):
    """
    Prepare standardized numerics, one-hot categoricals, priors, and metadata.
    """
    df = load_tabular(cfg["data"]["input_csv"])
    num_cols = cfg["data"]["numerical"]
    cat_cols = cfg["data"]["categorical"]

    meta = {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "cat_uniques": {},
        "scaler": {},
        "df": df,  # keep real df for priors
    }

    # Standardize numeric columns
    X_num = []
    for c in num_cols:
        x = df[c].values.astype(np.float32)
        m, s = float(np.mean(x)), float(np.std(x) + 1e-6)
        meta["scaler"][c] = (m, s)
        X_num.append(((x - m) / s)[:, None])
    X_num = np.hstack(X_num) if X_num else np.zeros((len(df), 0), dtype=np.float32)

    # One-hot categoricals for discriminator and cond vector
    one_hots = []
    for c in cat_cols:
        u = np.unique(df[c].values)
        meta["cat_uniques"][c] = u
        one_hots.append(_one_hot(df[c].values, u))
    X_cat_oh = np.hstack(one_hots) if one_hots else np.zeros((len(df), 0), dtype=np.float32)

    # Dimensions
    gen_out_dim = X_num.shape[1] + sum(len(u) for u in meta["cat_uniques"].values())
    meta["cond_dim"] = X_cat_oh.shape[1]
    meta["gen_out_dim"] = gen_out_dim
    meta["num_dim"] = X_num.shape[1]

    # Priors per categorical column (by original data frequency)
    priors = {}
    for c in cat_cols:
        u = meta["cat_uniques"][c]
        p = np.array([(df[c].values == val).mean() for val in u], dtype=np.float32)
        priors[c] = p / p.sum()
    meta["priors"] = priors

    # Real input to D is [num_std, cat_onehot], and condition vector is the same one-hot block(s)
    X_real = np.hstack([X_num, X_cat_oh]).astype(np.float32)

    # Condition slices for each categorical
    meta["cond_slices"] = _build_cond_slices(meta)

    return df, X_real, meta


def _override_probs_for_column(uniques, base_probs, override_dict):
    """Apply per-value override weights if provided; fall back to base if invalid."""
    if override_dict is None:
        return base_probs
    w = np.array(
        [float(override_dict.get(str(val), override_dict.get(int(val), override_dict.get(val, 0.0))))
         for val in uniques],
        dtype=np.float32,
    )
    s = w.sum()
    if s > 0:
        return w / s
    return base_probs


def _sample_condition(meta, batch, overrides=None):
    """
    Build a full condition vector by sampling each categorical column.
    Overrides can specify desired category probabilities per column.
    """
    blocks = []
    for c in meta["cat_cols"]:
        u = meta["cat_uniques"][c]
        p = meta["priors"][c]
        if overrides and c in overrides:
            p = _override_probs_for_column(u, p, overrides[c])
        idx = np.random.choice(len(u), size=batch, p=p)
        oh = np.zeros((batch, len(u)), dtype=np.float32)
        oh[np.arange(batch), idx] = 1.0
        blocks.append(oh)
    return np.hstack(blocks).astype(np.float32) if blocks else np.zeros((batch, 0), dtype=np.float32)


def _split_gen_output(meta, g_out):
    """
    Split generator raw output into numeric and per-categorical segments.
    Returns:
      num:  [B, num_dim]
      cat_segments: list of [B, K_c] raw logits per categorical column
    """
    num_dim = meta["num_dim"]
    pos = num_dim
    outs = {"num": g_out[:, :num_dim], "cat_segments": []}
    for c in meta["cat_cols"]:
        k = len(meta["cat_uniques"][c])
        outs["cat_segments"].append(g_out[:, pos:pos + k])
        pos += k
    return outs


def train_ctgan(cfg):
    """
    Train conditional tabular GAN with condition-consistency loss so G matches sampled conditions.
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

    # Hyperparams
    lr_g = float(cfg["train"]["lr_g"])
    lr_d = float(cfg["train"]["lr_d"])
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    dp_on = cfg.get("dp", {}).get("enabled", False)

    cond_columns = set(cfg["model"].get("cond_columns", []))  # which cats to enforce
    cond_weight = float(cfg["model"].get("cond_weight", 1.0))  # CE loss weight
    overrides = cfg.get("sampling", {}).get("overrides", None)

    g_opt = torch.optim.Adam(G.parameters(), lr=lr_g)
    d_opt = torch.optim.Adam(D.parameters(), lr=lr_d)
    bce = torch.nn.BCEWithLogitsLoss()

    dataset = TensorDataset(torch.tensor(X_real, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        for (x_real,) in loader:
            bs = x_real.size(0)
            x_real = x_real.to(device)
            # Real condition vector is exactly the one-hot part of x_real
            c_real = x_real[:, meta["num_dim"] :]

            # -----------------
            # Train D
            # -----------------
            # Sample fake condition (with optional overrides)
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

            # -----------------
            # Train G
            # -----------------
            z = torch.randn(bs, cfg["model"]["latent_dim"], device=device)
            g_raw = G(z, c_fake)
            adv_loss = bce(D(torch.cat([g_raw, c_fake], dim=1)), torch.ones(bs, 1, device=device))

            # Condition-consistency loss: force selected categorical outputs to match c_fake
            outs = _split_gen_output(meta, g_raw)
            ce_sum = 0.0
            offset = 0
            for j, cname in enumerate(meta["cat_cols"]):
                k = len(meta["cat_uniques"][cname])
                if cname in cond_columns:
                    logits = outs["cat_segments"][j]               # [B, K]
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


def generate_ctgan(cfg, g_state, meta, n_samples=None, overrides=None):
    """
    Generate synthetic rows honoring (optional) sampling overrides via the condition vector.
    """
    from torch.nn import functional as F

    df_real = load_tabular(cfg["data"]["input_csv"])
    meta["df"] = df_real  # ensure priors align with the current real df
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

    data = []
    with torch.no_grad():
        bs = 512
        total = 0
        while total < n:
            cur = min(bs, n - total)
            c_np = _sample_condition(meta, cur, overrides)
            c = torch.tensor(c_np, dtype=torch.float32, device=device)

            z = torch.randn(cur, cfg["model"]["latent_dim"], device=device)
            g_raw = G(z, c)
            outs = _split_gen_output(meta, g_raw)

            # De-standardize numerics
            num_vals = outs["num"].cpu().numpy()
            num_cols = meta["num_cols"]
            restored_num = []
            for i, cname in enumerate(num_cols):
                m, s = meta["scaler"][cname]
                restored_num.append((num_vals[:, i] * s + m)[:, None])
            restored_num = np.hstack(restored_num) if restored_num else np.zeros((cur, 0), dtype=np.float32)

            # Decode categoricals via argmax
            cats = []
            for j, cname in enumerate(meta["cat_cols"]):
                logits = outs["cat_segments"][j]
                idx = torch.argmax(F.softmax(logits, dim=1), dim=1).cpu().numpy()
                vals = meta["cat_uniques"][cname][idx]
                cats.append(vals[:, None])
            cats = np.hstack(cats) if cats else np.zeros((cur, 0), dtype=object)

            row = {}
            for i, cname in enumerate(num_cols):
                row[cname] = restored_num[:, i]
            for j, cname in enumerate(meta["cat_cols"]):
                row[cname] = cats[:, j]

            batch_df = pd.DataFrame(row)[df_real.columns]
            data.append(batch_df)
            total += cur

    return pd.concat(data, ignore_index=True)
