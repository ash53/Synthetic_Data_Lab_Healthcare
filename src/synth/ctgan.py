import torch, torch.nn as nn
import numpy as np, pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from src.models.generator import MLPGenerator
from src.models.discriminator import MLPDiscriminator
from src.synth.dp import sanitize_gradients
from src.utils.io import load_tabular

def _one_hot(values, uniques):
    idx = np.array([np.where(uniques==v)[0][0] for v in values])
    oh = np.zeros((len(values), len(uniques)), dtype=np.float32)
    oh[np.arange(len(values)), idx] = 1.0
    return oh

def _sample_condition(meta, batch, overrides=None):
    """Sample one-hot blocks for each categorical column; apply per-column overrides if given."""
    import numpy as np
    blocks = []
    for c in meta["cat_cols"]:
        u = meta["cat_uniques"][c]
        # base prior
        p = np.array([(meta["df"][c].values == val).mean() for val in u], dtype=np.float32)
        p = p / p.sum()
        # override?
        if overrides and c in overrides:
            o = overrides[c]
            # map override weights onto uniques order
            w = np.array([float(o.get(str(val), o.get(int(val), o.get(val, 0.0)))) for val in u], dtype=np.float32)
            if w.sum() > 0:
                p = w / w.sum()
        idx = np.random.choice(len(u), size=batch, p=p)
        oh = np.zeros((batch, len(u)), dtype=np.float32)
        oh[np.arange(batch), idx] = 1.0
        blocks.append(oh)
    return np.hstack(blocks).astype(np.float32) if blocks else np.zeros((batch, 0), dtype=np.float32)


def _prep_ctgan(cfg):
    df = load_tabular(cfg["data"]["input_csv"])
    num_cols = cfg["data"]["numerical"]
    cat_cols = cfg["data"]["categorical"]
    meta = {"num_cols": num_cols, "cat_cols": cat_cols, "cat_uniques": {}, "scaler": {}}

    # Standardize numerical
    X_num = []
    for c in num_cols:
        x = df[c].values.astype(np.float32)
        m, s = float(np.mean(x)), float(np.std(x) + 1e-6)
        meta["scaler"][c] = (m, s)
        X_num.append(((x - m)/s)[:,None])
    X_num = np.hstack(X_num) if X_num else np.zeros((len(df),0), dtype=np.float32)

    # One-hot categorical
    one_hots = []
    for c in cat_cols:
        u = np.unique(df[c].values)
        meta["cat_uniques"][c] = u
        one_hots.append(_one_hot(df[c].values, u))
    X_cat_oh = np.hstack(one_hots) if one_hots else np.zeros((len(df),0), dtype=np.float32)

    gen_out_dim = X_num.shape[1] + sum(len(u) for u in meta["cat_uniques"].values())
    meta["cond_dim"] = X_cat_oh.shape[1]
    meta["gen_out_dim"] = gen_out_dim
    meta["cat_cols"] = cat_cols  # order-preserving

    # Build priors dict keyed by column
    priors = {}
    for c in cat_cols:
        u = meta["cat_uniques"][c]
        counts = np.array([(df[c].values == val).mean() for val in u], dtype=np.float32)
        priors[c] = counts/np.sum(counts)
    meta["priors"] = priors

    X_real = np.hstack([X_num, X_cat_oh]).astype(np.float32)
    return df, X_real, X_num.shape[1], meta

def _override_probs_for_column(uniques, base_probs, override_dict):
    if override_dict is None:
        return base_probs
    # Map override keys (str/int) onto uniques array order
    p = np.array([
        float(override_dict.get(str(val), override_dict.get(int(val), override_dict.get(val, 0.0))))
        for val in uniques
    ], dtype=np.float32)
    if p.sum() > 0:
        p = p / p.sum()
        return p
    return base_probs

def _sample_condition(meta, batch, overrides=None):
    cond_blocks = []
    for c in meta["cat_cols"]:
        u = meta["cat_uniques"][c]
        p = meta["priors"][c]
        if overrides and c in overrides:
            p = _override_probs_for_column(u, p, overrides[c])
        idx = np.random.choice(len(u), size=batch, p=p)
        oh = np.zeros((batch, len(u)), dtype=np.float32)
        oh[np.arange(batch), idx] = 1.0
        cond_blocks.append(oh)
    if len(cond_blocks)==0:
        return np.zeros((batch,0), dtype=np.float32)
    return np.hstack(cond_blocks).astype(np.float32)

def _split_gen_output(meta, g_out):
    num_dim = sum(1 for _ in meta["scaler"].keys())
    pos = num_dim
    outs = {}
    outs["num"] = g_out[:, :num_dim]
    cat_segments = []
    for c in meta["cat_cols"]:
        k = len(meta["cat_uniques"][c])
        cat_segments.append(g_out[:, pos:pos+k])
        pos += k
    outs["cat_segments"] = cat_segments
    return outs

def train_ctgan(cfg):
    df, X_real, num_dim, meta = _prep_ctgan(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    G = MLPGenerator(cfg["model"]["latent_dim"], meta["gen_out_dim"], cfg["model"]["hidden_dim"], cond_dim=meta["cond_dim"]).to(device)
    D = MLPDiscriminator(meta["gen_out_dim"] + meta["cond_dim"], cfg["model"]["hidden_dim"]).to(device)

    lr_g = float(cfg["train"]["lr_g"])
    lr_d = float(cfg["train"]["lr_d"])
    batch_size = int(cfg["train"]["batch_size"])
    epochs = int(cfg["train"]["epochs"])
    tau = float(cfg["model"].get("gumbel_tau", 0.5))

    g_opt = torch.optim.Adam(G.parameters(), lr=lr_g)
    d_opt = torch.optim.Adam(D.parameters(), lr=lr_d)

    dataset = TensorDataset(torch.tensor(X_real, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    bce = torch.nn.BCEWithLogitsLoss()
    dp_on = cfg.get("dp",{}).get("enabled", False)
    overrides = cfg.get("sampling", {}).get("overrides", None)

    for epoch in range(epochs):
        for (x_real,) in loader:
            bs = x_real.size(0)
            # Sample condition (with optional overrides)
            c = torch.tensor(_sample_condition(meta, bs, overrides), device=device)
            x_real = x_real.to(device)

            # Train D
            z = torch.randn(bs, cfg["model"]["latent_dim"], device=device)
            g_raw = G(z, c)
            d_real = D(torch.cat([x_real, c], dim=1))
            d_fake = D(torch.cat([g_raw.detach(), c], dim=1))
            d_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            d_opt.zero_grad(); d_loss.backward()
            if dp_on: sanitize_gradients(D.parameters())
            d_opt.step()

            # Train G
            z = torch.randn(bs, cfg["model"]["latent_dim"], device=device)
            g_raw = G(z, c)
            g_loss = bce(D(torch.cat([g_raw, c], dim=1)), torch.ones(bs,1, device=device))
            g_opt.zero_grad(); g_loss.backward()
            if dp_on: sanitize_gradients(G.parameters())
            g_opt.step()

        if (epoch+1) % 5 == 0:
            print(f"[CTGAN] epoch {epoch+1}/{epochs} d={d_loss.item():.3f} g={g_loss.item():.3f}")

    meta["num_dim"] = num_dim
    return G.state_dict(), meta

def generate_ctgan(cfg, g_state, meta, n_samples=None, overrides=None):
    import torch, numpy as np, pandas as pd
    from torch.nn import functional as F
    from src.utils.io import load_tabular

    df_real = load_tabular(cfg["data"]["input_csv"])
    # keep real df inside meta for sampler priors
    meta["df"] = df_real
    n = n_samples or len(df_real)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    G = MLPGenerator(cfg["model"]["latent_dim"], meta["gen_out_dim"], cfg["model"]["hidden_dim"], cond_dim=meta["cond_dim"]).to(device)
    G.load_state_dict(g_state); G.eval()

    data = []
    with torch.no_grad():
        bs = 512
        total = 0
        while total < n:
            cur = min(bs, n - total)
            c = torch.tensor(_sample_condition(meta, cur, overrides), device=device, dtype=torch.float32)
            z = torch.randn(cur, cfg["model"]["latent_dim"], device=device)
            g_raw = G(z, c)

            outs = _split_gen_output(meta, g_raw)
            # restore numerics
            num_vals = outs["num"].cpu().numpy()
            num_cols = list(meta["scaler"].keys())
            restored_num = []
            for i, cname in enumerate(num_cols):
                m, s = meta["scaler"][cname]
                restored_num.append((num_vals[:, i] * s + m)[:, None])
            restored_num = np.hstack(restored_num) if restored_num else np.zeros((cur, 0), dtype=np.float32)

            # decode categoricals
            cats = []
            for j, cname in enumerate(meta["cat_cols"]):
                logits = outs["cat_segments"][j]
                idx = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()
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

