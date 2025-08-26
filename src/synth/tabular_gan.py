import torch, pandas as pd, numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.models.generator import MLPGenerator
from src.models.discriminator import MLPDiscriminator
from src.synth.dp import sanitize_gradients
from src.utils.io import load_tabular

def _prep_tabular(cfg):
    df = load_tabular(cfg["data"]["input_csv"])
    cols = cfg["data"]["numerical"] + cfg["data"]["categorical"]
    X = df[cols].values.astype(np.float32)
    meta = {"in_dim": X.shape[1], "cols": cols}
    return df, X, meta

def train_tabular_gan(cfg):
    _, X, meta = _prep_tabular(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    G = MLPGenerator(cfg["model"]["latent_dim"], meta["in_dim"], cfg["model"]["hidden_dim"]).to(device)
    D = MLPDiscriminator(meta["in_dim"], cfg["model"]["hidden_dim"]).to(device)

    g_opt = torch.optim.Adam(G.parameters(), lr=cfg["train"]["lr_g"])
    d_opt = torch.optim.Adam(D.parameters(), lr=cfg["train"]["lr_d"])

    dataset = TensorDataset(torch.tensor(X))
    loader = DataLoader(dataset, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True)
    bce = torch.nn.BCEWithLogitsLoss()

    dp_on = cfg.get("dp", {}).get("enabled", False)
    for epoch in range(cfg["train"]["epochs"]):
        for (x_real,) in loader:
            x_real = x_real.to(device)
            bs = x_real.size(0)

            # Train D
            z = torch.randn(bs, cfg["model"]["latent_dim"], device=device)
            x_fake = G(z).detach()
            d_real = D(x_real)
            d_fake = D(x_fake)
            d_loss = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            d_opt.zero_grad(); d_loss.backward()
            if dp_on: sanitize_gradients(D.parameters())
            d_opt.step()

            # Train G
            z = torch.randn(bs, cfg["model"]["latent_dim"], device=device)
            fake = G(z)
            g_loss = bce(D(fake), torch.ones(bs,1, device=device))
            g_opt.zero_grad(); g_loss.backward()
            if dp_on: sanitize_gradients(G.parameters())
            g_opt.step()
        if (epoch+1) % 5 == 0:
            print(f"[TabularGAN] epoch {epoch+1}/{cfg['train']['epochs']} d={d_loss.item():.3f} g={g_loss.item():.3f}")
    return G.state_dict(), meta

def generate_tabular(cfg, g_state, meta):
    import torch
    df = load_tabular(cfg["data"]["input_csv"]).copy()
    cols = meta["cols"]
    G = MLPGenerator(cfg["model"]["latent_dim"], meta["in_dim"], cfg["model"]["hidden_dim"])
    G.load_state_dict(g_state); G.eval()
    n = len(df)
    with torch.no_grad():
        z = torch.randn(n, cfg["model"]["latent_dim"])
        synth = G(z).cpu().numpy()
    df[cols] = synth
    return df
