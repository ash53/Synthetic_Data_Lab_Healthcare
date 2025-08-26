import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils

class VAE(nn.Module):
    def __init__(self, latent=16, hidden=256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, 28*28), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparam(self, mu, logvar):
        std = (0.5*logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        x = self.dec(z)
        return x.view(-1,1,28,28)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def loss_fn(recon, x, mu, logvar):
    bce = torch.nn.functional.binary_cross_entropy(recon, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + kld) / x.size(0)

def train_image_vae(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.MNIST(root="data/raw", train=True, download=True, transform=tfm)
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True)

    vae = VAE(cfg["model"]["latent_dim"], cfg["model"]["hidden_dim"]).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=cfg["train"]["lr"])

    for epoch in range(cfg["train"]["epochs"]):
        vae.train()
        total = 0
        for x,_ in loader:
            x = x.to(device)
            recon, mu, logvar = vae(x)
            loss = loss_fn(recon, x, mu, logvar)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        print(f"[VAE] epoch {epoch+1}/{cfg['train']['epochs']} loss={total/len(loader):.3f}")
    return vae.cpu()

def generate_images(cfg, vae):
    vae.eval()
    with torch.no_grad():
        z = torch.randn(64, cfg["model"]["latent_dim"])
        imgs = vae.decode(z).clamp(0,1)
        vutils.save_image(imgs, cfg["output"]["sample_grid_png"], nrow=8)
    return imgs
