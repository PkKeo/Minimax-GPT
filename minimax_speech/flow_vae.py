import torch
import torch.nn as nn
from config import Config

class VAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_mu = nn.Linear(Config.n_mels * Config.hop_length, Config.vae_latent_dim)
        self.fc_logvar = nn.Linear(Config.n_mels * Config.hop_length, Config.vae_latent_dim)

    def forward(self, mel):
        B, C, T = mel.size()
        x = mel.view(B, -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        return z, mu, logvar

class CouplingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Config.vae_latent_dim//2, Config.flow_hidden),
            nn.ReLU(),
            nn.Linear(Config.flow_hidden, Config.vae_latent_dim)
        )

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        h = self.net(x1)
        s, t = h.chunk(2, dim=-1)
        z2 = x2 * torch.exp(s) + t
        return torch.cat([x1, z2], dim=-1)

    def inverse(self, z):
        z1, z2 = z.chunk(2, dim=-1)
        h = self.net(z1)
        s, t = h.chunk(2, dim=-1)
        x2 = (z2 - t) * torch.exp(-s)
        return torch.cat([z1, x2], dim=-1)

class FlowVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VAEEncoder()
        self.flows = nn.ModuleList([CouplingLayer() for _ in range(Config.flow_steps)])
        self.decoder = nn.Sequential(
            nn.Linear(Config.vae_latent_dim, Config.n_mels * Config.hop_length),
            nn.Unflatten(1, (Config.n_mels, Config.hop_length)),
            nn.ConvTranspose1d(Config.n_mels, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, mel):
        z, mu, logvar = self.encoder(mel)
        z0 = z
        for f in self.flows:
            z0 = f(z0)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(-1).mean()
        recon = self.decoder(z)
        return recon, z, z0, kl