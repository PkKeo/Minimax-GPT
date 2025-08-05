import torch
import torch.nn as nn
from config import Config

class FlowMatchTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Conv1d(Config.ar_dim, Config.fm_dim, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(scale_factor=4, mode='nearest')
        self.time_emb = nn.Linear(1, Config.fm_dim)
        layers = []
        for _ in range(Config.fm_layers):
            layers.append(nn.TransformerEncoderLayer(d_model=Config.fm_dim, nhead=Config.fm_heads))
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(Config.fm_dim)
        self.out = nn.Linear(Config.fm_dim, Config.vae_latent_dim)

    def forward(self, c, v, t):
        x = self.input_proj(c.transpose(1,2))
        x = self.upsample(x).transpose(1,2)
        B, T, D = x.size()
        v_exp = v.unsqueeze(1).expand(-1, T, -1)
        t_emb = self.time_emb(t.unsqueeze(-1))
        h = x + v_exp + t_emb
        for l in self.layers:
            h = l(h)
        h = self.norm(h)
        z_hat = self.out(h)
        return z_hat