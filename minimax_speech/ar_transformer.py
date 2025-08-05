import torch
import torch.nn as nn
from config import Config

class ARTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, Config.ar_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 1000, Config.ar_dim))
        layers = []
        for _ in range(Config.ar_layers):
            layers.append(nn.TransformerEncoderLayer(
                d_model=Config.ar_dim, nhead=Config.ar_heads, dropout=Config.ar_dropout))
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(Config.ar_dim)
        self.out = nn.Linear(Config.ar_dim, Config.vq_codebook_size)

    def forward(self, tokens):
        x = self.token_emb(tokens) + self.pos_emb[:,:tokens.size(1)]
        for l in self.layers:
            x = l(x)
        x = self.norm(x)
        logits = self.out(x)
        return logits