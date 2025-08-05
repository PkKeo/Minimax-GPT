import torch
import torch.nn as nn
from config import Config

class SpeakerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_ch = 1
        for _ in range(Config.spk_enc_layers):
            layers += [
                nn.Conv1d(in_ch, Config.spk_enc_channels, kernel_size=3, padding=1),
                nn.BatchNorm1d(Config.spk_enc_channels),
                nn.ReLU(inplace=True)
            ]
            in_ch = Config.spk_enc_channels
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(Config.spk_enc_channels, Config.spk_enc_dim)

    def forward(self, mel):
        x = self.conv(mel)
        x = self.pool(x).squeeze(-1)
        emb = self.fc(x)
        return emb