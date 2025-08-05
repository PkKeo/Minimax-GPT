import sentencepiece as spm
import torch
import torchaudio
from config import Config

class TextTokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
    def encode(self, text):
        return self.sp.EncodeAsIds(text)
    def decode(self, ids):
        return self.sp.DecodeIds(ids)

class AudioTokenizer:
    def __init__(self, n_mels, hop_length, codebook_size, frame_rate):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=Config.sample_rate,
            n_fft=Config.win_length,
            hop_length=hop_length,
            n_mels=n_mels
        )
        self.codebook = torch.randn(codebook_size, n_mels)
        self.frame_rate = frame_rate

    def encode(self, wav):
        mel = self.mel_transform(wav).squeeze(0).transpose(0,1)
        dist = (mel.unsqueeze(1) - self.codebook.unsqueeze(0)).pow(2).sum(-1)
        tokens = dist.argmin(-1)
        step = int(Config.sample_rate / self.frame_rate / Config.hop_length)
        return tokens[::step]

    def decode(self, tokens):
        mel = self.codebook[tokens]
        return mel.transpose(0,1).unsqueeze(0)