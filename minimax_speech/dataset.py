import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from config import Config
from tokenizer import TextTokenizer, AudioTokenizer

class TTSDataset(Dataset):
    def __init__(self, metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        self.items = [line.split('|') for line in lines]
        self.text_tokenizer = TextTokenizer(Config.bpe_model_path)
        self.audio_tokenizer = AudioTokenizer(
            n_mels=Config.n_mels,
            hop_length=Config.hop_length,
            codebook_size=Config.vq_codebook_size,
            frame_rate=Config.vq_frame_rate
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        text, audio_path = self.items[idx]
        text_ids = self.text_tokenizer.encode(text)
        waveform, sr = torchaudio.load(audio_path)
        if sr != Config.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, Config.sample_rate)
        audio_tokens = self.audio_tokenizer.encode(waveform)
        return {
            'text_ids': torch.LongTensor(text_ids),
            'audio_tokens': torch.LongTensor(audio_tokens)
        }

def collate_fn(batch):
    text_seqs = [item['text_ids'] for item in batch]
    text_lens = [len(seq) for seq in text_seqs]
    max_text_len = max(text_lens)
    padded_texts = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    for i, seq in enumerate(text_seqs):
        padded_texts[i, :len(seq)] = seq

    audio_seqs = [item['audio_tokens'] for item in batch]
    audio_lens = [len(seq) for seq in audio_seqs]
    max_audio_len = max(audio_lens)
    padded_audios = torch.zeros(len(batch), max_audio_len, dtype=torch.long)
    for i, seq in enumerate(audio_seqs):
        padded_audios[i, :len(seq)] = seq

    return {
        'text_ids': padded_texts,
        'text_lens': torch.LongTensor(text_lens),
        'audio_tokens': padded_audios,
        'audio_lens': torch.LongTensor(audio_lens),
    }

def get_dataloader(metadata_path, stage):
    dataset = TTSDataset(metadata_path)
    shuffle = True if stage in ['vae', 'flow_matching'] else False
    return DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )