import argparse
import torch
from config import Config
from dataset import get_dataloader
from flow_vae import FlowVAE
from flow_matching import FlowMatchTransformer
from speaker_encoder import SpeakerEncoder
from ar_transformer import ARTransformer
from tokenizer import TextTokenizer, AudioTokenizer
from utils import save_checkpoint, compute_loss

def train_vae():
    vae = FlowVAE().to(Config.device)
    opt = torch.optim.Adam(vae.parameters(), lr=Config.lr_vae)
    loader = get_dataloader('metadata.txt', stage='vae')
    for ep in range(Config.epochs_vae):
        for batch in loader:
            mel = AudioTokenizer(Config.n_mels, Config.hop_length, Config.vq_codebook_size, Config.vq_frame_rate).decode(batch['audio_tokens'])
            recon, z, z0, kl = vae(mel.to(Config.device))
            loss = compute_loss(recon, mel.to(Config.device), kl)
            opt.zero_grad()
            loss.backward()
            opt.step()
        save_checkpoint(vae, opt, f"vae_ep{ep}.pt")

def train_flow_matching():
    vae = FlowVAE().to(Config.device)
    vae.load_state_dict(torch.load('vae_final.pt')['model'])
    vae.eval()

    fm = FlowMatchTransformer().to(Config.device)
    spk_enc = SpeakerEncoder().to(Config.device)
    ar = ARTransformer(vocab_size=Config.vq_codebook_size).to(Config.device)
    opt = torch.optim.Adam(list(fm.parameters()) + list(spk_enc.parameters()) + list(ar.parameters()), lr=Config.lr_fm)
    loader = get_dataloader('metadata.txt', stage='flow_matching')

    for ep in range(Config.epochs_fm):
        for batch in loader:
            tokens = batch['text_ids'].to(Config.device)
            audio_tokens = batch['audio_tokens'].to(Config.device)
            ar_logits = ar(tokens)
            c = ar_logits.softmax(-1)
            mel = AudioTokenizer(Config.n_mels, Config.hop_length, Config.vq_codebook_size, Config.vq_frame_rate).decode(audio_tokens)
            v = spk_enc(mel.to(Config.device))
            with torch.no_grad():
                _, z, z0, _ = vae(mel.to(Config.device))
            t = torch.linspace(0,1,z.size(1)).to(Config.device)
            z_hat = fm(c, v, t)
            loss = ((z_hat - z)**2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        save_checkpoint(fm, opt, f"fm_ep{ep}.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['vae','flow_matching'], required=True)
    args = parser.parse_args()
    if args.stage == 'vae':
        train_vae()
    else:
        train_flow_matching()