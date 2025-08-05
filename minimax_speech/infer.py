import torch
import torchaudio
from config import Config
from tokenizer import TextTokenizer, AudioTokenizer
from speaker_encoder import SpeakerEncoder
from ar_transformer import ARTransformer
from flow_matching import FlowMatchTransformer
from flow_vae import FlowVAE

def inference(text, ref_audio_path, out_path):
    spk_enc = SpeakerEncoder().to(Config.device).eval()
    ar = ARTransformer(vocab_size=Config.vq_codebook_size).to(Config.device).eval()
    fm = FlowMatchTransformer().to(Config.device).eval()
    decoder = FlowVAE().decoder.to(Config.device).eval()

    tt = TextTokenizer(Config.bpe_model_path)
    tokens = torch.LongTensor(tt.encode(text)).unsqueeze(0).to(Config.device)

    wav, sr = torchaudio.load(ref_audio_path)
    if sr != Config.sample_rate:
        wav = torchaudio.functional.resample(wav, sr, Config.sample_rate)
    mel = AudioTokenizer(Config.n_mels, Config.hop_length, Config.vq_codebook_size, Config.vq_frame_rate).mel_transform(wav)
    v = spk_enc(mel.to(Config.device))

    ar_logits = ar(tokens)
    c = ar_logits.softmax(-1)

    T = int(ar_logits.size(1) * (Config.sample_rate/Config.hop_length/Config.vq_frame_rate))
    t = torch.linspace(0,1,T).to(Config.device)
    z_hat = fm(c, v, t)

    wav_hat = decoder(z_hat)
    torchaudio.save(out_path, wav_hat.cpu(), Config.sample_rate)

if __name__ == '__main__':
    inference("Hello world", "ref.wav", "out.wav")