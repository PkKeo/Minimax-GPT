class Config:
    # Audio
    sample_rate = 22050
    n_mels = 80
    win_length = 1024
    hop_length = 256

    # Tokenizer
    bpe_model_path = "./tokenizer/bpe.model"
    vq_codebook_size = 1024
    vq_frame_rate = 25  # tokens per second

    # Speaker Encoder
    spk_enc_dim = 256
    spk_enc_layers = 6
    spk_enc_channels = 64

    # AR Transformer
    ar_dim = 512
    ar_heads = 8
    ar_layers = 12
    ar_dropout = 0.1

    # Flow-VAE
    vae_latent_dim = 512
    flow_steps = 6
    flow_hidden = 512

    # Flow Matching
    fm_dim = 512
    fm_heads = 8
    fm_layers = 6

    # Training
    batch_size = 16
    lr_vae = 1e-4
    lr_fm = 3e-4
    epochs_vae = 100
    epochs_fm = 200
    device = 'cuda'
    save_dir = './checkpoints'