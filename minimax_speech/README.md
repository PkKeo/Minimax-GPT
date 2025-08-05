# MiniMax-Speech PyTorch Implementation

## Directory Structure

- `config.py`: Hyperparameters
- `dataset.py`: Data loading and preprocessing
- `tokenizer.py`: Text & audio tokenizers
- `speaker_encoder.py`: Speaker encoder module
- `ar_transformer.py`: AR Transformer module
- `flow_vae.py`: VAE encoder + normalizing flow + decoder
- `flow_matching.py`: Flow matching transformer
- `train.py`: Training pipeline (VAE pretrain, Flow matching)
- `infer.py`: Inference pipeline (text→waveform)
- `utils.py`: Utility functions (logging, metrics)
- `requirements.txt`: Python dependencies

## Installation
```bash
pip install -r requirements.txt
```

## Training
1. **Pretrain Flow-VAE**:
   ```bash
   python train.py --stage vae
   ```
2. **Train Flow Matching**:
   ```bash
   python train.py --stage flow_matching
   ```

## Inference
```bash
python infer.py --text "Hello world" --ref_audio path/to/audio.wav --output out.wav
```