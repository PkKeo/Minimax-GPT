import torch

def save_checkpoint(model, optimizer, path):
    torch.save({'model': model.state_dict(),
                'opt': optimizer.state_dict()}, path)

def compute_loss(recon, mel, kl):
    recon_loss = ((recon - mel)**2).mean()
    return recon_loss + kl