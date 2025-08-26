import torch

def add_noise(tensor, sigma):
    return tensor + sigma * torch.randn_like(tensor)

def sanitize_gradients(params, clip=1.0, sigma=0.5):
    torch.nn.utils.clip_grad_norm_(params, clip)
    for p in params:
        if p.grad is not None:
            p.grad = add_noise(p.grad, sigma)
