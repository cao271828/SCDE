# -*- coding: utf-8 -*-
"""abundance-map spatial augmentations. Applied after VCA projection, before ResBlock."""

import torch

@torch.no_grad()
def spatial_shuffle(x, p=0.5):
    """Shuffle spatial positions within each abundance map.
    x: (N, k, H, W)"""
    N, k, H, W = x.shape
    mask = torch.rand(N, device=x.device) < p
    if not mask.any():
        return x
    x_out = x.clone()
    idx = x_out[mask]
    n_aug = idx.shape[0]
    idx_flat = idx.view(n_aug, k, H * W)
    for i in range(n_aug):
        perm = torch.randperm(H * W, device=x.device)
        idx_flat[i] = idx_flat[i, :, perm]
    x_out[mask] = idx_flat.view(n_aug, k, H, W)
    return x_out

@torch.no_grad()
def block_cutmix(x, block_size=3, alpha=0.2):
    """Block-level CutMix on abundance maps.
    x: (N, k, H, W)"""
    N, k, H, W = x.shape
    rand_idx = torch.randperm(N, device=x.device)
    x_mixed = x.clone()
    for i in range(N):
        j = rand_idx[i]
        bh = min(block_size, max(1, torch.randint(1, block_size + 1, (1,), device=x.device).item()))
        bw = min(block_size, max(1, torch.randint(1, block_size + 1, (1,), device=x.device).item()))
        y = torch.randint(0, max(1, H - bh + 1), (1,), device=x.device).item()
        x_c = torch.randint(0, max(1, W - bw + 1), (1,), device=x.device).item()
        x_mixed[i, :, y:y+bh, x_c:x_c+bw] = x[j, :, y:y+bh, x_c:x_c+bw]
    return x_mixed

@torch.no_grad()
def rand_erasing(x, scale=(0.02, 0.1)):
    """Random rectangular erasing on abundance maps.
    x: (N, k, H, W)"""
    N, k, H, W = x.shape
    area = H * W
    x_out = x.clone()
    for i in range(N):
        target_area = (scale[0] + torch.rand(1, device=x.device).item() * (scale[1] - scale[0])) * area
        aspect = torch.exp(torch.randn(1, device=x.device) * 0.5).item()
        h = min(H, max(1, int(round((target_area * aspect) ** 0.5))))
        w = min(W, max(1, int(round((target_area / aspect) ** 0.5))))
        y = torch.randint(0, H - h + 1, (1,), device=x.device).item()
        x_c = torch.randint(0, W - w + 1, (1,), device=x.device).item()
        x_out[i, :, y:y+h, x_c:x_c+w] = 0
    return x_out

@torch.no_grad()
def channel_dropout(x, p=0.1):
    """Drop random channels (endmembers) in abundance map.
    x: (N, k, H, W)"""
    if p <= 0:
        return x
    N, k, H, W = x.shape
    mask = torch.rand(N, k, 1, 1, device=x.device) > p
    return x * mask.float()

def apply_aug(x, aug_type, aug_param):
    """Apply augmentation to abundance maps.
    x: (N, k, H, W) after VCA projection + squeeze
    aug_type: str, one of 'shuffle','cutmix','erase','chdrop','none'
    aug_param: float/int/string, parameter for augmentation
    """
    if aug_type == 'none' or aug_type is None:
        return x
    elif aug_type == 'shuffle':
        p = float(aug_param) if aug_param else 0.5
        return spatial_shuffle(x, p=p)
    elif aug_type == 'cutmix':
        bs = int(aug_param) if aug_param else 3
        return block_cutmix(x, block_size=bs)
    elif aug_type == 'erase':
        if isinstance(aug_param, (tuple, list)):
            return rand_erasing(x, scale=tuple(aug_param))
        elif isinstance(aug_param, str) and ',' in aug_param:
            parts = [float(p.strip()) for p in aug_param.split(',')]
            return rand_erasing(x, scale=tuple(parts[:2]))
        elif aug_param == 3:
            return rand_erasing(x, scale=(0.02, 0.1))
        elif aug_param == 4:
            return rand_erasing(x, scale=(0.05, 0.2))
        else:
            return rand_erasing(x)
    elif aug_type == 'chdrop':
        p = float(aug_param) if aug_param else 0.1
        return channel_dropout(x, p=p)
    else:
        return x