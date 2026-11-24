"""Stable population normalization without changing the PPO scale convention."""
import torch

VERSION = 'centered-float64-advantage-normalization-v1'


def normalize_advantages(values):
    if values.ndim != 1 or not values.is_floating_point() or not torch.isfinite(values).all():
        raise ValueError('finite floating-point advantage vector required')
    if values.numel() <= 1:
        return values.clone()  # Existing PPO leaves a singleton unnormalized.
    centered = values.to(torch.float64)
    centered = centered - centered[0]
    centered = centered - centered.mean()
    scale = centered.square().mean().sqrt().clamp_min(1e-8)
    return (centered / scale).to(values.dtype)
