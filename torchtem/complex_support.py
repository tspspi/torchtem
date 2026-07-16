from __future__ import annotations

import torch


def abs2(x: torch.Tensor) -> torch.Tensor:
    """Absolute square for complex or real tensors."""
    if torch.is_complex(x):
        return x.real.square() + x.imag.square()
    return x.square()


def complex_exponential(x: torch.Tensor) -> torch.Tensor:
    """Compute exp(i x), preserving stability for complex inputs."""
    if torch.is_complex(x):
        return torch.exp(1.0j * x)
    return torch.complex(torch.cos(x), torch.sin(x))


def complex_exponential_scaled(x: torch.Tensor, scale: float | torch.Tensor) -> torch.Tensor:
    """Compute exp(i * scale * x) without avoidable temporaries."""
    base_dtype = x.real.dtype if torch.is_complex(x) else x.dtype
    scale_tensor = torch.as_tensor(scale, device=x.device, dtype=base_dtype)
    if torch.is_complex(x):
        return torch.exp(1.0j * scale_tensor * x)
    phase = scale_tensor * x
    return torch.complex(torch.cos(phase), torch.sin(phase))
