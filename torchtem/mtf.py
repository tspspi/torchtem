from __future__ import annotations

import torch
from torch import nn

from tspi.torchtem.physics import reciprocal_mesh


def default_mtf_func(k: torch.Tensor, c0: float, c1: float, c2: float, c3: float) -> torch.Tensor:
    return (c0 - c1) / (1.0 + (k / (2.0 * c2)) ** abs(c3)) + c1


class MTF(nn.Module):
    """Apply a detector modulation transfer function in Fourier space."""

    def __init__(self, func=None, **params) -> None:
        super().__init__()
        self.f = default_mtf_func if func is None else func
        self.params = params

    def kernel(
        self,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        *,
        device=None,
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        ky, kx = reciprocal_mesh(gpts, sampling, device=device, dtype=dtype)
        k = torch.sqrt(ky.square() + kx.square())
        return self.f(k, **self.params).to(dtype)

    def forward(self, image: torch.Tensor, *, sampling: tuple[float, float]) -> torch.Tensor:
        gpts = (image.shape[-2], image.shape[-1])
        kernel = self.kernel(gpts, sampling, device=image.device, dtype=torch.float64)
        fourier = torch.fft.fft2(image, norm="ortho")
        filtered = torch.fft.ifft2(fourier * torch.sqrt(torch.clamp(kernel, min=0.0)), norm="ortho")
        return filtered.real
