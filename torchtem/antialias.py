from __future__ import annotations

import math

import torch
from torch import nn

from torchtem.physics import reciprocal_mesh


def antialias_aperture(
    gpts: tuple[int, int],
    sampling: tuple[float, float],
    *,
    cutoff: float = 2.0 / 3.0,
    taper: float = 1.0 / 12.0,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    ky, kx = reciprocal_mesh(gpts, sampling, device=device, dtype=dtype)
    r = torch.sqrt(ky.square() + kx.square())
    radial_cutoff = cutoff / max(sampling) / 2.0
    radial_taper = taper / max(sampling)

    if radial_taper > 0.0:
        aperture = 0.5 * (1.0 + torch.cos(math.pi * (r - radial_cutoff + radial_taper) / radial_taper))
        aperture = torch.where(r > radial_cutoff, torch.zeros_like(aperture), aperture)
        aperture = torch.where(r > radial_cutoff - radial_taper, aperture, torch.ones_like(aperture))
    else:
        aperture = (r < radial_cutoff).to(dtype)
    return aperture


class AntialiasFilter(nn.Module):
    def __init__(
        self,
        *,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        cutoff: float = 2.0 / 3.0,
        taper: float = 1.0 / 12.0,
    ) -> None:
        super().__init__()
        self.gpts = gpts
        self.sampling = sampling
        self.cutoff = float(cutoff)
        self.taper = float(taper)

    def kernel(self, *, device=None, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        return antialias_aperture(
            self.gpts,
            self.sampling,
            cutoff=self.cutoff,
            taper=self.taper,
            device=device,
            dtype=dtype,
        )

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel(device=wave.device, dtype=torch.float64).to(wave.dtype)
        return torch.fft.ifft2(torch.fft.fft2(wave, norm="ortho") * kernel, norm="ortho")
