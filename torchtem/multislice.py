from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from torchtem.physics import energy2sigma, energy2wavelength, reciprocal_mesh


class FresnelPropagator(nn.Module):
    """Exact Fourier-space Fresnel propagator following abTEM's multislice kernel."""

    def __init__(
        self,
        *,
        energy_eV: float,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        thickness_A: float,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        super().__init__()
        self.energy_eV = float(energy_eV)
        self.gpts = gpts
        self.sampling = sampling
        self.thickness_A = float(thickness_A)
        self.dtype = dtype
        self.device_hint = device

    def kernel(self) -> torch.Tensor:
        wavelength = energy2wavelength(self.energy_eV, device=self.device_hint, dtype=self.dtype)
        ky, kx = reciprocal_mesh(
            self.gpts, self.sampling, device=self.device_hint, dtype=self.dtype
        )
        k2 = ky.square() + kx.square()
        x = wavelength.square() * k2
        phase = torch.empty_like(x, dtype=torch.complex128)
        propagating = x <= 1.0
        evanescent = ~propagating
        phase[propagating] = (
            (2.0 * torch.pi * self.thickness_A / wavelength)
            * (-x[propagating] / (torch.sqrt(1.0 - x[propagating]) + 1.0))
        ).to(torch.complex128)
        phase[evanescent] = (
            (2.0 * torch.pi * self.thickness_A / wavelength)
            * (1.0j * torch.sqrt(x[evanescent] - 1.0) - 1.0)
        )
        return torch.exp(phase)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        kernel = self.kernel()
        return torch.fft.ifft2(torch.fft.fft2(wave, norm="ortho") * kernel, norm="ortho")


class MultisliceSystem(nn.Module):
    """Differentiable multislice forward model for projected electrostatic slices."""

    def __init__(
        self,
        *,
        energy_eV: float,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        slice_thickness_A: float,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        super().__init__()
        self.energy_eV = float(energy_eV)
        self.gpts = gpts
        self.sampling = sampling
        self.slice_thickness_A = float(slice_thickness_A)
        self.dtype = dtype
        self.device_hint = device
        self.propagator = FresnelPropagator(
            energy_eV=energy_eV,
            gpts=gpts,
            sampling=sampling,
            thickness_A=slice_thickness_A,
            dtype=dtype,
            device=device,
        )

    def transmission_function(self, projected_potential: torch.Tensor) -> torch.Tensor:
        sigma = energy2sigma(self.energy_eV, device=projected_potential.device, dtype=self.dtype)
        return torch.exp(1.0j * sigma * projected_potential.to(torch.complex128))

    def forward(
        self,
        incident_wave: torch.Tensor,
        projected_potential_slices: torch.Tensor | Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(projected_potential_slices, torch.Tensor):
            slices = projected_potential_slices
        else:
            slices = torch.stack(tuple(projected_potential_slices), dim=0)

        wave = incident_wave.to(torch.complex128)
        for slice_potential in slices:
            wave = self.transmission_function(slice_potential) * wave
            wave = self.propagator(wave)
        return wave

