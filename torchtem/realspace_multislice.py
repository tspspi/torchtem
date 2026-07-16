from __future__ import annotations

import math

import torch
from torch import nn

from torchtem.physics import energy2sigma, energy2wavelength


class RealSpaceLaplaceOperator(nn.Module):
    """Periodic centered-difference Laplacian."""

    def __init__(
        self,
        *,
        sampling: tuple[float, float],
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.sampling = sampling
        self.dtype = dtype

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        dy, dx = self.sampling
        return (
            (torch.roll(wave, 1, dims=-2) - 2.0 * wave + torch.roll(wave, -1, dims=-2))
            / (dy * dy)
            + (torch.roll(wave, 1, dims=-1) - 2.0 * wave + torch.roll(wave, -1, dims=-1))
            / (dx * dx)
        )


class RealSpaceMultisliceSystem(nn.Module):
    """Real-space multislice based on a truncated exponential-series propagator."""

    def __init__(
        self,
        *,
        energy_eV: float,
        sampling: tuple[float, float],
        slice_thickness_A: float,
        series_order: int = 1,
        max_terms: int = 12,
        tolerance: float = 1e-12,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.energy_eV = float(energy_eV)
        self.sampling = sampling
        self.slice_thickness_A = float(slice_thickness_A)
        self.series_order = int(series_order)
        self.max_terms = int(max_terms)
        self.tolerance = float(tolerance)
        self.dtype = dtype
        self.laplace = RealSpaceLaplaceOperator(sampling=sampling, dtype=dtype)

    def conventional_operator(
        self, waves: torch.Tensor, transmission_function: torch.Tensor, wavelength: torch.Tensor
    ) -> torch.Tensor:
        k0 = 1.0 / wavelength
        return self.laplace(waves) / (4.0 * math.pi * k0) + transmission_function * waves

    def propagator_taylor_series(
        self, waves: torch.Tensor, transmission_function: torch.Tensor, wavelength: torch.Tensor
    ) -> torch.Tensor:
        if self.series_order < 1:
            raise ValueError("series_order must be at least 1")
        thickness = self.slice_thickness_A
        if self.series_order == 1:
            return self.conventional_operator(waves, transmission_function, wavelength) * 1.0j * thickness

        k0 = 1.0 / wavelength
        laplace_waves = self.laplace(waves) / (4.0 * math.pi * k0)
        series = laplace_waves.clone()
        temp = laplace_waves.clone()
        for i in range(2, self.series_order + 1):
            prefactor = (wavelength / (-2.0 * math.pi)) ** (i - 1) * 0.5
            temp = self.laplace(temp) / (4.0 * math.pi * k0)
            series = series + temp * prefactor
        return (series + waves * transmission_function) * 1.0j * thickness

    def _exponential_series(self, waves: torch.Tensor, transmission_function: torch.Tensor) -> torch.Tensor:
        wavelength = energy2wavelength(self.energy_eV, device=waves.device, dtype=self.dtype)
        initial_norm = waves.abs().sum()
        result = waves.clone()
        temp = self.propagator_taylor_series(waves, transmission_function, wavelength)
        result = result + temp
        for i in range(2, self.max_terms + 1):
            temp = self.propagator_taylor_series(temp, transmission_function, wavelength) / i
            result = result + temp
            if (temp.abs().sum() / (initial_norm + 1e-30)).item() <= self.tolerance:
                break
        return result

    def transmission_function(self, projected_potential: torch.Tensor) -> torch.Tensor:
        sigma = energy2sigma(self.energy_eV, device=projected_potential.device, dtype=self.dtype)
        return projected_potential.to(torch.complex128) * sigma / self.slice_thickness_A

    def forward(self, incident_wave: torch.Tensor, projected_potential_slices: torch.Tensor) -> torch.Tensor:
        wave = incident_wave.to(torch.complex128)
        for slice_potential in projected_potential_slices:
            wave = self._exponential_series(wave, self.transmission_function(slice_potential))
        return wave

