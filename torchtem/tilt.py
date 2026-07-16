from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from torchtem.distributions import DistributionFromValues, from_values, validate_distribution
from torchtem.physics import energy2wavelength, real_space_mesh


def precession_tilts(
    precession_angle_mrad: float,
    num_samples: int,
    min_azimuth_rad: float = 0.0,
    max_azimuth_rad: float = 2.0 * math.pi,
    endpoint: bool = False,
) -> torch.Tensor:
    azimuth = torch.linspace(
        min_azimuth_rad,
        max_azimuth_rad,
        num_samples,
        dtype=torch.float64,
    )
    if not endpoint and num_samples > 1:
        step = (max_azimuth_rad - min_azimuth_rad) / num_samples
        azimuth = torch.linspace(
            min_azimuth_rad,
            max_azimuth_rad - step,
            num_samples,
            dtype=torch.float64,
        )
    tilt_x = precession_angle_mrad * torch.cos(azimuth)
    tilt_y = precession_angle_mrad * torch.sin(azimuth)
    return torch.stack((tilt_y, tilt_x), dim=-1)


def apply_beam_tilt(
    wave: torch.Tensor,
    *,
    tilt_mrad: torch.Tensor,
    energy_eV: float,
    sampling: tuple[float, float],
) -> torch.Tensor:
    if tilt_mrad.ndim == 1:
        tilt_mrad = tilt_mrad.unsqueeze(0)
    yy, xx = real_space_mesh(
        (wave.shape[-2], wave.shape[-1]),
        sampling,
        device=wave.device,
        dtype=torch.float64,
    )
    wavelength = energy2wavelength(energy_eV, device=wave.device, dtype=torch.float64)
    ky = tilt_mrad[:, 0, None, None].to(torch.float64) * 1e-3 / wavelength
    kx = tilt_mrad[:, 1, None, None].to(torch.float64) * 1e-3 / wavelength
    phase = 2.0j * math.pi * (ky * yy[None] + kx * xx[None])
    tilted = wave.unsqueeze(0) * torch.exp(phase)
    return tilted if tilted.shape[0] > 1 else tilted[0]


@dataclass
class BeamTilt:
    tilt: tuple[float, float] | DistributionFromValues | torch.Tensor

    def values(self) -> torch.Tensor:
        tilt = self.tilt
        if isinstance(tilt, DistributionFromValues):
            return tilt.values
        tilt = torch.as_tensor(tilt, dtype=torch.float64)
        if tilt.ndim == 1:
            return tilt.unsqueeze(0)
        return tilt


class AxisAlignedBeamTilt:
    def __init__(self, tilt: float | DistributionFromValues | torch.Tensor = 0.0, direction: str = "x"):
        self.tilt = validate_distribution(tilt)
        self.direction = direction

    def values(self) -> torch.Tensor:
        if isinstance(self.tilt, DistributionFromValues):
            return self.tilt.values
        return torch.as_tensor([float(self.tilt)], dtype=torch.float64)


class BeamTilt2D:
    def __init__(self, tilt_x, tilt_y):
        self.tilt_x = validate_distribution(tilt_x)
        self.tilt_y = validate_distribution(tilt_y)

    def mesh(self) -> torch.Tensor:
        if isinstance(self.tilt_x, DistributionFromValues):
            tilt_x = self.tilt_x.values
        else:
            tilt_x = torch.as_tensor([float(self.tilt_x)], dtype=torch.float64)
        if isinstance(self.tilt_y, DistributionFromValues):
            tilt_y = self.tilt_y.values
        else:
            tilt_y = torch.as_tensor([float(self.tilt_y)], dtype=torch.float64)
        yy, xx = torch.meshgrid(tilt_y, tilt_x, indexing="ij")
        return torch.stack((yy, xx), dim=-1)


class BeamTiltLayer(nn.Module):
    def __init__(self, *, energy_eV: float, sampling: tuple[float, float]) -> None:
        super().__init__()
        self.energy_eV = float(energy_eV)
        self.sampling = sampling

    def forward(self, wave: torch.Tensor, tilt_mrad: torch.Tensor) -> torch.Tensor:
        return apply_beam_tilt(
            wave,
            tilt_mrad=tilt_mrad,
            energy_eV=self.energy_eV,
            sampling=self.sampling,
        )
