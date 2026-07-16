from __future__ import annotations

import math
from typing import Iterable

import torch

PI = math.pi
PLANCK = 6.62607015e-34
LIGHT_SPEED = 299792458.0
ELEMENTARY_CHARGE = 1.602176634e-19
ELECTRON_MASS = 9.1093837015e-31


def _as_float_tensor(value: float | torch.Tensor, *, device=None, dtype=None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value
        if device is not None or dtype is not None:
            tensor = tensor.to(device=device or tensor.device, dtype=dtype or tensor.dtype)
        return tensor
    return torch.as_tensor(value, device=device, dtype=dtype or torch.float64)


def energy2wavelength(
    energy_eV: float | torch.Tensor,
    *,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Relativistic electron wavelength in angstrom."""
    energy = _as_float_tensor(energy_eV, device=device, dtype=dtype)
    numerator = PLANCK * LIGHT_SPEED
    denominator = torch.sqrt(
        energy * (2.0 * ELECTRON_MASS * LIGHT_SPEED**2 / ELEMENTARY_CHARGE + energy)
    )
    return numerator / denominator / ELEMENTARY_CHARGE * 1.0e10


def energy2mass(
    energy_eV: float | torch.Tensor,
    *,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    energy = _as_float_tensor(energy_eV, device=device, dtype=dtype)
    return (1.0 + ELEMENTARY_CHARGE * energy / (ELECTRON_MASS * LIGHT_SPEED**2)) * ELECTRON_MASS


def energy2sigma(
    energy_eV: float | torch.Tensor,
    *,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Interaction parameter in 1 / (angstrom * eV)."""
    energy = _as_float_tensor(energy_eV, device=device, dtype=dtype)
    mass = energy2mass(energy, device=device, dtype=dtype)
    wavelength = energy2wavelength(energy, device=device, dtype=dtype)
    return (
        2.0
        * PI
        * mass
        * ELEMENTARY_CHARGE
        * wavelength
        / PLANCK**2
        * 1.0e-20
    )


def spatial_frequencies(
    gpts: tuple[int, int],
    sampling: tuple[float, float],
    *,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    ky = torch.fft.fftfreq(gpts[0], d=sampling[0], device=device).to(dtype=dtype)
    kx = torch.fft.fftfreq(gpts[1], d=sampling[1], device=device).to(dtype=dtype)
    return ky, kx


def reciprocal_mesh(
    gpts: tuple[int, int],
    sampling: tuple[float, float],
    *,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    ky, kx = spatial_frequencies(gpts, sampling, device=device, dtype=dtype)
    return torch.meshgrid(ky, kx, indexing="ij")


def polar_spatial_frequencies(
    gpts: tuple[int, int],
    sampling: tuple[float, float],
    *,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    ky, kx = reciprocal_mesh(gpts, sampling, device=device, dtype=dtype)
    k = torch.sqrt(ky.square() + kx.square())
    phi = torch.atan2(kx, ky)
    return k, phi


def real_space_mesh(
    gpts: tuple[int, int],
    sampling: tuple[float, float],
    *,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.arange(gpts[0], device=device, dtype=dtype) * sampling[0]
    x = torch.arange(gpts[1], device=device, dtype=dtype) * sampling[1]
    return torch.meshgrid(y, x, indexing="ij")


def stack_complex(real: torch.Tensor, imag: torch.Tensor | None = None) -> torch.Tensor:
    if imag is None:
        imag = torch.zeros_like(real)
    return torch.complex(real, imag)


def ensure_tuple2(value: float | Iterable[float]) -> tuple[float, float]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"Expected length-2 value, got {value}")
        return float(value[0]), float(value[1])
    value = float(value)
    return value, value

