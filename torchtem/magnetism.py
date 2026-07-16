from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn


def _lyon_data_path() -> Path:
    candidates = (
        Path(__file__).resolve().parents[1] / "snapshot" / "src" / "abtem" / "magnetism" / "parametrizations" / "data" / "lyon.json",
        Path(__file__).resolve().parents[2] / "abtem" / "magnetism" / "parametrizations" / "data" / "lyon.json",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not locate lyon.json in snapshot or source tree.")


@lru_cache(maxsize=1)
def _load_lyon_parameters() -> dict[str, torch.Tensor]:
    data = json.loads(_lyon_data_path().read_text())
    return {symbol: torch.tensor(values, dtype=torch.float64) for symbol, values in data.items()}


def lyon_magnetic_parameters(symbol: str) -> torch.Tensor:
    return _load_lyon_parameters()[symbol].clone()


def cutoff_taper(radial_gpts: torch.Tensor, cutoff: float, taper_fraction: float = 0.85) -> torch.Tensor:
    taper_start = taper_fraction * cutoff
    taper_values = torch.ones_like(radial_gpts)
    taper_mask = radial_gpts > taper_start
    taper_values = torch.where(
        taper_mask,
        (torch.cos(torch.pi * (radial_gpts - taper_start) / (cutoff - taper_start)) + 1.0) / 2.0,
        taper_values,
    )
    return torch.where(radial_gpts <= cutoff, taper_values, torch.zeros_like(taper_values))


def _radial_components(
    r: torch.Tensor,
    parameters: torch.Tensor,
    cutoff_A: float,
    taper_fraction: float = 0.85,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    parameters = parameters.to(device=r.device, dtype=r.dtype)
    r_safe = torch.clamp(r, min=1e-8)
    a = parameters[:, 0]
    b = parameters[:, 1]
    n_i = torch.arange(5, device=r.device, dtype=r.dtype) / 2.0 + 3.0

    r_pow = r_safe[..., None].pow(n_i)
    denom = r_pow + b
    taper = cutoff_taper(r_safe, cutoff_A, taper_fraction=taper_fraction)

    prefactor_a = (a / denom).sum(dim=-1) * taper
    prefactor_b1 = (a * n_i * r_safe[..., None].pow(n_i - 2.0) / denom.square()).sum(dim=-1) * taper
    prefactor_b2 = (a * (2.0 * b - (n_i - 2.0) * r_pow) / denom.square()).sum(dim=-1) * taper
    return prefactor_a, prefactor_b1, prefactor_b2


def _real_space_mesh(
    gpts: tuple[int, int],
    sampling: tuple[float, float],
    device=None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.arange(gpts[0], device=device, dtype=dtype) * sampling[0]
    x = torch.arange(gpts[1], device=device, dtype=dtype) * sampling[1]
    return torch.meshgrid(y, x, indexing="ij")


def projected_quasi_dipole_vector_potential(
    *,
    gpts: tuple[int, int],
    sampling: tuple[float, float],
    positions_A: torch.Tensor,
    magnetic_moments: torch.Tensor,
    parameters: list[torch.Tensor],
    slice_limits_A: tuple[float, float],
    quadrature_points: int = 33,
    cutoff_A: float = 4.25,
    taper_fraction: float = 0.85,
) -> torch.Tensor:
    yy, xx = _real_space_mesh(gpts, sampling, device=positions_A.device, dtype=positions_A.dtype)
    z = torch.linspace(
        slice_limits_A[0],
        slice_limits_A[1],
        quadrature_points,
        device=positions_A.device,
        dtype=positions_A.dtype,
    )
    projected = torch.zeros((3,) + gpts, device=positions_A.device, dtype=torch.float64)

    for position, moment, atom_parameters in zip(positions_A, magnetic_moments, parameters):
        dy = yy - position[0]
        dx = xx - position[1]
        dz = z[:, None, None] - position[2]
        r = torch.sqrt(dy.square()[None] + dx.square()[None] + dz.square() + 1e-8)
        prefactor_a, _, _ = _radial_components(
            r, atom_parameters, cutoff_A=cutoff_A, taper_fraction=taper_fraction
        )

        r_vec = torch.stack(
            (
                dy.unsqueeze(0).expand_as(r),
                dx.unsqueeze(0).expand_as(r),
                dz.expand_as(r),
            ),
            dim=-1,
        )
        cross = torch.linalg.cross(moment.to(r_vec.dtype).view(1, 1, 1, 3).expand_as(r_vec), r_vec, dim=-1)
        integrand = prefactor_a[..., None] * cross
        projected = projected + torch.trapezoid(integrand, z, dim=0).movedim(-1, 0)

    return projected


def projected_quasi_dipole_magnetic_field(
    *,
    gpts: tuple[int, int],
    sampling: tuple[float, float],
    positions_A: torch.Tensor,
    magnetic_moments: torch.Tensor,
    parameters: list[torch.Tensor],
    slice_limits_A: tuple[float, float],
    quadrature_points: int = 33,
    cutoff_A: float = 4.25,
    taper_fraction: float = 0.85,
) -> torch.Tensor:
    yy, xx = _real_space_mesh(gpts, sampling, device=positions_A.device, dtype=positions_A.dtype)
    z = torch.linspace(
        slice_limits_A[0],
        slice_limits_A[1],
        quadrature_points,
        device=positions_A.device,
        dtype=positions_A.dtype,
    )
    projected = torch.zeros((3,) + gpts, device=positions_A.device, dtype=torch.float64)

    for position, moment, atom_parameters in zip(positions_A, magnetic_moments, parameters):
        dy = yy - position[0]
        dx = xx - position[1]
        dz = z[:, None, None] - position[2]
        r = torch.sqrt(dy.square()[None] + dx.square()[None] + dz.square() + 1e-8)
        _, prefactor_b1, prefactor_b2 = _radial_components(
            r, atom_parameters, cutoff_A=cutoff_A, taper_fraction=taper_fraction
        )

        r_vec = torch.stack(
            (
                dy.unsqueeze(0).expand_as(r),
                dx.unsqueeze(0).expand_as(r),
                dz.expand_as(r),
            ),
            dim=-1,
        )
        moment_vec = moment.to(r_vec.dtype).view(1, 1, 1, 3).expand_as(r_vec)
        mr = (r_vec * moment_vec).sum(dim=-1, keepdim=True)
        integrand = prefactor_b1[..., None] * r_vec * mr + prefactor_b2[..., None] * moment_vec
        projected = projected + torch.trapezoid(integrand, z, dim=0).movedim(-1, 0)

    return projected


class LyonProjectedVectorPotential(nn.Module):
    """Differentiable projected vector potential from Lyon quasi-dipole parameters."""

    def __init__(
        self,
        *,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        symbols: list[str],
        positions_A: torch.Tensor,
        magnetic_moments: torch.Tensor,
        slice_limits_A: tuple[float, float],
        quadrature_points: int = 33,
        cutoff_A: float = 4.25,
    ) -> None:
        super().__init__()
        self.gpts = gpts
        self.sampling = sampling
        self.symbols = list(symbols)
        self.slice_limits_A = slice_limits_A
        self.quadrature_points = int(quadrature_points)
        self.cutoff_A = float(cutoff_A)
        self.positions_A = nn.Parameter(positions_A.to(torch.float64))
        self.magnetic_moments = nn.Parameter(magnetic_moments.to(torch.float64))

    def forward(self) -> torch.Tensor:
        parameters = [lyon_magnetic_parameters(symbol) for symbol in self.symbols]
        return projected_quasi_dipole_vector_potential(
            gpts=self.gpts,
            sampling=self.sampling,
            positions_A=self.positions_A,
            magnetic_moments=self.magnetic_moments,
            parameters=parameters,
            slice_limits_A=self.slice_limits_A,
            quadrature_points=self.quadrature_points,
            cutoff_A=self.cutoff_A,
        )


class LyonProjectedMagneticField(nn.Module):
    """Differentiable projected magnetic field from Lyon quasi-dipole parameters."""

    def __init__(
        self,
        *,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        symbols: list[str],
        positions_A: torch.Tensor,
        magnetic_moments: torch.Tensor,
        slice_limits_A: tuple[float, float],
        quadrature_points: int = 33,
        cutoff_A: float = 4.25,
    ) -> None:
        super().__init__()
        self.gpts = gpts
        self.sampling = sampling
        self.symbols = list(symbols)
        self.slice_limits_A = slice_limits_A
        self.quadrature_points = int(quadrature_points)
        self.cutoff_A = float(cutoff_A)
        self.positions_A = nn.Parameter(positions_A.to(torch.float64))
        self.magnetic_moments = nn.Parameter(magnetic_moments.to(torch.float64))

    def forward(self) -> torch.Tensor:
        parameters = [lyon_magnetic_parameters(symbol) for symbol in self.symbols]
        return projected_quasi_dipole_magnetic_field(
            gpts=self.gpts,
            sampling=self.sampling,
            positions_A=self.positions_A,
            magnetic_moments=self.magnetic_moments,
            parameters=parameters,
            slice_limits_A=self.slice_limits_A,
            quadrature_points=self.quadrature_points,
            cutoff_A=self.cutoff_A,
        )


def central_difference_gradient_pbc(
    wave_functions: torch.Tensor,
    *,
    sampling: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_y = (torch.roll(wave_functions, -1, dims=-2) - torch.roll(wave_functions, 1, dims=-2)) / (
        2.0 * sampling[0]
    )
    grad_x = (torch.roll(wave_functions, -1, dims=-1) - torch.roll(wave_functions, 1, dims=-1)) / (
        2.0 * sampling[1]
    )
    return grad_x, grad_y


def apply_A_xy_dot_nabla_xy(
    A: torch.Tensor,
    wave_functions: torch.Tensor,
    *,
    sampling: tuple[float, float],
) -> torch.Tensor:
    grad_x, grad_y = central_difference_gradient_pbc(wave_functions, sampling=sampling)
    return A[0] * grad_x + A[1] * grad_y


class PauliVectorPotentialInteraction(nn.Module):
    """Apply the projected Pauli interaction term A_xy · ∇_xy to wavefunctions."""

    def __init__(self, *, sampling: tuple[float, float], prefactor: complex = 1.0j) -> None:
        super().__init__()
        self.sampling = sampling
        self.prefactor = complex(prefactor)

    def forward(self, vector_potential: torch.Tensor, wave_functions: torch.Tensor) -> torch.Tensor:
        interaction = apply_A_xy_dot_nabla_xy(
            vector_potential.to(wave_functions.dtype),
            wave_functions,
            sampling=self.sampling,
        )
        return wave_functions + self.prefactor * interaction
