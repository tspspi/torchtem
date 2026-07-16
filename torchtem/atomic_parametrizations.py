from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from tspi.torchtem.constants import kappa
from tspi.torchtem.physics import ensure_tuple2, real_space_mesh


def _parametrization_data_path(filename: str) -> Path:
    tspi_dir = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[2]
    path = (
        tspi_dir
        / "snapshot"
        / "src"
        / "abtem"
        / "parametrizations"
        / "data"
        / filename
    )
    if not path.exists():
        path = repo_root / "abtem" / "parametrizations" / "data" / filename
    return path


@lru_cache(maxsize=None)
def _load_parametrization_table(filename: str) -> dict[str, torch.Tensor]:
    path = _parametrization_data_path(filename)
    data = json.loads(path.read_text())
    return {symbol: torch.tensor(values, dtype=torch.float64) for symbol, values in data.items()}


@lru_cache(maxsize=1)
def _load_lobato_parameters() -> dict[str, torch.Tensor]:
    return _load_parametrization_table("lobato.json")


@lru_cache(maxsize=1)
def _load_kirkland_parameters() -> dict[str, torch.Tensor]:
    return _load_parametrization_table("kirkland.json")


@lru_cache(maxsize=1)
def _load_peng_parameters(filename: str = "peng_high.json") -> dict[str, torch.Tensor]:
    return _load_parametrization_table(filename)


def lobato_scaled_projected_parameters(symbol: str) -> torch.Tensor:
    parameters = _load_lobato_parameters()[symbol]
    a = torch.pi**2 * parameters[0] / torch.pow(parameters[1], 1.5) / kappa
    b = 2.0 * torch.pi / torch.sqrt(parameters[1])
    return torch.stack((a, b), dim=0)


def kirkland_scaled_projected_parameters(symbol: str) -> torch.Tensor:
    parameters = _load_kirkland_parameters()[symbol]
    a = torch.pi * parameters[0] / kappa
    b = 2.0 * torch.pi * torch.sqrt(parameters[1])
    c = (
        torch.pi ** 1.5
        * parameters[2]
        / torch.pow(parameters[3], 1.5)
        / kappa
    )
    d = torch.pi**2 / parameters[3]
    return torch.stack((a, b, c, d), dim=0)


def peng_scaled_parameters(symbol: str, table: str = "peng_high.json") -> dict[str, torch.Tensor]:
    scattering_factor = _load_peng_parameters(table)[symbol].clone()
    scattering_factor[1] = scattering_factor[1] / 4.0

    projected_potential = torch.stack(
        [
            torch.pi * scattering_factor[0] / (kappa * scattering_factor[1]),
            torch.pi**2 / scattering_factor[1],
        ],
        dim=0,
    )
    projected_scattering_factor = torch.stack(
        [scattering_factor[0] / kappa, scattering_factor[1]],
        dim=0,
    )
    finite_projected_potential = torch.cat(
        [projected_potential, torch.pi / torch.sqrt(projected_potential[1]).unsqueeze(0)], dim=0
    )
    return {
        "projected_potential": projected_potential,
        "projected_scattering_factor": projected_scattering_factor,
        "finite_projected_potential": finite_projected_potential,
        "finite_projected_scattering_factor": finite_projected_potential,
    }


def render_lobato_projected_potential(
    *,
    gpts: tuple[int, int],
    sampling: tuple[float, float] | float,
    positions_A: torch.Tensor,
    scaled_parameters: torch.Tensor,
    amplitudes: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float64,
    device=None,
    eps: float = 1e-8,
) -> torch.Tensor:
    sampling = ensure_tuple2(sampling)
    yy, xx = real_space_mesh(gpts, sampling, device=device or positions_A.device, dtype=dtype)
    dy = yy.unsqueeze(0) - positions_A[:, 0].unsqueeze(-1).unsqueeze(-1)
    dx = xx.unsqueeze(0) - positions_A[:, 1].unsqueeze(-1).unsqueeze(-1)
    r = torch.sqrt(dx.square() + dy.square() + eps)

    a = scaled_parameters[:, 0, :].unsqueeze(-1).unsqueeze(-1)
    b = scaled_parameters[:, 1, :].unsqueeze(-1).unsqueeze(-1)
    br = b * r.unsqueeze(1)
    contribution = 2.0 * (
        2.0 * a / b * torch.special.modified_bessel_k0(br)
        + a * r.unsqueeze(1) * torch.special.modified_bessel_k1(br)
    ).sum(dim=1)
    if amplitudes is not None:
        contribution = contribution * amplitudes.unsqueeze(-1).unsqueeze(-1)
    return contribution.sum(dim=0)


def render_kirkland_projected_potential(
    *,
    gpts: tuple[int, int],
    sampling: tuple[int, int] | tuple[float, float] | float,
    positions_A: torch.Tensor,
    scaled_parameters: torch.Tensor,
    amplitudes: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float64,
    device=None,
    eps: float = 1e-8,
) -> torch.Tensor:
    sampling = ensure_tuple2(sampling)
    yy, xx = real_space_mesh(gpts, sampling, device=device or positions_A.device, dtype=dtype)
    dy = yy.unsqueeze(0) - positions_A[:, 0].unsqueeze(-1).unsqueeze(-1)
    dx = xx.unsqueeze(0) - positions_A[:, 1].unsqueeze(-1).unsqueeze(-1)
    r = torch.sqrt(dx.square() + dy.square() + eps)

    a = scaled_parameters[:, 0, :].unsqueeze(-1).unsqueeze(-1)
    b = scaled_parameters[:, 1, :].unsqueeze(-1).unsqueeze(-1)
    c = scaled_parameters[:, 2, :].unsqueeze(-1).unsqueeze(-1)
    d = scaled_parameters[:, 3, :].unsqueeze(-1).unsqueeze(-1)
    contribution = (
        2.0 * a * torch.special.modified_bessel_k0(b * r.unsqueeze(1))
        + torch.sqrt(torch.pi / d) * c * torch.exp(-d * r.unsqueeze(1).square())
    ).sum(dim=1)
    if amplitudes is not None:
        contribution = contribution * amplitudes.unsqueeze(-1).unsqueeze(-1)
    return contribution.sum(dim=0)


def render_peng_finite_projected_potential(
    *,
    gpts: tuple[int, int],
    sampling: tuple[int, int] | tuple[float, float] | float,
    positions_A: torch.Tensor,
    scaled_parameters: torch.Tensor,
    slice_limits_A: tuple[float, float],
    z_positions_A: torch.Tensor | None = None,
    amplitudes: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> torch.Tensor:
    sampling = ensure_tuple2(sampling)
    yy, xx = real_space_mesh(gpts, sampling, device=device or positions_A.device, dtype=dtype)
    dy = yy.unsqueeze(0) - positions_A[:, 0].unsqueeze(-1).unsqueeze(-1)
    dx = xx.unsqueeze(0) - positions_A[:, 1].unsqueeze(-1).unsqueeze(-1)
    r2 = dx.square() + dy.square()

    a = scaled_parameters[:, 0, :].unsqueeze(-1).unsqueeze(-1)
    b = scaled_parameters[:, 1, :].unsqueeze(-1).unsqueeze(-1)
    c = scaled_parameters[:, 2, :].unsqueeze(-1).unsqueeze(-1)
    if z_positions_A is None:
        z_positions_A = torch.zeros((positions_A.shape[0],), device=positions_A.device, dtype=dtype)
    lower = (
        torch.as_tensor(slice_limits_A[0], device=positions_A.device, dtype=dtype)
        - z_positions_A
    ).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    upper = (
        torch.as_tensor(slice_limits_A[1], device=positions_A.device, dtype=dtype)
        - z_positions_A
    ).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    contribution = (
        torch.abs(torch.erf(c * upper) - torch.erf(c * lower))
        * a
        * torch.exp(-b * r2.unsqueeze(1))
    ).sum(dim=1) / 2.0
    if amplitudes is not None:
        contribution = contribution * amplitudes.unsqueeze(-1).unsqueeze(-1)
    return contribution.sum(dim=0)


def ewald_gaussian_potential(r: torch.Tensor, Z: torch.Tensor, width: torch.Tensor) -> torch.Tensor:
    return Z * torch.erf(r / (torch.sqrt(torch.tensor(2.0, dtype=r.dtype, device=r.device)) * width)) / r


def ewald_point_charge_potential(r: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    return Z / r


def render_ewald_potential_projection(
    *,
    gpts: tuple[int, int],
    sampling: tuple[int, int] | tuple[float, float] | float,
    positions_A: torch.Tensor,
    atomic_numbers: torch.Tensor,
    widths_A: torch.Tensor,
    amplitudes: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float64,
    device=None,
    eps: float = 1e-8,
) -> torch.Tensor:
    sampling = ensure_tuple2(sampling)
    yy, xx = real_space_mesh(gpts, sampling, device=device or positions_A.device, dtype=dtype)
    dy = yy.unsqueeze(0) - positions_A[:, 0].unsqueeze(-1).unsqueeze(-1)
    dx = xx.unsqueeze(0) - positions_A[:, 1].unsqueeze(-1).unsqueeze(-1)
    r = torch.sqrt(dx.square() + dy.square() + eps)
    Z = atomic_numbers.unsqueeze(-1).unsqueeze(-1).to(dtype)
    width = widths_A.unsqueeze(-1).unsqueeze(-1)
    contribution = ewald_point_charge_potential(r, Z) - ewald_gaussian_potential(r, Z, width)
    if amplitudes is not None:
        contribution = contribution * amplitudes.unsqueeze(-1).unsqueeze(-1)
    return contribution.sum(dim=0)


class LobatoProjectedPotential(nn.Module):
    """Differentiable projected potential using abTEM's Lobato parametrization."""

    def __init__(
        self,
        *,
        gpts: tuple[int, int],
        sampling: tuple[float, float] | float,
        symbols: Sequence[str],
        positions_A: torch.Tensor,
        amplitudes: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        super().__init__()
        self.gpts = gpts
        self.sampling = ensure_tuple2(sampling)
        self.dtype = dtype
        self.device_hint = device
        self.symbols = tuple(symbols)
        if len(self.symbols) != int(positions_A.shape[0]):
            raise ValueError("Length of symbols must match number of atom positions")
        self.positions_A = nn.Parameter(positions_A.to(device=device, dtype=dtype))
        if amplitudes is None:
            amplitudes = torch.ones((positions_A.shape[0],), dtype=dtype)
        self.amplitudes = nn.Parameter(amplitudes.to(device=device, dtype=dtype))
        scaled = torch.stack(
            [lobato_scaled_projected_parameters(symbol) for symbol in self.symbols], dim=0
        )
        self.register_buffer("scaled_parameters", scaled.to(device=device, dtype=dtype))

    def forward(self) -> torch.Tensor:
        return render_lobato_projected_potential(
            gpts=self.gpts,
            sampling=self.sampling,
            positions_A=self.positions_A,
            scaled_parameters=self.scaled_parameters,
            amplitudes=self.amplitudes,
            dtype=self.dtype,
            device=self.positions_A.device,
        )


class KirklandProjectedPotential(nn.Module):
    """Differentiable projected potential using abTEM's Kirkland parametrization."""

    def __init__(
        self,
        *,
        gpts: tuple[int, int],
        sampling: tuple[float, float] | float,
        symbols: Sequence[str],
        positions_A: torch.Tensor,
        amplitudes: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        super().__init__()
        self.gpts = gpts
        self.sampling = ensure_tuple2(sampling)
        self.dtype = dtype
        self.device_hint = device
        self.symbols = tuple(symbols)
        if len(self.symbols) != int(positions_A.shape[0]):
            raise ValueError("Length of symbols must match number of atom positions")
        self.positions_A = nn.Parameter(positions_A.to(device=device, dtype=dtype))
        if amplitudes is None:
            amplitudes = torch.ones((positions_A.shape[0],), dtype=dtype)
        self.amplitudes = nn.Parameter(amplitudes.to(device=device, dtype=dtype))
        scaled = torch.stack(
            [kirkland_scaled_projected_parameters(symbol) for symbol in self.symbols], dim=0
        )
        self.register_buffer("scaled_parameters", scaled.to(device=device, dtype=dtype))

    def forward(self) -> torch.Tensor:
        return render_kirkland_projected_potential(
            gpts=self.gpts,
            sampling=self.sampling,
            positions_A=self.positions_A,
            scaled_parameters=self.scaled_parameters,
            amplitudes=self.amplitudes,
            dtype=self.dtype,
            device=self.positions_A.device,
        )


class PengFiniteProjectedPotential(nn.Module):
    """Differentiable finite projected potential using abTEM's Peng parametrization."""

    def __init__(
        self,
        *,
        gpts: tuple[int, int],
        sampling: tuple[float, float] | float,
        symbols: Sequence[str],
        positions_A: torch.Tensor,
        slice_limits_A: tuple[float, float],
        table: str = "peng_high.json",
        amplitudes: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        super().__init__()
        self.gpts = gpts
        self.sampling = ensure_tuple2(sampling)
        self.slice_limits_A = slice_limits_A
        self.table = table
        self.dtype = dtype
        self.device_hint = device
        self.symbols = tuple(symbols)
        if len(self.symbols) != int(positions_A.shape[0]):
            raise ValueError("Length of symbols must match number of atom positions")
        self.positions_A = nn.Parameter(positions_A.to(device=device, dtype=dtype))
        if amplitudes is None:
            amplitudes = torch.ones((positions_A.shape[0],), dtype=dtype)
        self.amplitudes = nn.Parameter(amplitudes.to(device=device, dtype=dtype))
        scaled = torch.stack(
            [peng_scaled_parameters(symbol, table)["finite_projected_potential"] for symbol in self.symbols],
            dim=0,
        )
        self.register_buffer("scaled_parameters", scaled.to(device=device, dtype=dtype))

    def forward(self) -> torch.Tensor:
        return render_peng_finite_projected_potential(
            gpts=self.gpts,
            sampling=self.sampling,
            positions_A=self.positions_A,
            scaled_parameters=self.scaled_parameters,
            slice_limits_A=self.slice_limits_A,
            amplitudes=self.amplitudes,
            dtype=self.dtype,
            device=self.positions_A.device,
        )


class EwaldPotentialProjection(nn.Module):
    """Differentiable screened point-charge projection using the Ewald functional form."""

    def __init__(
        self,
        *,
        gpts: tuple[int, int],
        sampling: tuple[float, float] | float,
        atomic_numbers: torch.Tensor,
        positions_A: torch.Tensor,
        widths_A: torch.Tensor | float = 1.0,
        amplitudes: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        super().__init__()
        self.gpts = gpts
        self.sampling = ensure_tuple2(sampling)
        self.dtype = dtype
        self.device_hint = device
        self.positions_A = nn.Parameter(positions_A.to(device=device, dtype=dtype))
        self.register_buffer("atomic_numbers", atomic_numbers.to(device=device, dtype=dtype))
        widths_A = torch.as_tensor(widths_A, dtype=dtype)
        if widths_A.ndim == 0:
            widths_A = widths_A.repeat(positions_A.shape[0])
        self.log_widths_A = nn.Parameter(widths_A.to(device=device, dtype=dtype).log())
        if amplitudes is None:
            amplitudes = torch.ones((positions_A.shape[0],), dtype=dtype)
        self.amplitudes = nn.Parameter(amplitudes.to(device=device, dtype=dtype))

    @property
    def widths_A(self) -> torch.Tensor:
        return self.log_widths_A.exp()

    def forward(self) -> torch.Tensor:
        return render_ewald_potential_projection(
            gpts=self.gpts,
            sampling=self.sampling,
            positions_A=self.positions_A,
            atomic_numbers=self.atomic_numbers,
            widths_A=self.widths_A,
            amplitudes=self.amplitudes,
            dtype=self.dtype,
            device=self.positions_A.device,
        )
