from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.special import sph_harm


def _parametrization_data_path(filename: str) -> Path:
    tspi_dir = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[2]
    path = tspi_dir / "snapshot" / "src" / "abtem" / "parametrizations" / "data" / filename
    if not path.exists():
        path = repo_root / "abtem" / "parametrizations" / "data" / filename
    return path


def spherical_coordinates(
    box: Sequence[float],
    gpts: Sequence[int],
    origin: Sequence[float],
    *,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xyz = []
    for L, n, o in zip(box, gpts, origin):
        step = float(L) / int(n)
        xyz.append(torch.arange(int(n), device=device, dtype=dtype) * step - float(o))
    x, y, z = torch.meshgrid(*xyz, indexing="ij")
    r = torch.sqrt(x.square() + y.square() + z.square())
    theta = torch.zeros_like(r)
    mask = r != 0.0
    theta[mask] = torch.arccos(z[mask] / r[mask])
    phi = torch.atan2(y, x) + torch.pi
    return r, theta, phi


def spline_orders(splines: Sequence[object]) -> dict[int, tuple[object, int, int]]:
    new_splines: dict[int, tuple[object, int, int]] = {}
    index = 0
    for spline in splines:
        for k in range(-int(spline.l), int(spline.l) + 1):
            new_splines[index] = (spline, int(spline.l), k)
            index += 1
    return new_splines


def real_sph_harm(
    degree: int, order: int, theta: torch.Tensor, phi: torch.Tensor
) -> torch.Tensor:
    theta_np = theta.detach().cpu().numpy()
    phi_np = phi.detach().cpu().numpy()
    if order < 0:
        values = np.sqrt(2.0) * (-1) ** order * sph_harm(abs(order), degree, phi_np, theta_np).imag
    elif order == 0:
        values = sph_harm(order, degree, phi_np, theta_np).real
    else:
        values = np.sqrt(2.0) * (-1) ** order * sph_harm(order, degree, phi_np, theta_np).imag
    return torch.as_tensor(values, device=theta.device, dtype=theta.dtype)


def unpack_density_matrix(packed_density_matrix: torch.Tensor) -> torch.Tensor:
    packed = torch.as_tensor(packed_density_matrix)
    n = int((np.sqrt(8 * packed.numel() + 1) - 1) / 2)
    out = torch.zeros((n, n), dtype=packed.dtype, device=packed.device)
    index = 0
    for i in range(n):
        for j in range(i + 1):
            out[i, j] = packed[index]
            out[j, i] = packed[index]
            index += 1
    return out


def sum_spherical_basis_functions(
    splines: Sequence[object],
    density_matrix: torch.Tensor,
    r: torch.Tensor,
    theta: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    density = torch.zeros_like(r)
    ordered = spline_orders(splines)
    for i in range(density_matrix.shape[0]):
        for j in range(density_matrix.shape[1]):
            spline_i, degree_i, order_i = ordered[i]
            spline_j, degree_j, order_j = ordered[j]
            density_element = density_matrix[i, j]
            if torch.abs(density_element) < 1e-8:
                continue
            density = density + (
                density_element
                * spline_i.map(r)
                * spline_j.map(r)
                * r ** (degree_i + degree_j)
                * real_sph_harm(degree_i, order_i, theta, phi)
                * real_sph_harm(degree_j, order_j, theta, phi)
            )
    return density


def sum_radial_basis_functions(
    splines: Sequence[object], scales: float | Sequence[float], r: torch.Tensor
) -> torch.Tensor:
    density = torch.zeros_like(r)
    if isinstance(scales, (float, int)):
        scales = (float(scales),) * len(splines)
    y00 = float(real_sph_harm(0, 0, torch.zeros((), dtype=r.dtype), torch.zeros((), dtype=r.dtype)).item())
    for spline, scale in zip(splines, scales):
        density = density + float(scale) * spline.map(r) * y00
    return density


@lru_cache(maxsize=1)
def load_waasmaier_kirfel_parameters() -> dict[str, torch.Tensor]:
    data = json.loads(_parametrization_data_path("waasmaier_kirfel.json").read_text())
    return {symbol: torch.tensor(values, dtype=torch.float64) for symbol, values in data.items()}
