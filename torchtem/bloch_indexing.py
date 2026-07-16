from __future__ import annotations

import math

import torch

from tspi.torchtem.bloch import excitation_errors, reciprocal_cell
from tspi.torchtem.physics import polar_spatial_frequencies


def pixel_edges(
    shape: tuple[int, int], sampling: tuple[float, float]
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.fft.fftshift(torch.fft.fftfreq(shape[0], d=1 / shape[0]))
    y = torch.fft.fftshift(torch.fft.fftfreq(shape[1], d=1 / shape[1]))
    x = (x - 0.5) * sampling[0]
    y = (y - 0.5) * sampling[1]
    return x, y


def find_projected_pixel_index(
    g: torch.Tensor,
    shape: tuple[int, int],
    sampling: tuple[float, float],
) -> torch.Tensor:
    x, y = pixel_edges(shape, sampling)
    n = torch.bucketize(g[..., 0].contiguous(), x) - 1
    m = torch.bucketize(g[..., 1].contiguous(), y) - 1
    return torch.stack((n, m), dim=-1)


def estimate_necessary_excitation_error(energy_eV: float, k_max: float) -> float:
    hkl_corner = torch.tensor([[math.sqrt(k_max), math.sqrt(k_max), 0.0]], dtype=torch.float64)
    return float(torch.abs(excitation_errors(hkl_corner, energy_eV)).item())


def validate_cell(cell: torch.Tensor | float | tuple[float, float, float]) -> torch.Tensor:
    if isinstance(cell, float):
        return torch.diag(torch.tensor([cell, cell, cell], dtype=torch.float64))
    if isinstance(cell, tuple):
        return torch.diag(torch.tensor(cell, dtype=torch.float64))
    cell = torch.as_tensor(cell, dtype=torch.float64)
    if cell.shape != (3, 3):
        return torch.diag(cell)
    return cell


def overlapping_spots_mask(nm: torch.Tensor, sg: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(nm.shape[0], dtype=torch.bool, device=nm.device)
    order = torch.argsort(torch.abs(sg))
    seen: set[tuple[int, int]] = set()
    for idx in order.tolist():
        pair = nm[idx]
        key = (int(pair[0].item()), int(pair[1].item()))
        if key not in seen:
            seen.add(key)
            mask[idx] = True
    return mask


def create_ellipse(a: int, b: int) -> torch.Tensor:
    y = torch.arange(-a, a + 1, dtype=torch.float64)[:, None]
    x = torch.arange(-b, b + 1, dtype=torch.float64)[None, :]
    a = max(a, 1)
    b = max(b, 1)
    return x.square() / (b**2) + y.square() / (a**2) <= 1.0


def antialiased_disk(r: float, sampling: tuple[float, float]) -> torch.Tensor:
    gpts = (
        2 * int(math.ceil(r / sampling[0])) + 1,
        2 * int(math.ceil(r / sampling[1])) + 1,
    )
    alpha, phi = polar_spatial_frequencies(
        gpts,
        (1 / (sampling[0] * gpts[0]), 1 / (sampling[1] * gpts[1])),
        dtype=torch.float64,
    )
    denominator = torch.sqrt(
        (torch.cos(phi) * sampling[0]).square() + (torch.sin(phi) * sampling[1]).square()
    )
    denominator[0, 0] = 1.0
    array = torch.clamp((r - alpha) / denominator + 0.5, min=0.0, max=1.0)
    array[0, 0] = 1.0
    return torch.fft.fftshift(array)


def integrate_ellipse_around_pixels(
    array: torch.Tensor,
    nm: torch.Tensor,
    r: float,
    sampling: tuple[float, float],
    priority: torch.Tensor | None = None,
) -> torch.Tensor:
    weights = antialiased_disk(r, sampling)
    a, b = weights.shape[0] // 2, weights.shape[1] // 2
    intensities = torch.zeros(nm.shape[-2], dtype=array.dtype, device=array.device)
    masked_array = array.clone()
    order = torch.arange(nm.shape[-2], device=array.device)
    if priority is not None:
        order = torch.argsort(priority)

    for idx in order.tolist():
        nmx = int(nm[idx, 0].item())
        nmy = int(nm[idx, 1].item())
        x_slice = slice(max(0, nmx - a), min(array.shape[-2], nmx + a + 1))
        y_slice = slice(max(0, nmy - b), min(array.shape[-1], nmy + b + 1))
        wx = slice(a - (nmx - x_slice.start), a + (x_slice.stop - nmx))
        wy = slice(b - (nmy - y_slice.start), b + (y_slice.stop - nmy))
        cropped_weights = weights[wx, wy]
        intensities[idx] = (masked_array[x_slice, y_slice] * cropped_weights).sum()
        masked_array[x_slice, y_slice] *= 1 - cropped_weights
    return intensities


def index_diffraction_spots(
    array: torch.Tensor,
    hkl: torch.Tensor,
    sampling: tuple[float, float],
    cell: torch.Tensor,
    energy_eV: float,
    orientation_matrix: torch.Tensor | None = None,
    radius: float | None = None,
) -> torch.Tensor:
    if orientation_matrix is None:
        orientation_matrix = torch.eye(3, dtype=torch.float64, device=array.device)
    reciprocal_lattice_vectors = reciprocal_cell(validate_cell(cell)).to(array.device) @ orientation_matrix.T
    g_vec = hkl.to(torch.float64).to(array.device) @ reciprocal_lattice_vectors
    nm = find_projected_pixel_index(g_vec, (array.shape[-2], array.shape[-1]), sampling)
    sg_abs = torch.abs(excitation_errors(g_vec, energy_eV))

    if radius is not None:
        intensities = integrate_ellipse_around_pixels(array, nm, radius, sampling, sg_abs)
    else:
        intensities = array[nm[:, 0], nm[:, 1]]

    mask = overlapping_spots_mask(nm, excitation_errors(g_vec, energy_eV))
    return intensities * mask.to(intensities.dtype)


def miller_to_miller_bravais(hkl: tuple[int, int, int]) -> tuple[int, int, int, int]:
    h, k, l = hkl
    H = 2 * h - k
    K = 2 * k - h
    I = -H - K
    L = l
    return H, K, I, L
