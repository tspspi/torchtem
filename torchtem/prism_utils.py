from __future__ import annotations

from typing import Tuple

import torch

from tspi.torchtem.physics import energy2wavelength


def batch_crop_2d(
    array: torch.Tensor, corners: torch.Tensor, new_shape: Tuple[int, int]
) -> torch.Tensor:
    """Crop a batch of 2D arrays at integer corners."""
    old_shape = None

    if array.ndim == 2:
        array = array.unsqueeze(0)
    elif array.ndim > 3:
        old_shape = array.shape
        batch_shape = array.shape[: -len(corners.shape) - 1]
        array = array.reshape((-1,) + array.shape[-2:])
        corners = corners.reshape((-1, 2))

        if batch_shape:
            if array.shape[0] != corners.shape[0] * int(torch.tensor(batch_shape).prod().item()):
                raise ValueError("Array batch shape and corners shape are incompatible")
            corners = corners.repeat(int(torch.tensor(batch_shape).prod().item()), 1)

    if corners.ndim == 1:
        corners = corners.unsqueeze(0)
    if array.shape[0] != corners.shape[0]:
        raise ValueError("Leading batch dimension of array and corners must match")

    crops = []
    for i in range(array.shape[0]):
        y0 = int(corners[i, 0].item())
        x0 = int(corners[i, 1].item())
        crops.append(array[i, y0 : y0 + new_shape[0], x0 : x0 + new_shape[1]])
    cropped = torch.stack(crops, dim=0)

    if old_shape is not None:
        cropped = cropped.reshape(old_shape[:-2] + cropped.shape[-2:])
    return cropped


def minimum_crop(
    positions: torch.Tensor, shape: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int], torch.Tensor]:
    offset = torch.tensor((shape[0] // 2, shape[1] // 2), device=positions.device)
    corners = torch.round(positions - offset).to(torch.int64)
    upper_corners = corners + torch.tensor(shape, device=positions.device)

    crop_corner = (
        int(torch.min(corners[..., 0]).item()),
        int(torch.min(corners[..., 1]).item()),
    )
    size = (
        int(torch.max(upper_corners[..., 0]).item() - crop_corner[0]),
        int(torch.max(upper_corners[..., 1]).item() - crop_corner[1]),
    )
    corners = corners - torch.tensor(crop_corner, device=positions.device)
    return crop_corner, size, corners


def wrapped_slices(start: int, stop: int, n: int) -> tuple[slice, slice]:
    if start < 0:
        if stop > n:
            raise RuntimeError(f"start = {start} stop = {stop}, n = {n}")
        return slice(start % n, None), slice(0, stop)
    if stop > n:
        if start < 0:
            raise RuntimeError(f"start = {start} stop = {stop}, n = {n}")
        return slice(start, None), slice(0, stop - n)
    return slice(start, stop), slice(0, 0)


def wrapped_crop_2d(
    array: torch.Tensor, corner: tuple[int, int], size: tuple[int, int]
) -> torch.Tensor:
    upper_corner = (corner[0] + size[0], corner[1] + size[1])
    a, c = wrapped_slices(corner[0], upper_corner[0], array.shape[-2])
    b, d = wrapped_slices(corner[1], upper_corner[1], array.shape[-1])

    A = array[..., a, b]
    B = array[..., c, b]
    D = array[..., c, d]
    C = array[..., a, d]

    if A.numel() == 0:
        AB = B
    elif B.numel() == 0:
        AB = A
    else:
        AB = torch.cat([A, B], dim=-2)

    if C.numel() == 0:
        CD = D
    elif D.numel() == 0:
        CD = C
    else:
        CD = torch.cat([C, D], dim=-2)

    if CD.numel() == 0:
        return AB
    if AB.numel() == 0:
        return CD
    return torch.cat([AB, CD], dim=-1)


def _planewave_shift_coefficients(
    positions: torch.Tensor, wave_vectors: torch.Tensor
) -> torch.Tensor:
    coefficients = torch.exp(
        -2.0j * torch.pi * positions[..., 0, None] * wave_vectors[:, 0][None]
    )
    coefficients = coefficients * torch.exp(
        -2.0j * torch.pi * positions[..., 1, None] * wave_vectors[:, 1][None]
    )
    return coefficients


def prism_coefficients(
    positions: torch.Tensor,
    wave_vectors: torch.Tensor,
    *,
    ctf_coefficients: torch.Tensor | None = None,
) -> torch.Tensor:
    coefficients = _planewave_shift_coefficients(positions, wave_vectors)
    if ctf_coefficients is not None:
        coefficients = coefficients * ctf_coefficients.to(coefficients.dtype)
    return coefficients


def prism_wave_vectors(
    cutoff_mrad: float,
    extent_A: tuple[float, float],
    energy_eV: float,
    interpolation: tuple[int, int],
    *,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    wavelength = energy2wavelength(energy_eV, device=device, dtype=dtype)
    n_term = cutoff_mrad / 1.0e3 / (wavelength / extent_A[0] * interpolation[0])
    m_term = cutoff_mrad / 1.0e3 / (wavelength / extent_A[1] * interpolation[1])
    n_max = int(
        torch.ceil(torch.as_tensor(n_term, device=device, dtype=dtype)).item()
    )
    m_max = int(
        torch.ceil(torch.as_tensor(m_term, device=device, dtype=dtype)).item()
    )

    n = torch.arange(-n_max, n_max + 1, dtype=dtype, device=device)
    m = torch.arange(-m_max, m_max + 1, dtype=dtype, device=device)
    kx = n / extent_A[0] * interpolation[0]
    ky = m / extent_A[1] * interpolation[1]
    KX, KY = torch.meshgrid(kx, ky, indexing="ij")
    mask = KX.square() + KY.square() < (cutoff_mrad / 1.0e3 / wavelength) ** 2
    return torch.stack((KX[mask], KY[mask]), dim=-1)
