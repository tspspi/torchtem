from __future__ import annotations

from itertools import product

import torch

from torchtem.complex_support import complex_exponential_scaled
from torchtem.physics import reciprocal_mesh


def fft2(x: torch.Tensor, norm: str = "backward") -> torch.Tensor:
    return torch.fft.fft2(x, norm=norm)


def ifft2(x: torch.Tensor, norm: str = "backward") -> torch.Tensor:
    return torch.fft.ifft2(x, norm=norm)


def fftn(x: torch.Tensor, dim: tuple[int, ...] | None = None, norm: str = "backward") -> torch.Tensor:
    return torch.fft.fftn(x, dim=dim, norm=norm)


def ifftn(x: torch.Tensor, dim: tuple[int, ...] | None = None, norm: str = "backward") -> torch.Tensor:
    return torch.fft.ifftn(x, dim=dim, norm=norm)


def fft2_convolve(x: torch.Tensor, kernel: torch.Tensor, norm: str = "backward") -> torch.Tensor:
    return ifft2(fft2(x, norm=norm) * kernel.to(torch.complex128), norm=norm)


def fft_shift_kernel(positions: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)
    if positions.shape[-1] != len(shape):
        raise ValueError("Last dimension of positions must match number of shifted dimensions")

    freqs = reciprocal_mesh(shape, (1.0,) * len(shape), device=positions.device, dtype=torch.float64)
    kernel = torch.ones(positions.shape[:-1] + shape, dtype=torch.complex128, device=positions.device)
    for i, freq in enumerate(freqs):
        pos = positions[..., i]
        while pos.ndim < kernel.ndim - len(shape):
            pos = pos.unsqueeze(-1)
        phase = freq
        for _ in range(kernel.ndim - phase.ndim):
            phase = phase.unsqueeze(0)
        kernel = kernel * complex_exponential_scaled(
            phase,
            -2.0 * torch.pi * pos.unsqueeze(-1).unsqueeze(-1),
        )
    return kernel


def fft_shift(array: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    shifted = ifft2(fft2(array) * fft_shift_kernel(positions, array.shape[-2:]))
    return shifted[0] if positions.ndim == 1 else shifted


def _fft_interpolation_slices_1d(n1: int, n2: int) -> list[tuple[slice, slice]]:
    n = min(n1, n2)
    if n == 1:
        return [(slice(0, 1), slice(0, 1))]
    if n % 2 == 0:
        h = n // 2
        t = n // 2
    else:
        h = n // 2 + 1
        t = n // 2
    return [
        (slice(0, h), slice(0, h)),
        (slice(n1 - t, n1), slice(n2 - t, n2)),
    ]


def fft_interpolation_masks(
    shape_in: tuple[int, ...], shape_out: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    mask1_1d = []
    mask2_1d = []
    for i, (n1, n2) in enumerate(zip(shape_in, shape_out)):
        m1 = torch.zeros(n1, dtype=torch.bool)
        m2 = torch.zeros(n2, dtype=torch.bool)
        for s1, s2 in _fft_interpolation_slices_1d(n1, n2):
            m1[s1] = True
            m2[s2] = True
        view = [None] * len(shape_in)
        view[i] = slice(None)
        mask1_1d.append(m1[tuple(view)])
        mask2_1d.append(m2[tuple(view)])

    mask1 = mask1_1d[0]
    for m in mask1_1d[1:]:
        mask1 = mask1 & m
    mask2 = mask2_1d[0]
    for m in mask2_1d[1:]:
        mask2 = mask2 & m
    return mask1, mask2


def fft_crop(array: torch.Tensor, new_shape: tuple[int, ...], normalize: bool = False) -> torch.Tensor:
    if len(new_shape) < len(array.shape):
        new_shape = array.shape[: -len(new_shape)] + new_shape

    slice_pairs: list[list[tuple[slice, slice]]] = []
    for n1, n2 in zip(array.shape, new_shape):
        if n1 == n2:
            slice_pairs.append([(slice(None), slice(None))])
        else:
            slice_pairs.append(_fft_interpolation_slices_1d(n1, n2))

    new_array = torch.zeros(new_shape, dtype=array.dtype, device=array.device)
    for combo in product(*slice_pairs):
        in_sl = tuple(p[0] for p in combo)
        out_sl = tuple(p[1] for p in combo)
        new_array[out_sl] = array[in_sl]

    if normalize:
        new_array = new_array * (new_array.numel() / array.numel())
    return new_array


def fft_interpolate(
    array: torch.Tensor,
    new_shape: tuple[int, ...],
    normalization: str = "values",
) -> torch.Tensor:
    old_size = 1
    for n in array.shape[-len(new_shape) :]:
        old_size *= n

    is_complex = torch.is_complex(array)
    working = array.to(torch.complex128)
    if len(new_shape) == 2:
        working = fft2(working)
        working = fft_crop(working, new_shape)
        working = ifft2(working)
    else:
        if len(new_shape) != len(array.shape):
            axes = tuple(range(len(array.shape) - len(new_shape), len(array.shape)))
        else:
            axes = tuple(range(len(array.shape)))
        working = fftn(working, dim=axes)
        working = fft_crop(working, new_shape)
        working = ifftn(working, dim=axes)

    if not is_complex:
        working = working.real

    new_size = 1
    for n in new_shape:
        new_size *= n
    if normalization == "values":
        working = working * (new_size / old_size)
    elif normalization in ("amplitude", "intensity"):
        pass
    else:
        raise ValueError(f"Normalization '{normalization}' not recognized.")
    return working
