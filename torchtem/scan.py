from __future__ import annotations

import math

import torch
from torch import nn

from torchtem.physics import ensure_tuple2, reciprocal_mesh


def fft_shift_wave(
    wave: torch.Tensor,
    positions_A: torch.Tensor,
    sampling: tuple[float, float],
) -> torch.Tensor:
    """Shift a complex wave in real space using the Fourier shift theorem.

    Parameters
    ----------
    wave:
        Complex wave of shape ``(ny, nx)``.
    positions_A:
        Real-space shifts in angstrom with shape ``(..., 2)`` in ``(y, x)`` order.
    sampling:
        Real-space sampling in angstrom per pixel.
    """
    if positions_A.ndim == 1:
        positions_A = positions_A.unsqueeze(0)

    freq_y, freq_x = reciprocal_mesh(
        (wave.shape[-2], wave.shape[-1]),
        sampling,
        device=wave.device,
        dtype=torch.float64,
    )
    fft_wave = torch.fft.fft2(wave, norm="ortho")
    phase = (
        -2.0j
        * math.pi
        * (
            positions_A[:, 0, None, None].to(freq_y.dtype) * freq_y[None]
            + positions_A[:, 1, None, None].to(freq_x.dtype) * freq_x[None]
        )
    )
    phase = torch.exp(phase).reshape((positions_A.shape[0],) + (1,) * (wave.ndim - 2) + wave.shape[-2:])
    shifted = torch.fft.ifft2(fft_wave.unsqueeze(0) * phase, norm="ortho")
    return shifted if shifted.shape[0] > 1 else shifted[0]


class GridScan(nn.Module):
    """Regular raster scan defined by start, end, and shape."""

    def __init__(
        self,
        *,
        start_A: tuple[float, float] | float,
        end_A: tuple[float, float] | float,
        shape: tuple[int, int],
        device=None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        start_y, start_x = ensure_tuple2(start_A)
        end_y, end_x = ensure_tuple2(end_A)
        self.register_parameter("start_y_A", nn.Parameter(torch.tensor(float(start_y), dtype=dtype)))
        self.register_parameter("start_x_A", nn.Parameter(torch.tensor(float(start_x), dtype=dtype)))
        self.register_parameter("end_y_A", nn.Parameter(torch.tensor(float(end_y), dtype=dtype)))
        self.register_parameter("end_x_A", nn.Parameter(torch.tensor(float(end_x), dtype=dtype)))
        self.shape = (int(shape[0]), int(shape[1]))
        self.device_hint = device
        self.dtype = dtype

    def positions(self) -> torch.Tensor:
        y_steps = torch.linspace(0.0, 1.0, self.shape[0], device=self.device_hint, dtype=self.dtype)
        x_steps = torch.linspace(0.0, 1.0, self.shape[1], device=self.device_hint, dtype=self.dtype)
        y = self.start_y_A + (self.end_y_A - self.start_y_A) * y_steps
        x = self.start_x_A + (self.end_x_A - self.start_x_A) * x_steps
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        return torch.stack((yy, xx), dim=-1).reshape(-1, 2)

    def forward(self) -> torch.Tensor:
        return self.positions()


class CustomScan(nn.Module):
    """User-provided scan positions."""

    def __init__(self, positions_A: torch.Tensor) -> None:
        super().__init__()
        self.register_parameter(
            "positions_A",
            nn.Parameter(positions_A.to(torch.float64)),
        )

    def forward(self) -> torch.Tensor:
        return self.positions_A
