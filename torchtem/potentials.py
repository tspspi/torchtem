from __future__ import annotations

import torch
from torch import nn

from torchtem.physics import ensure_tuple2, real_space_mesh


def render_gaussian_projection(
    *,
    gpts: tuple[int, int],
    sampling: tuple[float, float] | float,
    positions_A: torch.Tensor,
    amplitudes: torch.Tensor,
    sigmas_A: torch.Tensor,
    dtype: torch.dtype = torch.float64,
    device=None,
) -> torch.Tensor:
    sampling = ensure_tuple2(sampling)
    yy, xx = real_space_mesh(gpts, sampling, device=device or positions_A.device, dtype=dtype)
    dy = yy.unsqueeze(0) - positions_A[:, 0].unsqueeze(-1).unsqueeze(-1)
    dx = xx.unsqueeze(0) - positions_A[:, 1].unsqueeze(-1).unsqueeze(-1)
    sigma2 = sigmas_A.square().unsqueeze(-1).unsqueeze(-1)
    gaussian = torch.exp(-(dx.square() + dy.square()) / (2.0 * sigma2))
    amplitude = amplitudes.unsqueeze(-1).unsqueeze(-1)
    return torch.sum(amplitude * gaussian, dim=0)


class GaussianAtomProjection(nn.Module):
    """Differentiable projected potential built from 2D Gaussian atoms.

    This is not a full IAM port. It is an intentionally smooth, trainable surrogate
    potential layer that supports gradients with respect to atom positions, amplitudes,
    and widths while the higher-level multislice/optics stack is being rebuilt.
    """

    def __init__(
        self,
        *,
        gpts: tuple[int, int],
        sampling: tuple[float, float] | float,
        positions_A: torch.Tensor,
        amplitudes: torch.Tensor,
        sigmas_A: torch.Tensor,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        super().__init__()
        self.gpts = gpts
        self.sampling = ensure_tuple2(sampling)
        self.dtype = dtype
        self.device_hint = device
        self.positions_A = nn.Parameter(positions_A.to(device=device, dtype=dtype))
        self.amplitudes = nn.Parameter(amplitudes.to(device=device, dtype=dtype))
        self.log_sigmas_A = nn.Parameter(sigmas_A.to(device=device, dtype=dtype).log())

    @property
    def sigmas_A(self) -> torch.Tensor:
        return self.log_sigmas_A.exp()

    def forward(self) -> torch.Tensor:
        return render_gaussian_projection(
            gpts=self.gpts,
            sampling=self.sampling,
            positions_A=self.positions_A,
            amplitudes=self.amplitudes,
            sigmas_A=self.sigmas_A,
            dtype=self.dtype,
            device=self.positions_A.device,
        )
