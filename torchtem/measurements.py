from __future__ import annotations

import torch
from torch import nn

from tspi.torchtem.physics import reciprocal_mesh


class GaussianBlur(nn.Module):
    """Differentiable Gaussian blur via FFT convolution."""

    def __init__(
        self,
        sigma_px: float | tuple[float, float],
        *,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if isinstance(sigma_px, (tuple, list)):
            self.sigma_px = (float(sigma_px[0]), float(sigma_px[1]))
        else:
            self.sigma_px = (float(sigma_px), float(sigma_px))
        self.dtype = dtype

    def _kernel(self, shape: tuple[int, int], device) -> torch.Tensor:
        y = torch.arange(shape[0], device=device, dtype=self.dtype)
        x = torch.arange(shape[1], device=device, dtype=self.dtype)
        y = torch.minimum(y, shape[0] - y)
        x = torch.minimum(x, shape[1] - x)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        kernel = torch.exp(
            -0.5
            * ((yy / max(self.sigma_px[0], 1e-12)) ** 2 + (xx / max(self.sigma_px[1], 1e-12)) ** 2)
        )
        kernel = kernel / kernel.sum()
        return kernel

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        kernel = self._kernel((image.shape[-2], image.shape[-1]), image.device)
        kernel_ft = torch.fft.fft2(kernel, norm="ortho")
        image_ft = torch.fft.fft2(image, norm="ortho")
        out = torch.fft.ifft2(image_ft * kernel_ft, norm="ortho")
        return out.real if not torch.is_complex(image) else out


class CenterOfMass(nn.Module):
    """Center of mass of reciprocal-space intensity."""

    def __init__(
        self,
        *,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        super().__init__()
        self.gpts = gpts
        self.sampling = sampling
        self.dtype = dtype
        self.device_hint = device

    def forward(self, diffraction_intensity: torch.Tensor) -> torch.Tensor:
        ky, kx = reciprocal_mesh(
            self.gpts, self.sampling, device=diffraction_intensity.device, dtype=self.dtype
        )
        total = diffraction_intensity.sum(dim=(-2, -1), keepdim=True) + 1e-30
        com_y = (diffraction_intensity * ky).sum(dim=(-2, -1), keepdim=True) / total
        com_x = (diffraction_intensity * kx).sum(dim=(-2, -1), keepdim=True) / total
        return torch.stack((com_y.squeeze(-1).squeeze(-1), com_x.squeeze(-1).squeeze(-1)), dim=-1)


class IntegrateGradient(nn.Module):
    """Integrate a 2D gradient field by Fourier inversion."""

    def __init__(self, *, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(self, gradient: torch.Tensor, sampling: tuple[float, float]) -> torch.Tensor:
        gy = gradient[..., 0, :, :]
        gx = gradient[..., 1, :, :]
        fft_gy = torch.fft.fft2(gy, norm="ortho")
        fft_gx = torch.fft.fft2(gx, norm="ortho")
        ky, kx = reciprocal_mesh(
            (gradient.shape[-2], gradient.shape[-1]),
            sampling,
            device=gradient.device,
            dtype=self.dtype,
        )
        kx_c = kx.to(torch.complex128)
        ky_c = ky.to(torch.complex128)
        k2 = kx.square() + ky.square()
        denom = (-4.0 * torch.pi**2) * k2
        denom = denom.to(torch.complex128)
        denom[0, 0] = 1.0 + 0.0j
        divergence_ft = 2.0j * torch.pi * (kx_c * fft_gx + ky_c * fft_gy)
        potential_ft = divergence_ft / denom
        potential_ft[..., 0, 0] = 0.0
        return torch.fft.ifft2(potential_ft, norm="ortho").real
