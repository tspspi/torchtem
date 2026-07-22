from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from torchtem.complex_support import abs2
from torchtem.fft_support import fft2, fft_interpolate
from torchtem.optics import CTF
from torchtem.physics import reciprocal_mesh


class DiffractionIntensity(nn.Module):
    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        diffraction = torch.fft.fft2(wave, norm="ortho")
        return abs2(diffraction)


class AnnularDetector(nn.Module):
    """Simple annular detector over diffraction intensity in mrad."""

    def __init__(
        self,
        *,
        energy_eV: float,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        inner_mrad: float,
        outer_mrad: float,
        offset_mrad: tuple[float, float] = (0.0, 0.0),
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        super().__init__()
        self.energy_eV = float(energy_eV)
        self.gpts = gpts
        self.sampling = sampling
        self.register_parameter("inner_mrad", nn.Parameter(torch.tensor(float(inner_mrad), dtype=dtype)))
        self.register_parameter("outer_mrad", nn.Parameter(torch.tensor(float(outer_mrad), dtype=dtype)))
        self.register_parameter("offset_y_mrad", nn.Parameter(torch.tensor(float(offset_mrad[0]), dtype=dtype)))
        self.register_parameter("offset_x_mrad", nn.Parameter(torch.tensor(float(offset_mrad[1]), dtype=dtype)))
        self.dtype = dtype
        self.device_hint = device
        self.intensity = DiffractionIntensity()

    def angle_components(self, wave: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from torchtem.physics import energy2wavelength

        wavelength = energy2wavelength(self.energy_eV, device=wave.device, dtype=self.dtype)
        ky, kx = reciprocal_mesh(self.gpts, self.sampling, device=wave.device, dtype=self.dtype)
        ay = 1e3 * wavelength * ky - self.offset_y_mrad.to(device=wave.device)
        ax = 1e3 * wavelength * kx - self.offset_x_mrad.to(device=wave.device)
        return ay, ax

    def mask(self, wave: torch.Tensor) -> torch.Tensor:
        ay, ax = self.angle_components(wave)
        alpha = torch.sqrt(ay.square() + ax.square())
        return ((alpha >= self.inner_mrad) & (alpha < self.outer_mrad)).to(self.dtype)

    def detector_region(self, wave: torch.Tensor) -> torch.Tensor:
        return self.mask(wave).to(torch.float64)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        intensity = self.intensity(wave)
        mask = self.mask(wave)
        return torch.sum(intensity * mask, dim=(-2, -1))


class FlexibleAnnularDetector(nn.Module):
    """Radially binned annular detector with configurable step size in mrad."""

    def __init__(
        self,
        *,
        energy_eV: float,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        step_size_mrad: float = 1.0,
        inner_mrad: float = 0.0,
        outer_mrad: float | None = None,
        offset_mrad: tuple[float, float] = (0.0, 0.0),
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        super().__init__()
        self.energy_eV = float(energy_eV)
        self.gpts = gpts
        self.sampling = sampling
        self.register_parameter("step_size_mrad", nn.Parameter(torch.tensor(float(step_size_mrad), dtype=dtype)))
        self.register_parameter("inner_mrad", nn.Parameter(torch.tensor(float(inner_mrad), dtype=dtype)))
        if outer_mrad is None:
            self.outer_mrad = None
        else:
            self.register_parameter("outer_mrad", nn.Parameter(torch.tensor(float(outer_mrad), dtype=dtype)))
        self.register_parameter("offset_y_mrad", nn.Parameter(torch.tensor(float(offset_mrad[0]), dtype=dtype)))
        self.register_parameter("offset_x_mrad", nn.Parameter(torch.tensor(float(offset_mrad[1]), dtype=dtype)))
        self.dtype = dtype
        self.device_hint = device
        self.intensity = DiffractionIntensity()

    def _angle_components(self, wave: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from torchtem.physics import energy2wavelength

        wavelength = energy2wavelength(self.energy_eV, device=wave.device, dtype=self.dtype)
        ky, kx = reciprocal_mesh(self.gpts, self.sampling, device=wave.device, dtype=self.dtype)
        ay = 1e3 * wavelength * ky - self.offset_y_mrad.to(device=wave.device)
        ax = 1e3 * wavelength * kx - self.offset_x_mrad.to(device=wave.device)
        return ay, ax

    def _angles(self, wave: torch.Tensor) -> torch.Tensor:
        ay, ax = self._angle_components(wave)
        return torch.sqrt(ay.square() + ax.square())

    def outer_limit(self, wave: torch.Tensor) -> float:
        if self.outer_mrad is not None:
            return float(self.outer_mrad.detach().item())
        alpha = self._angles(wave)
        return float(alpha.max().item())

    def nbins_radial(self, wave: torch.Tensor) -> int:
        outer = self.outer_limit(wave)
        inner = float(self.inner_mrad.detach().item())
        step = float(self.step_size_mrad.detach().item())
        return max(1, int(math.floor((outer - inner) / step)))

    def radial_bin_map(self, wave: torch.Tensor) -> torch.Tensor:
        alpha = self._angles(wave)
        outer = self.outer_limit(wave)
        nbins = self.nbins_radial(wave)
        bins = torch.full(self.gpts, -1, device=wave.device, dtype=torch.long)
        valid = (alpha >= self.inner_mrad) & (alpha < outer)
        radial = torch.floor((alpha - self.inner_mrad) / self.step_size_mrad).to(torch.long)
        radial = torch.clamp(radial, 0, nbins - 1)
        bins[valid] = radial[valid]
        return bins

    def detector_regions(self, wave: torch.Tensor) -> torch.Tensor:
        bins = self.radial_bin_map(wave)
        nbins = self.nbins_radial(wave)
        regions = torch.full(
            (nbins,) + self.gpts,
            float("nan"),
            device=wave.device,
            dtype=torch.float64,
        )
        ones = torch.ones(self.gpts, device=wave.device, dtype=torch.float64)
        nans = torch.full(self.gpts, float("nan"), device=wave.device, dtype=torch.float64)
        for idx in range(nbins):
            regions[idx] = torch.where(bins == idx, ones, nans)
        return regions

    def integrate_radial_bins(
        self,
        radial_bins: torch.Tensor,
        *,
        inner_mrad: float,
        outer_mrad: float | None = None,
    ) -> torch.Tensor:
        if radial_bins.ndim == 0:
            raise ValueError("radial_bins must have at least one dimension")

        outer = self.outer_mrad if outer_mrad is None else float(outer_mrad)
        inner = float(inner_mrad)
        if outer is None:
            outer = float(self.inner_mrad.detach().item()) + float(self.step_size_mrad.detach().item()) * radial_bins.shape[-1]

        base_inner = float(self.inner_mrad.detach().item())
        step = float(self.step_size_mrad.detach().item())
        start = max(0, int(math.floor((inner - base_inner) / step)))
        stop = min(
            radial_bins.shape[-1],
            int(math.ceil((float(outer) - base_inner) / step)),
        )
        if stop <= start:
            return torch.zeros(radial_bins.shape[:-1], device=radial_bins.device, dtype=radial_bins.dtype)
        return radial_bins[..., start:stop].sum(dim=-1)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        if wave.ndim > 2:
            return torch.stack([self.forward(item) for item in wave], dim=0)

        intensity = self.intensity(wave)
        bins = self.radial_bin_map(wave)
        nbins = self.nbins_radial(wave)

        out = torch.zeros(nbins, device=wave.device, dtype=intensity.dtype)
        valid = bins >= 0
        out.scatter_add_(0, bins[valid], intensity[valid])
        return out


class PixelatedDetector(nn.Module):
    """Return diffraction intensity or real-space intensity, optionally resampled."""

    def __init__(
        self,
        *,
        energy_eV: float | None = None,
        sampling: tuple[float, float] | None = None,
        reciprocal_space: bool = True,
        gpts: tuple[int, int] | None = None,
        max_angle_mrad: float | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.energy_eV = None if energy_eV is None else float(energy_eV)
        self.sampling = sampling
        self.reciprocal_space = bool(reciprocal_space)
        self.gpts = None if gpts is None else (int(gpts[0]), int(gpts[1]))
        if max_angle_mrad is None:
            self.max_angle_mrad = None
        else:
            self.register_parameter("max_angle_mrad", nn.Parameter(torch.tensor(float(max_angle_mrad), dtype=dtype)))
        self.dtype = dtype
        self.intensity = DiffractionIntensity()

    def reciprocal_mask(self, wave: torch.Tensor) -> torch.Tensor:
        if self.max_angle_mrad is None:
            return torch.ones(wave.shape[-2:], device=wave.device, dtype=self.dtype)
        if self.energy_eV is None or self.sampling is None:
            raise ValueError("energy_eV and sampling are required when max_angle_mrad is set")
        from torchtem.physics import energy2wavelength

        wavelength = energy2wavelength(self.energy_eV, device=wave.device, dtype=self.dtype)
        ky, kx = reciprocal_mesh(wave.shape[-2:], self.sampling, device=wave.device, dtype=self.dtype)
        alpha = 1e3 * wavelength * torch.sqrt(ky.square() + kx.square())
        return (alpha <= self.max_angle_mrad).to(self.dtype)

    def detector_region(self, wave: torch.Tensor) -> torch.Tensor:
        if self.reciprocal_space:
            return self.reciprocal_mask(wave).to(torch.float64)
        return torch.ones(wave.shape[-2:], device=wave.device, dtype=torch.float64)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        if self.reciprocal_space:
            output = self.intensity(wave)
            if self.max_angle_mrad is not None:
                output = output * self.reciprocal_mask(wave)
        else:
            output = abs2(wave)
        if self.gpts is not None and tuple(output.shape[-2:]) != self.gpts:
            output = F.interpolate(
                output.unsqueeze(0).unsqueeze(0),
                size=self.gpts,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0).squeeze(0)
        return output


class ImageDetector(nn.Module):
    """Return a real-space image, optionally after an image-forming CTF."""

    def __init__(
        self,
        *,
        ctf: CTF | None = None,
        output: str = "intensity",
    ) -> None:
        super().__init__()
        self.ctf = ctf
        self.output = str(output)

    def _apply_ctf(self, wave: torch.Tensor) -> torch.Tensor:
        if self.ctf is None:
            return wave
        transfer = self.ctf().to(wave.dtype)
        return torch.fft.ifft2(torch.fft.fft2(wave, norm="ortho") * transfer, norm="ortho")

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        image_wave = self._apply_ctf(wave)
        if self.output == "complex":
            return image_wave
        if self.output == "intensity":
            return abs2(image_wave)
        if self.output == "phase":
            return torch.angle(image_wave)
        if self.output == "real":
            return image_wave.real
        if self.output == "imag":
            return image_wave.imag
        raise ValueError(f"Unknown image detector output mode: {self.output}")


class WavesDetector(nn.Module):
    """Return the complex wave directly, optionally resampled or converted to reciprocal space."""

    def __init__(
        self,
        *,
        gpts: tuple[int, int] | None = None,
        reciprocal_space: bool = False,
    ) -> None:
        super().__init__()
        self.gpts = None if gpts is None else (int(gpts[0]), int(gpts[1]))
        self.reciprocal_space = bool(reciprocal_space)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        output = fft2(wave, norm="ortho") if self.reciprocal_space else wave
        if self.gpts is not None and tuple(output.shape[-2:]) != self.gpts:
            output = fft_interpolate(output, self.gpts, normalization="values")
        if self.reciprocal_space:
            return output
        if torch.is_complex(output):
            return output
        return output.to(torch.complex128)


class SegmentedDetector(nn.Module):
    """Angular/radial segmentation over diffraction intensity."""

    def __init__(
        self,
        *,
        energy_eV: float,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        inner_mrad: float,
        outer_mrad: float,
        nbins_radial: int,
        nbins_azimuthal: int,
        rotation_rad: float = 0.0,
        offset_mrad: tuple[float, float] = (0.0, 0.0),
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        super().__init__()
        self.energy_eV = float(energy_eV)
        self.gpts = gpts
        self.sampling = sampling
        self.register_parameter("inner_mrad", nn.Parameter(torch.tensor(float(inner_mrad), dtype=dtype)))
        self.register_parameter("outer_mrad", nn.Parameter(torch.tensor(float(outer_mrad), dtype=dtype)))
        self.nbins_radial = int(nbins_radial)
        self.nbins_azimuthal = int(nbins_azimuthal)
        self.register_parameter("rotation_rad", nn.Parameter(torch.tensor(float(rotation_rad), dtype=dtype)))
        self.register_parameter("offset_y_mrad", nn.Parameter(torch.tensor(float(offset_mrad[0]), dtype=dtype)))
        self.register_parameter("offset_x_mrad", nn.Parameter(torch.tensor(float(offset_mrad[1]), dtype=dtype)))
        self.dtype = dtype
        self.device_hint = device
        self.intensity = DiffractionIntensity()

    def bins(self, wave: torch.Tensor) -> torch.Tensor:
        from torchtem.physics import energy2wavelength

        wavelength = energy2wavelength(self.energy_eV, device=wave.device, dtype=self.dtype)
        ky, kx = reciprocal_mesh(self.gpts, self.sampling, device=wave.device, dtype=self.dtype)
        ay = 1e3 * wavelength * ky - self.offset_y_mrad.to(device=wave.device)
        ax = 1e3 * wavelength * kx - self.offset_x_mrad.to(device=wave.device)
        alpha = torch.sqrt(ay.square() + ax.square())
        phi = torch.remainder(torch.atan2(ax, ay) - self.rotation_rad.to(device=wave.device), 2.0 * math.pi)

        bins = torch.full(self.gpts, -1, device=wave.device, dtype=torch.long)
        valid = (alpha >= self.inner_mrad) & (alpha < self.outer_mrad)
        radial = torch.floor(
            self.nbins_radial * (alpha - self.inner_mrad) / (self.outer_mrad - self.inner_mrad)
        ).to(torch.long)
        radial = torch.clamp(radial, 0, self.nbins_radial - 1)
        azimuthal = torch.floor(self.nbins_azimuthal * phi / (2.0 * math.pi)).to(torch.long)
        azimuthal = torch.clamp(azimuthal, 0, self.nbins_azimuthal - 1)
        bins[valid] = radial[valid] * self.nbins_azimuthal + azimuthal[valid]
        return bins

    def detector_regions(self, wave: torch.Tensor) -> torch.Tensor:
        bins = self.bins(wave)
        regions = torch.full(
            (self.nbins_radial, self.nbins_azimuthal) + self.gpts,
            float("nan"),
            device=wave.device,
            dtype=torch.float64,
        )
        for r in range(self.nbins_radial):
            for a in range(self.nbins_azimuthal):
                idx = r * self.nbins_azimuthal + a
                regions[r, a] = torch.where(
                    bins == idx,
                    torch.ones(self.gpts, device=wave.device, dtype=torch.float64),
                    torch.full(self.gpts, float("nan"), device=wave.device, dtype=torch.float64),
                )
        return regions

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        if wave.ndim > 2:
            return torch.stack([self.forward(item) for item in wave], dim=0)

        intensity = self.intensity(wave)
        bins = self.bins(wave)
        out = torch.zeros(
            self.nbins_radial * self.nbins_azimuthal,
            device=wave.device,
            dtype=intensity.dtype,
        )
        valid = bins >= 0
        out.scatter_add_(0, bins[valid], intensity[valid])
        return out.reshape(self.nbins_radial, self.nbins_azimuthal)
