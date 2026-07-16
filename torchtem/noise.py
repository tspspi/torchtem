from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class PoissonNoise(nn.Module):
    """Poisson count noise with optional straight-through gradients."""

    def __init__(self, *, dose: float = 1.0, straight_through: bool = False) -> None:
        super().__init__()
        self.dose = float(dose)
        self.straight_through = bool(straight_through)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        lam = torch.clamp(image, min=0.0) * self.dose
        sample = torch.poisson(lam)
        if self.straight_through:
            return sample + (lam - lam.detach())
        return sample


class ScanDistortion(nn.Module):
    """Apply a smooth distortion field to a 2D image using grid sampling."""

    def __init__(
        self,
        *,
        rms_power: float,
        max_frequency: float,
        num_components: int,
        seed: int = 0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.rms_power = float(rms_power)
        self.max_frequency = float(max_frequency)
        self.num_components = int(num_components)
        self.seed = int(seed)
        self.dtype = dtype

    def _pixel_times(self, shape: tuple[int, int], dwell_time: float, flyback_time: float) -> torch.Tensor:
        line_time = dwell_time * shape[0] + flyback_time
        slow = torch.linspace(line_time, shape[1] * line_time, shape[1], dtype=self.dtype)
        slow = slow.repeat(shape[0], 1)
        fast = torch.linspace(
            (line_time - flyback_time) / shape[1],
            line_time - flyback_time,
            shape[0],
            dtype=self.dtype,
        ).unsqueeze(1).repeat(1, shape[1])
        return slow + fast

    def _single_axis_distortion(self, time: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        frequencies = torch.rand((self.num_components, 1, 1), generator=generator, dtype=self.dtype) * self.max_frequency
        amplitudes = torch.rand((self.num_components, 1, 1), generator=generator, dtype=self.dtype) / torch.sqrt(
            torch.clamp(frequencies, min=1e-12)
        )
        displacements = torch.rand((self.num_components, 1, 1), generator=generator, dtype=self.dtype) / torch.clamp(
            frequencies, min=1e-12
        )
        return (amplitudes * torch.sin(2.0 * math.pi * (time + displacements) * frequencies)).sum(dim=0)

    def displacement_field(
        self, shape: tuple[int, int], dwell_time: float, flyback_time: float, device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time = self._pixel_times(shape, dwell_time, flyback_time).to(device=device)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        profile_x = self._single_axis_distortion(time.cpu(), generator).to(device=device)
        profile_y = self._single_axis_distortion(time.cpu(), generator).to(device=device)
        grad_x = torch.gradient(profile_x, dim=1)[0]
        grad_y = torch.gradient(profile_y, dim=0)[0]
        frame_dev = torch.sqrt(torch.mean((((1 + grad_x) * (1 + grad_y) - 1) ** 2))) + 1e-12
        scale = self.rms_power / (2.355 * 100.0 * frame_dev)
        return profile_x * scale, profile_y * scale

    def forward(self, image: torch.Tensor, dwell_time: float, flyback_time: float) -> torch.Tensor:
        if image.ndim != 2:
            raise ValueError("ScanDistortion expects a 2D image")
        h, w = image.shape
        dx, dy = self.displacement_field((h, w), dwell_time, flyback_time, image.device)
        y = torch.linspace(-1.0, 1.0, h, device=image.device, dtype=self.dtype)
        x = torch.linspace(-1.0, 1.0, w, device=image.device, dtype=self.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack(
            (
                xx + 2.0 * dx / max(w - 1, 1),
                yy + 2.0 * dy / max(h - 1, 1),
            ),
            dim=-1,
        ).unsqueeze(0)
        sampled = F.grid_sample(
            image[None, None].to(self.dtype),
            grid.to(self.dtype),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled[0, 0]
