from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
from torch import nn

from tspi.torchtem.measurements import CenterOfMass, GaussianBlur, IntegrateGradient
from tspi.torchtem.mtf import MTF
from tspi.torchtem.noise import PoissonNoise, ScanDistortion


@dataclass
class GaussianBlurConfig:
    sigma_px: float | tuple[float, float]


@dataclass
class PoissonNoiseConfig:
    dose: float = 1.0
    straight_through: bool = False


@dataclass
class ScanDistortionConfig:
    rms_power: float
    max_frequency: float
    num_components: int
    dwell_time: float
    flyback_time: float
    seed: int = 0


@dataclass
class MTFConfig:
    c0: float
    c1: float
    c2: float
    c3: float


@dataclass
class CenterOfMassConfig:
    gpts: tuple[int, int] | None = None
    sampling: tuple[float, float] | None = None


@dataclass
class IntegrateGradientConfig:
    sampling: tuple[float, float] | None = None


PipelineStepConfig = (
    GaussianBlurConfig
    | PoissonNoiseConfig
    | ScanDistortionConfig
    | MTFConfig
    | CenterOfMassConfig
    | IntegrateGradientConfig
)


class ConfiguredMTF(nn.Module):
    def __init__(self, mtf: MTF, *, sampling: tuple[float, float]) -> None:
        super().__init__()
        self.mtf = mtf
        self.sampling = sampling

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.mtf(image, sampling=self.sampling)


class ConfiguredIntegrateGradient(nn.Module):
    def __init__(self, integrator: IntegrateGradient, *, sampling: tuple[float, float]) -> None:
        super().__init__()
        self.integrator = integrator
        self.sampling = sampling

    def forward(self, gradient: torch.Tensor) -> torch.Tensor:
        return self.integrator(gradient, sampling=self.sampling)


class ConfiguredScanDistortion(nn.Module):
    def __init__(
        self,
        distortion: ScanDistortion,
        *,
        dwell_time: float,
        flyback_time: float,
    ) -> None:
        super().__init__()
        self.distortion = distortion
        self.dwell_time = float(dwell_time)
        self.flyback_time = float(flyback_time)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.distortion(image, self.dwell_time, self.flyback_time)


class MeasurementPipeline(nn.Module):
    """Apply a sequence of measurement/noise operators to detector outputs."""

    def __init__(self, steps: Sequence[nn.Module]) -> None:
        super().__init__()
        self.steps = nn.ModuleList(list(steps))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        current = value
        for step in self.steps:
            current = step(current)
        return current


def build_pipeline_step(
    config: PipelineStepConfig,
    *,
    gpts: tuple[int, int],
    sampling: tuple[float, float],
) -> nn.Module:
    if isinstance(config, GaussianBlurConfig):
        return GaussianBlur(config.sigma_px)
    if isinstance(config, PoissonNoiseConfig):
        return PoissonNoise(dose=config.dose, straight_through=config.straight_through)
    if isinstance(config, ScanDistortionConfig):
        return ConfiguredScanDistortion(
            ScanDistortion(
                rms_power=config.rms_power,
                max_frequency=config.max_frequency,
                num_components=config.num_components,
                seed=config.seed,
            ),
            dwell_time=config.dwell_time,
            flyback_time=config.flyback_time,
        )
    if isinstance(config, MTFConfig):
        return ConfiguredMTF(
            MTF(c0=config.c0, c1=config.c1, c2=config.c2, c3=config.c3),
            sampling=sampling,
        )
    if isinstance(config, CenterOfMassConfig):
        return CenterOfMass(
            gpts=config.gpts or gpts,
            sampling=config.sampling or sampling,
        )
    return ConfiguredIntegrateGradient(
        IntegrateGradient(),
        sampling=config.sampling or sampling,
    )


def build_measurement_pipeline(
    configs: Sequence[PipelineStepConfig],
    *,
    gpts: tuple[int, int],
    sampling: tuple[float, float],
) -> MeasurementPipeline:
    return MeasurementPipeline(
        [build_pipeline_step(config, gpts=gpts, sampling=sampling) for config in configs]
    )
