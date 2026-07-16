from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from torch import nn

from torchtem.array_object import ArrayObject
from torchtem.axes import AxisMetadata, RealSpaceAxis, ReciprocalSpaceAxis
from torchtem.fft_support import fft2, ifft2
from torchtem.grid import Grid, HasGrid2DMixin
from torchtem.optics import CTF, PlaneWave as PlaneWaveField, Probe as ProbeField
from torchtem.physics import energy2wavelength


class BaseWaves(HasGrid2DMixin, ABC):
    @property
    @abstractmethod
    def device(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def metadata(self) -> dict:
        raise NotImplementedError

    @property
    def energy_eV(self) -> float:
        return float(self.metadata["energy"])

    @property
    def wavelength(self) -> torch.Tensor:
        return energy2wavelength(self.energy_eV)

    @property
    def base_axes_metadata(self) -> list[AxisMetadata]:
        self.grid.check_is_defined()
        assert self.sampling is not None
        return [
            RealSpaceAxis(label="x", sampling=self.sampling[0], units="Å", endpoint=False),
            RealSpaceAxis(label="y", sampling=self.sampling[1], units="Å", endpoint=False),
        ]

    @property
    def reciprocal_space_axes_metadata(self) -> list[AxisMetadata]:
        self.grid.check_is_defined()
        return [
            ReciprocalSpaceAxis(
                label="scattering angle x",
                sampling=self.angular_sampling[0],
                units="mrad",
            ),
            ReciprocalSpaceAxis(
                label="scattering angle y",
                sampling=self.angular_sampling[1],
                units="mrad",
            ),
        ]

    @property
    def angular_sampling(self) -> tuple[float, float]:
        wavelength = float(self.wavelength.item())
        reciprocal = self.grid.reciprocal_space_sampling
        return (
            1e3 * wavelength * reciprocal[0],
            1e3 * wavelength * reciprocal[1],
        )


class Waves(BaseWaves, ArrayObject):
    _base_dims = 2

    def __init__(
        self,
        array,
        *,
        energy_eV: float,
        extent: Optional[float | tuple[float, float]] = None,
        sampling: Optional[float | tuple[float, float]] = None,
        reciprocal_space: bool = False,
        ensemble_axes_metadata: Optional[list[AxisMetadata]] = None,
        metadata: Optional[dict] = None,
    ):
        if sampling is not None and extent is not None:
            extent = None
        self._grid = Grid(extent=extent, gpts=array.shape[-2:], sampling=sampling, lock_gpts=True)
        self._reciprocal_space = bool(reciprocal_space)
        self._energy_eV = float(energy_eV)
        super().__init__(
            array=array,
            ensemble_axes_metadata=ensemble_axes_metadata,
            metadata=metadata,
        )

    @property
    def device(self) -> str:
        if isinstance(self.array, torch.Tensor):
            return "gpu" if self.array.is_cuda else "cpu"
        return "cpu"

    @property
    def reciprocal_space(self) -> bool:
        return self._reciprocal_space

    @property
    def metadata(self) -> dict:
        self._metadata["energy"] = self._energy_eV
        self._metadata["reciprocal_space"] = self.reciprocal_space
        return self._metadata

    @classmethod
    def from_array_and_metadata(
        cls,
        array,
        axes_metadata: list[AxisMetadata],
        metadata: dict,
    ) -> "Waves":
        energy_eV = metadata["energy"]
        reciprocal_space = metadata.get("reciprocal_space", False)
        x_axis, y_axis = axes_metadata[-2], axes_metadata[-1]
        if not isinstance(x_axis, RealSpaceAxis) or not isinstance(y_axis, RealSpaceAxis):
            raise ValueError("Last two axes must be RealSpaceAxis")
        sampling = (x_axis.sampling, y_axis.sampling)
        return cls(
            array=array,
            energy_eV=energy_eV,
            sampling=sampling,
            reciprocal_space=reciprocal_space,
            ensemble_axes_metadata=axes_metadata[:-2],
            metadata=metadata,
        )

    def intensity(self) -> torch.Tensor:
        array = self.array
        if not isinstance(array, torch.Tensor):
            array = torch.as_tensor(array)
        return array.abs().square()

    def phase(self) -> torch.Tensor:
        array = self.array
        if not isinstance(array, torch.Tensor):
            array = torch.as_tensor(array)
        return torch.angle(array)

    def diffraction_patterns(self) -> "Waves":
        array = self.array if isinstance(self.array, torch.Tensor) else torch.as_tensor(self.array)
        transformed = fft2(array, norm="ortho") if not self.reciprocal_space else array
        return Waves(
            transformed,
            energy_eV=self.energy_eV,
            sampling=self.sampling,
            reciprocal_space=True,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=dict(self.metadata),
        )

    def real_space(self) -> "Waves":
        array = self.array if isinstance(self.array, torch.Tensor) else torch.as_tensor(self.array)
        transformed = ifft2(array, norm="ortho") if self.reciprocal_space else array
        return Waves(
            transformed,
            energy_eV=self.energy_eV,
            sampling=self.sampling,
            reciprocal_space=False,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=dict(self.metadata),
        )

    def normalize(self, mode: str = "values") -> "Waves":
        array = self.array if isinstance(self.array, torch.Tensor) else torch.as_tensor(self.array)
        if mode == "values":
            denom = array.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        elif mode == "reciprocal_space":
            reciprocal = fft2(array, norm="ortho") if not self.reciprocal_space else array
            denom = reciprocal.abs().square().sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-12)
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")
        return Waves(
            array / denom,
            energy_eV=self.energy_eV,
            sampling=self.sampling,
            reciprocal_space=self.reciprocal_space,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=dict(self.metadata),
        )

    def apply_ctf(self, ctf: CTF) -> "Waves":
        array = self.array if isinstance(self.array, torch.Tensor) else torch.as_tensor(self.array)
        reciprocal = array if self.reciprocal_space else fft2(array, norm="ortho")
        transformed = reciprocal * ctf().to(reciprocal.dtype)
        if self.reciprocal_space:
            output = transformed
            reciprocal_space = True
        else:
            output = ifft2(transformed, norm="ortho")
            reciprocal_space = False
        return Waves(
            output,
            energy_eV=self.energy_eV,
            sampling=self.sampling,
            reciprocal_space=reciprocal_space,
            ensemble_axes_metadata=self.ensemble_axes_metadata,
            metadata=dict(self.metadata),
        )


class WavesBuilder(BaseWaves, nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def device(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def metadata(self) -> dict:
        raise NotImplementedError

    @property
    def base_shape(self) -> tuple[int, int]:
        return self._valid_gpts

    @abstractmethod
    def build(self) -> Waves:
        raise NotImplementedError

    def forward(self) -> Waves:
        return self.build()


class PlaneWaveBuilder(WavesBuilder):
    def __init__(
        self,
        *,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        energy_eV: float,
        amplitude: complex = 1.0 + 0.0j,
        device=None,
    ) -> None:
        super().__init__()
        self._grid = Grid(gpts=gpts, sampling=sampling, lock_gpts=True)
        self.field = PlaneWaveField(gpts=gpts, amplitude=amplitude, device=device)
        self._energy_eV = float(energy_eV)
        self._device = "gpu" if (device == "cuda") else "cpu"

    @property
    def device(self) -> str:
        return self._device

    @property
    def metadata(self) -> dict:
        return {"energy": self._energy_eV, "normalization": "values"}

    def build(self) -> Waves:
        return Waves(
            self.field(),
            energy_eV=self._energy_eV,
            sampling=self.sampling,
            reciprocal_space=False,
            metadata=self.metadata,
        )


class ProbeWavesBuilder(WavesBuilder):
    def __init__(self, probe: ProbeField) -> None:
        super().__init__()
        self.probe = probe
        self._grid = Grid(gpts=probe.ctf.gpts, sampling=probe.ctf.sampling, lock_gpts=True)
        self._energy_eV = float(probe.ctf.energy_eV)
        self._device = "cpu"

    @property
    def device(self) -> str:
        return self._device

    @property
    def metadata(self) -> dict:
        return {"energy": self._energy_eV, "normalization": "reciprocal_space"}

    def build(self) -> Waves:
        return Waves(
            self.probe(),
            energy_eV=self._energy_eV,
            sampling=self.sampling,
            reciprocal_space=False,
            metadata=self.metadata,
        )
