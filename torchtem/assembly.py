from __future__ import annotations

from collections import OrderedDict
from typing import Any

import torch
from torch import nn


def _validate_num_slices(num_slices: int) -> int:
    value = int(num_slices)
    if value < 1:
        raise ValueError("num_slices must be at least 1")
    return value


def apply_detector_recursive(detector: nn.Module, waves: torch.Tensor) -> torch.Tensor:
    if waves.ndim < 2:
        raise ValueError("waves must have at least 2 dimensions")
    if waves.ndim == 2:
        return detector(waves)

    outputs = []
    for index in range(waves.shape[0]):
        outputs.append(apply_detector_recursive(detector, waves[index]))
    return torch.stack(outputs, dim = 0)


class PotentialSliceInteraction(nn.Module):
    """Unary interaction layer binding a potential-producing module to a multislice propagator."""

    def __init__(
        self,
        *,
        multislice: nn.Module,
        potential: nn.Module,
        num_slices: int = 1,
    ) -> None:
        super().__init__()
        self.multislice = multislice
        self.potential = potential
        self.num_slices = _validate_num_slices(num_slices)

    def potential_slices(self) -> torch.Tensor:
        potential = self.potential()
        if potential.ndim == 3:
            return potential
        return potential.unsqueeze(0).repeat(self.num_slices, 1, 1) / self.num_slices

    def forward(self, incident_wave: torch.Tensor) -> torch.Tensor:
        return self.multislice(incident_wave, self.potential_slices())


class DetectorReadout(nn.Module):
    """Unary wrapper for detector/readout modules."""

    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        self.detector = detector

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        return apply_detector_recursive(self.detector, wave)


class LayerStack(nn.Module):
    """Ordered differentiable stack for source, interaction, and readout layers."""

    def __init__(
        self,
        *,
        source: nn.Module,
        layers: list[tuple[str, nn.Module]] | None = None,
    ) -> None:
        super().__init__()
        self.source = source
        self.layers = nn.ModuleDict(OrderedDict(layers or []))

    def append(self, name: str, layer: nn.Module) -> None:
        if not name:
            raise ValueError("layer name must not be empty")
        self.layers[name] = layer

    def parameter_tensor_dict(self) -> dict[str, torch.Tensor]:
        parameters: dict[str, torch.Tensor] = {}
        for name, param in self.named_parameters():
            parameters[name] = param
        return parameters

    def set_parameter(self, name: str, value: torch.Tensor | float | complex) -> None:
        module_name, _, param_name = name.rpartition(".")
        target = self.get_submodule(module_name) if module_name else self
        if not hasattr(target, param_name):
            raise AttributeError(f"unknown parameter '{name}'")

        parameter = getattr(target, param_name)
        if not isinstance(parameter, torch.Tensor):
            raise TypeError(f"attribute '{name}' is not a tensor parameter")

        with torch.no_grad():
            parameter.copy_(torch.as_tensor(value, device=parameter.device, dtype=parameter.dtype))

    def forward(
        self,
        *,
        return_intermediates: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        current = self.source()
        if not return_intermediates:
            for layer in self.layers.values():
                current = layer(current)
            return current

        states: dict[str, torch.Tensor] = {"source": current}
        for name, layer in self.layers.items():
            current = layer(current)
            if name != "source_tilt":
                states[name] = current
        return states


class ScanLayer(nn.Module):
    """Apply a scan-position generator and shift a reference wave for each scan point."""

    def __init__(
        self,
        *,
        scan: nn.Module,
        shift_wave,
        sampling: tuple[float, float],
    ) -> None:
        super().__init__()
        self.scan = scan
        self.shift_wave = shift_wave
        self.sampling = sampling

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        positions = self.scan()
        if positions.ndim != 2:
            raise ValueError("scan positions must have shape (count, 2)")
        if positions.shape[1] != 2:
            raise ValueError("scan positions must contain x/y offsets in Angstrom")

        shifted = self.shift_wave(wave, positions, self.sampling)
        if shifted.ndim == wave.ndim:
            return shifted.unsqueeze(0)
        return shifted


class BatchDetectorReadout(nn.Module):
    """Apply a detector independently across a leading batch dimension."""

    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        self.detector = detector

    def forward(self, waves: torch.Tensor) -> torch.Tensor:
        return apply_detector_recursive(self.detector, waves)
