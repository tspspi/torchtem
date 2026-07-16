from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any

import torch

from tspi.torchtem.plotting import plot_detector_outputs, plot_series_outputs


def _to_cpu_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {name: _to_cpu_value(item) for name, item in value.items()}
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


@dataclass
class SimulationResult:
    outputs: torch.Tensor | Mapping[str, torch.Tensor]
    detector_names: tuple[str, ...] | None = None
    exit_wave: torch.Tensor | None = None
    intermediates: dict[str, torch.Tensor] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_mapping(self) -> bool:
        return isinstance(self.outputs, Mapping)

    def to_cpu(self) -> "SimulationResult":
        outputs = _to_cpu_value(self.outputs)
        intermediates = None if self.intermediates is None else _to_cpu_value(self.intermediates)
        return SimulationResult(
            outputs=outputs,
            detector_names=self.detector_names,
            exit_wave=None if self.exit_wave is None else _to_cpu_value(self.exit_wave),
            intermediates=intermediates,
            metadata=dict(self.metadata),
        )

    def named_outputs(self) -> Mapping[str, torch.Tensor]:
        if isinstance(self.outputs, Mapping):
            return self.outputs
        name = self.detector_names[0] if self.detector_names else "detector"
        return {name: self.outputs}

    def output_shapes(self) -> dict[str, tuple[int, ...]]:
        if isinstance(self.outputs, Mapping):
            return {name: tuple(value.shape) for name, value in self.outputs.items()}
        name = self.detector_names[0] if self.detector_names else "detector"
        return {name: tuple(self.outputs.shape)}

    def plot(
        self,
        *,
        cmap: str = "magma",
        figsize: tuple[float, float] | None = None,
    ):
        return plot_detector_outputs(
            self.outputs,
            detector_names=self.detector_names,
            cmap=cmap,
            figsize=figsize,
        )


@dataclass
class SimulationSeriesResult:
    outputs: torch.Tensor | Mapping[str, torch.Tensor]
    parameter_name: str
    parameter_values: torch.Tensor
    detector_names: tuple[str, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_mapping(self) -> bool:
        return isinstance(self.outputs, Mapping)

    def to_cpu(self) -> "SimulationSeriesResult":
        return SimulationSeriesResult(
            outputs=_to_cpu_value(self.outputs),
            parameter_name=self.parameter_name,
            parameter_values=self.parameter_values.detach().cpu(),
            detector_names=self.detector_names,
            metadata=dict(self.metadata),
        )

    def named_outputs(self) -> Mapping[str, torch.Tensor]:
        if isinstance(self.outputs, Mapping):
            return self.outputs
        name = self.detector_names[0] if self.detector_names else "detector"
        return {name: self.outputs}

    def output_shapes(self) -> dict[str, tuple[int, ...]]:
        if isinstance(self.outputs, Mapping):
            return {name: tuple(value.shape) for name, value in self.outputs.items()}
        name = self.detector_names[0] if self.detector_names else "detector"
        return {name: tuple(self.outputs.shape)}

    def at(self, index: int) -> SimulationResult:
        if isinstance(self.outputs, Mapping):
            outputs = {name: value[index] for name, value in self.outputs.items()}
        else:
            outputs = self.outputs[index]
        metadata = dict(self.metadata)
        metadata["series_parameter_name"] = self.parameter_name
        metadata["series_parameter_value"] = self.parameter_values[index]
        return SimulationResult(
            outputs=outputs,
            detector_names=self.detector_names,
            intermediates=None,
            exit_wave=None,
            metadata=metadata,
        )

    def plot(
        self,
        index: int = 0,
        *,
        cmap: str = "magma",
        figsize: tuple[float, float] | None = None,
    ):
        return self.at(index).plot(cmap=cmap, figsize=figsize)

    def plot_series(
        self,
        *,
        selector_index: int = 0,
        cmap: str = "magma",
        figsize: tuple[float, float] | None = None,
    ):
        return plot_series_outputs(
            self.outputs,
            parameter_name=self.parameter_name,
            parameter_values=self.parameter_values,
            detector_names=self.detector_names,
            selector_index=selector_index,
            cmap=cmap,
            figsize=figsize,
        )
