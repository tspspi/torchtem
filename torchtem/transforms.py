from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from torch import nn

from torchtem.distributions import DistributionFromValues
from torchtem.fft_support import fft2, ifft2


def join_tuples(tuples: tuple[tuple[Any, ...], ...]) -> tuple[Any, ...]:
    return tuple(item for subtuple in tuples for item in subtuple)


@dataclass
class ParameterAxis:
    values: tuple[Any, ...]
    label: str | None = None
    ensemble_mean: bool = False


class TensorTransform(nn.Module, ABC):
    """Minimal tensor-native transform analogue of abTEM's transform layer."""

    @property
    def metadata(self) -> dict[str, Any]:
        return {}

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def ensemble_axes_metadata(self) -> list[ParameterAxis]:
        return []

    @abstractmethod
    def apply(self, array: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        raise NotImplementedError

    def forward(self, array: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        return self.apply(array)


class EmptyTransform(TensorTransform):
    def apply(self, array: torch.Tensor) -> torch.Tensor:
        return array


class EnsembleTransform(TensorTransform):
    def __init__(self, distributions: tuple[str, ...] = ()) -> None:
        super().__init__()
        self._distributions = distributions

    @property
    def distributions(self) -> tuple[str, ...]:
        return self._distributions

    def get_axes_metadata_from_distributions(
        self, **kwargs: dict[str, dict[str, Any]]
    ) -> list[ParameterAxis]:
        axes: list[ParameterAxis] = []
        for name, meta in kwargs.items():
            if name not in self._distributions:
                continue
            distribution = getattr(self, name)
            if isinstance(distribution, DistributionFromValues):
                axes.append(
                    ParameterAxis(
                        values=tuple(distribution.values.reshape(-1).tolist()),
                        label=meta.get("label"),
                        ensemble_mean=bool(distribution.ensemble_mean),
                    )
                )
        return axes


class TransformFromFunc(TensorTransform):
    def __init__(
        self, func: Callable[..., torch.Tensor], func_kwargs: dict[str, Any] | None = None
    ) -> None:
        super().__init__()
        self._func = func
        self._func_kwargs = {} if func_kwargs is None else dict(func_kwargs)

    @property
    def func(self) -> Callable[..., torch.Tensor]:
        return self._func

    @property
    def func_kwargs(self) -> dict[str, Any]:
        return self._func_kwargs

    def apply(self, array: torch.Tensor) -> torch.Tensor:
        return self.func(array, **self.func_kwargs)


class ReciprocalSpaceMultiplication(TensorTransform):
    """Multiply a wave tensor by a reciprocal-space kernel."""

    def __init__(self, *, in_place: bool = False) -> None:
        super().__init__()
        self.in_place = bool(in_place)

    @abstractmethod
    def evaluate_kernel(self, wave: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def apply(self, wave: torch.Tensor) -> torch.Tensor:
        reciprocal_wave = fft2(wave, norm="ortho")
        kernel = self.evaluate_kernel(reciprocal_wave).to(reciprocal_wave.dtype)
        while kernel.ndim < reciprocal_wave.ndim:
            kernel = kernel.unsqueeze(0)

        if self.in_place:
            reciprocal_wave = reciprocal_wave.clone()
            reciprocal_wave *= kernel
            transformed = reciprocal_wave
        else:
            transformed = reciprocal_wave * kernel
        return ifft2(transformed, norm="ortho")


class KernelTransform(ReciprocalSpaceMultiplication):
    """Reciprocal-space multiplication using a fixed or callable kernel."""

    def __init__(
        self,
        kernel: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
        *,
        in_place: bool = False,
    ) -> None:
        super().__init__(in_place=in_place)
        self._kernel = kernel

    def evaluate_kernel(self, wave: torch.Tensor) -> torch.Tensor:
        if callable(self._kernel):
            return self._kernel(wave)
        return self._kernel


class SequentialTransform(TensorTransform):
    """Compose multiple tensor transforms sequentially."""

    def __init__(self, transforms: Sequence[TensorTransform]) -> None:
        super().__init__()
        self.transforms = nn.ModuleList(transforms)

    @property
    def metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for transform in self.transforms:
            metadata.update(transform.metadata)
        return metadata

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        return join_tuples(tuple(transform.ensemble_shape for transform in self.transforms))

    @property
    def ensemble_axes_metadata(self) -> list[ParameterAxis]:
        axes: list[ParameterAxis] = []
        for transform in self.transforms:
            axes.extend(transform.ensemble_axes_metadata)
        return axes

    def apply(self, array: torch.Tensor) -> torch.Tensor:
        out = array
        for transform in self.transforms:
            result = transform(out)
            if isinstance(result, list):
                raise TypeError("SequentialTransform does not support list-valued transforms")
            out = result
        return out
