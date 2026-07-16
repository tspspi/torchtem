from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
from typing import SupportsFloat

import torch


def _as_tuple(value, dimension: int):
    if isinstance(value, (tuple, list)):
        if len(value) != dimension:
            raise ValueError(f"Expected length {dimension}, got {value}")
        return tuple(value)
    return (value,) * dimension


@dataclass
class DistributionFromValues:
    values: torch.Tensor
    weights: torch.Tensor
    ensemble_mean: bool = False

    def __post_init__(self) -> None:
        self.values = self.values.to(torch.float64)
        self.weights = self.weights.to(torch.float64)

    def __neg__(self) -> "DistributionFromValues":
        return DistributionFromValues(
            values=-self.values, weights=self.weights.clone(), ensemble_mean=self.ensemble_mean
        )

    @property
    def dimensions(self) -> int:
        return 1

    @property
    def shape(self) -> tuple[int]:
        return (int(self.values.shape[0]),)

    def divide(self, chunks: int) -> list["DistributionFromValues"]:
        if chunks <= 0:
            raise ValueError("chunks must be positive")
        boundaries = torch.linspace(0, len(self.values), chunks + 1, dtype=torch.int64)
        blocks = []
        for i in range(chunks):
            start = int(boundaries[i].item())
            stop = int(boundaries[i + 1].item())
            if stop > start:
                blocks.append(
                    DistributionFromValues(
                        values=self.values[start:stop].clone(),
                        weights=self.weights[start:stop].clone(),
                        ensemble_mean=self.ensemble_mean,
                    )
                )
        return blocks

    def combine(self, other: "DistributionFromValues") -> "MultidimensionalDistribution":
        return MultidimensionalDistribution([self, other])


@dataclass
class MultidimensionalDistribution:
    distributions: list[DistributionFromValues]

    def __post_init__(self) -> None:
        if any(distribution.dimensions != 1 for distribution in self.distributions):
            raise ValueError("Only one-dimensional component distributions are supported.")

    def __neg__(self) -> "MultidimensionalDistribution":
        return MultidimensionalDistribution([-distribution for distribution in self.distributions])

    @property
    def dimensions(self) -> int:
        return len(self.distributions)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(distribution.values.shape[0]) for distribution in self.distributions)

    @property
    def values(self) -> torch.Tensor:
        if self.dimensions == 1:
            return self.distributions[0].values
        meshes = torch.meshgrid(
            *[distribution.values for distribution in self.distributions], indexing="ij"
        )
        return torch.stack(meshes, dim=-1)

    @property
    def weights(self) -> torch.Tensor:
        weights = self.distributions[0].weights
        for distribution in self.distributions[1:]:
            weights = torch.outer(weights.reshape(-1), distribution.weights).reshape(
                weights.shape + distribution.weights.shape
            )
        return weights

    @property
    def ensemble_mean(self) -> bool:
        flags = [distribution.ensemble_mean for distribution in self.distributions]
        if not all(flag == flags[0] for flag in flags):
            raise ValueError("All component distributions must share the same ensemble_mean.")
        return flags[0]


def from_values(
    values,
    weights: torch.Tensor | None = None,
    ensemble_mean: bool = False,
) -> DistributionFromValues:
    values = torch.as_tensor(values, dtype=torch.float64)
    if weights is None:
        weights = torch.ones(values.shape[0], dtype=torch.float64)
    else:
        weights = torch.as_tensor(weights, dtype=torch.float64)
    return DistributionFromValues(values=values, weights=weights, ensemble_mean=ensemble_mean)


def uniform(
    low: float,
    high: float,
    num_samples: int,
    endpoint: bool = True,
    ensemble_mean: bool = False,
) -> DistributionFromValues:
    values = torch.linspace(low, high, num_samples, dtype=torch.float64)
    if not endpoint and num_samples > 1:
        step = (high - low) / num_samples
        values = torch.linspace(low, high - step, num_samples, dtype=torch.float64)
    return from_values(values, ensemble_mean=ensemble_mean)


def gaussian(
    standard_deviation: float | tuple[float, ...],
    num_samples: int | tuple[int, ...],
    *,
    dimension: int = 1,
    center: float | tuple[float, ...] = 0.0,
    ensemble_mean: bool | tuple[bool, ...] = True,
    sampling_limit: float | tuple[float, ...] = 3.0,
    normalize: str = "intensity",
) -> MultidimensionalDistribution:
    center = _as_tuple(center, dimension)
    standard_deviation = _as_tuple(standard_deviation, dimension)
    ensemble_mean = _as_tuple(ensemble_mean, dimension)
    sampling_limit = _as_tuple(sampling_limit, dimension)
    num_samples = _as_tuple(num_samples, dimension)

    distributions = []
    for i in range(dimension):
        values = torch.linspace(
            center[i] - standard_deviation[i] * sampling_limit[i],
            center[i] + standard_deviation[i] * sampling_limit[i],
            int(num_samples[i]),
            dtype=torch.float64,
        )
        weights = torch.exp(-0.5 * ((values - center[i]) / standard_deviation[i]) ** 2)
        if normalize == "intensity":
            weights = weights / torch.sqrt((weights.square()).sum())
        elif normalize == "amplitude":
            weights = weights / weights.sum()
        else:
            raise RuntimeError(f"Unknown normalization method: {normalize}")
        distributions.append(
            DistributionFromValues(
                values=values,
                weights=weights,
                ensemble_mean=bool(ensemble_mean[i]),
            )
        )
    return MultidimensionalDistribution(distributions)


def validate_distribution(
    distribution: DistributionFromValues | MultidimensionalDistribution | list | tuple | torch.Tensor | SupportsFloat,
):
    if isinstance(distribution, (DistributionFromValues, MultidimensionalDistribution, Number, str)):
        return distribution
    if isinstance(distribution, torch.Tensor) and distribution.ndim == 0:
        return distribution.item()
    if isinstance(distribution, (list, tuple, torch.Tensor)):
        values = torch.as_tensor(distribution, dtype=torch.float64)
        if values.ndim == 0:
            return values.item()
        return from_values(values)
    raise ValueError(f"value {distribution} is not a single number or valid distribution")
