from __future__ import annotations

import math
from collections.abc import Iterable as IterableABC
from typing import Callable, Optional, Sequence, TypeVar

import numpy as np

from tspi.torchtem.misc_utils import CopyMixin, EqualityMixin

T = TypeVar("T", int, float)
U = TypeVar("U")


def validate_gpts(gpts: tuple[int, ...]) -> tuple[int, ...]:
    gpts = tuple(int(n) for n in gpts)
    if not all(n > 0 for n in gpts):
        raise ValueError("gpts must be greater than 0")
    return gpts


def adjusted_gpts(
    target_sampling: tuple[float, ...],
    old_sampling: tuple[float, ...],
    old_gpts: tuple[int, ...],
) -> tuple[tuple[float, ...], tuple[int, ...]]:
    new_gpts = tuple(
        int(math.ceil(n * (d / d_target)))
        for d_target, d, n in zip(target_sampling, old_sampling, old_gpts)
    )
    new_sampling = tuple(
        d * n / new_n for d, n, new_n in zip(old_sampling, old_gpts, new_gpts)
    )
    return new_sampling, new_gpts


class GridUndefinedError(Exception):
    pass


class Grid(CopyMixin, EqualityMixin):
    def __init__(
        self,
        extent: Optional[float | Sequence[float]] = None,
        gpts: Optional[int | Sequence[int]] = None,
        sampling: Optional[float | Sequence[float]] = None,
        dimensions: int = 2,
        endpoint: bool | Sequence[bool] = False,
        lock_extent: bool = False,
        lock_gpts: bool = False,
        lock_sampling: bool = False,
    ) -> None:
        self._dimensions = int(dimensions)
        if isinstance(endpoint, bool):
            endpoint = (endpoint,) * self._dimensions
        self._endpoint = tuple(bool(e) for e in endpoint)

        self._extent = self._validate(extent, float)
        self._gpts = self._validate(gpts, int)
        self._sampling = self._validate(sampling, float)

        self._lock_extent = bool(lock_extent)
        self._lock_gpts = bool(lock_gpts)
        self._lock_sampling = bool(lock_sampling)

        if self.extent is None:
            self._adjust_extent(self.gpts, self.sampling)
        if self.gpts is None:
            self._adjust_gpts(self.extent, self.sampling)
        if sampling is None or extent is not None:
            self._adjust_sampling(self.extent, self.gpts)

    def _validate(
        self, value: Optional[T | Sequence[T]], dtype: Callable[[T], U]
    ) -> Optional[tuple[U, ...]]:
        if isinstance(value, (np.ndarray, list, tuple)):
            if len(value) != self.dimensions:
                raise RuntimeError(
                    f"Grid value length {len(value)} != {self._dimensions}"
                )
            return tuple(dtype(v) for v in value)
        if isinstance(value, (int, float)):
            return (dtype(value),) * self.dimensions
        if value is None:
            return None
        raise RuntimeError(f"Invalid grid property {value}")

    def __len__(self) -> int:
        return self.dimensions

    @property
    def endpoint(self) -> tuple[bool, ...]:
        return self._endpoint

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def extent(self) -> tuple[float, ...] | None:
        return self._extent

    @extent.setter
    def extent(self, extent: float | Sequence[float] | None) -> None:
        if extent is None:
            self._extent = None
            return
        if self._lock_extent and self.extent is not None and not np.allclose(extent, self.extent):
            raise RuntimeError("Extent cannot be modified")

        validated = self._validate(extent, float)
        if self._lock_sampling or self.gpts is None:
            self._adjust_gpts(validated, self.sampling)
            self._adjust_sampling(validated, self.gpts)
        elif self.gpts is not None:
            self._adjust_sampling(validated, self.gpts)
        self._extent = validated

    @property
    def gpts(self) -> tuple[int, ...] | None:
        return self._gpts

    @gpts.setter
    def gpts(self, gpts: int | Sequence[int]) -> None:
        if self._lock_gpts:
            raise RuntimeError("Grid gpts cannot be modified")
        validated = self._validate(gpts, int)
        if self._lock_sampling:
            self._adjust_extent(validated, self.sampling)
        elif self.extent is not None:
            self._adjust_sampling(self.extent, validated)
        else:
            self._adjust_extent(validated, self.sampling)
        self._gpts = validated

    @property
    def sampling(self) -> tuple[float, ...] | None:
        return self._sampling

    @sampling.setter
    def sampling(self, sampling: float | Sequence[float]) -> None:
        if self._lock_sampling:
            raise RuntimeError("Sampling cannot be modified")
        validated = self._validate(sampling, float)
        if self._lock_gpts:
            self._adjust_extent(self.gpts, validated)
        elif self.extent is not None:
            self._adjust_gpts(self.extent, validated)
        else:
            self._adjust_extent(self.gpts, validated)

        if self.extent is None or self.gpts is None:
            self._sampling = validated
        else:
            self._adjust_sampling(self.extent, self.gpts)

    @property
    def reciprocal_space_sampling(self) -> tuple[float, ...]:
        self.check_is_defined()
        assert self.sampling is not None and self.gpts is not None
        return tuple(1.0 / (n * d) for n, d in zip(self.gpts, self.sampling))

    def _adjust_extent(
        self, gpts: tuple[int, ...] | None, sampling: tuple[float, ...] | None
    ) -> None:
        if gpts is not None and sampling is not None:
            self._extent = tuple(
                (n - 1) * d if endpoint else n * d
                for n, d, endpoint in zip(gpts, sampling, self._endpoint)
            )
            self._extent = self._validate(self._extent, float)

    def _adjust_gpts(
        self, extent: tuple[float, ...] | None, sampling: tuple[float, ...] | None
    ) -> None:
        if extent is not None and sampling is not None:
            self._gpts = tuple(
                int(math.ceil(r / d)) + 1 if endpoint else int(math.ceil(r / d))
                for r, d, endpoint in zip(extent, sampling, self._endpoint)
            )

    def _adjust_sampling(
        self, extent: tuple[float, ...] | None, gpts: tuple[int, ...] | None
    ) -> None:
        if extent is not None and gpts is not None:
            def safe_divide(a: float, b: float) -> float:
                return 0.0 if b == 0 else a / b

            self._sampling = tuple(
                safe_divide(r, n - 1) if endpoint else safe_divide(r, n)
                for r, n, endpoint in zip(extent, gpts, self._endpoint)
            )
            self._sampling = self._validate(self._sampling, float)

    def check_is_defined(self, raise_error: bool = True) -> bool:
        is_defined = self.extent is not None and self.gpts is not None
        if raise_error and not is_defined:
            raise GridUndefinedError("grid is not defined")
        return is_defined

    def match(self, other: Grid | HasGrid2DMixin, check_match: bool = False) -> None:
        if check_match:
            self.check_match(other)
        if other.extent is None:
            other.extent = self.extent
        elif self.extent is not None and not np.allclose(self.extent, other.extent):
            self.extent = other.extent

        if other.gpts is None:
            other.gpts = self.gpts
        elif self.gpts is not None and tuple(self.gpts) != tuple(other.gpts):
            self.gpts = other.gpts

        if other.sampling is None:
            other.sampling = self.sampling
        elif self.sampling is not None and not np.allclose(self.sampling, other.sampling):
            self.sampling = other.sampling

    def check_match(self, other: Grid | HasGrid2DMixin) -> None:
        if self.extent is not None and other.extent is not None:
            if not np.allclose(self.extent, other.extent):
                raise RuntimeError(
                    f"Inconsistent grid extent ({self.extent} != {other.extent})"
                )
        if self.gpts is not None and other.gpts is not None:
            if tuple(self.gpts) != tuple(other.gpts):
                raise RuntimeError(
                    f"Inconsistent grid gpts ({self.gpts} != {other.gpts})"
                )

    def round_to_power(self, powers: Optional[int | list[int]] = None) -> tuple[int, ...]:
        if powers is None:
            powers = [2, 3, 5, 7]
        elif not isinstance(powers, IterableABC):
            powers = [int(powers)]
        assert self.gpts is not None
        rounded = tuple(
            int(min(power ** math.ceil(math.log(max(1, n), power)) for power in powers))
            for n in self.gpts
        )
        self.gpts = rounded
        return rounded

    @property
    def _valid_extent(self) -> tuple[float, ...]:
        if self.extent is None:
            raise GridUndefinedError("Grid extent is not defined")
        return self.extent

    @property
    def _valid_gpts(self) -> tuple[int, ...]:
        if self.gpts is None:
            raise GridUndefinedError("Grid gpts is not defined")
        return self.gpts

    @property
    def _valid_sampling(self) -> tuple[float, ...]:
        if self.sampling is None:
            raise GridUndefinedError("Grid sampling is not defined")
        return self.sampling


class HasGrid2DMixin:
    _grid: Grid

    def match_grid(self, other: HasGrid2DMixin, check_match: bool = False):
        self.grid.match(other, check_match=check_match)
        return self

    @property
    def grid(self) -> Grid:
        return self._grid

    @property
    def extent(self) -> tuple[float, float] | None:
        extent = self.grid.extent
        if extent is not None:
            assert len(extent) == 2
        return extent

    @extent.setter
    def extent(self, extent: tuple[float, float] | None) -> None:
        self.grid.extent = extent

    @property
    def gpts(self) -> tuple[int, int] | None:
        gpts = self.grid.gpts
        if gpts is not None:
            assert len(gpts) == 2
        return gpts

    @gpts.setter
    def gpts(self, gpts: tuple[int, int]) -> None:
        self.grid.gpts = gpts

    @property
    def sampling(self) -> tuple[float, float] | None:
        sampling = self.grid.sampling
        if sampling is not None:
            assert len(sampling) == 2
        return sampling

    @sampling.setter
    def sampling(self, sampling: tuple[float, float]) -> None:
        self.grid.sampling = sampling
