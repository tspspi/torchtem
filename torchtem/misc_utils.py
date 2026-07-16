from __future__ import annotations

import copy
import math
from typing import Any, Optional, Sequence, TypeVar

import numpy as np

T = TypeVar("T", float, int, bool)


def number_to_tuple(
    value: T | tuple[T, ...], dimension: Optional[int] = None
) -> tuple[T, ...]:
    if isinstance(value, (float, int, bool)):
        if dimension is None:
            return (value,)
        return (value,) * dimension
    if dimension is not None and len(value) != dimension:
        raise ValueError(f"Expected tuple of length {dimension}, got {value}")
    return value


def itemset(arr: np.ndarray, args: int | slice | Sequence[int], item: Any) -> None:
    if arr.shape == ():
        arr[...] = item
        return
    if isinstance(args, tuple):
        arr[args] = item
        return
    if isinstance(args, int) and len(arr.shape) == 1:
        arr[args] = item
        return
    if isinstance(args, int):
        if not all(n == 1 for n in arr.shape[1:]):
            raise RuntimeError("Integer indexing requires trailing singleton axes")
        arr[(args,) + (0,) * (len(arr.shape) - 1)] = item
        return
    raise RuntimeError("Unsupported itemset arguments")


def safe_equality(a: Any, b: Any, exclude: tuple[str, ...] = ()) -> bool:
    if not isinstance(b, a.__class__):
        return False

    for key, value in a.__dict__.items():
        if key in exclude:
            continue
        if key not in b.__dict__:
            return False

        other_value = b.__dict__[key]
        if isinstance(value, EqualityMixin):
            equal = safe_equality(value, other_value)
        else:
            try:
                equal = bool(np.allclose(value, other_value))
            except (TypeError, ValueError):
                equal = value == other_value

        if not equal:
            return False
    return True


class CopyMixin:
    def copy(self):
        return copy.deepcopy(self)


class EqualityMixin:
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return safe_equality(self, other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)


def safe_floor_int(n: float, tol: int = 7) -> int:
    return int(np.floor(np.round(n, decimals=tol)))


def safe_ceiling_int(n: float, tol: int = 7) -> int:
    return int(np.ceil(np.round(n, decimals=tol)))


def ensure_list(x: Any) -> list[Any]:
    return [x] if not isinstance(x, list) else x


def normalize_axes(
    axes: tuple[int, ...] | int, shape: tuple[int, ...]
) -> tuple[int, ...]:
    if isinstance(axes, int):
        axes = (axes,)
    normalized = []
    ndim = len(shape)
    for axis in axes:
        axis = int(axis)
        if axis < 0:
            axis += ndim
        if axis < 0 or axis >= ndim:
            raise ValueError(f"Axis {axis} out of bounds for shape {shape}")
        normalized.append(axis)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Duplicate axes are not allowed: {axes}")
    return tuple(normalized)


def is_broadcastable(*shapes: tuple[int, ...]) -> bool | tuple[int, ...]:
    if not shapes:
        return True

    reversed_shapes = [shape[::-1] for shape in shapes]
    result = []
    for dims in zip(*map(lambda s: s + (1,) * (max(map(len, shapes)) - len(s)), reversed_shapes)):
        dim = max(dims)
        if any(d not in (1, dim) for d in dims):
            return False
        result.append(dim)
    return tuple(reversed(result))


def safe_log2_ceil(n: int) -> int:
    if n <= 0:
        raise ValueError("n must be positive")
    return int(math.ceil(math.log2(n)))
