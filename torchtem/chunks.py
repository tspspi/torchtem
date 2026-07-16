from __future__ import annotations

import itertools
from itertools import accumulate
from typing import Optional, TypeGuard, Union

import numpy as np

Chunks = Union[int, str, tuple[Union[int, str, tuple[int, ...]], ...]]
ValidatedChunks = tuple[tuple[int, ...], ...]


def is_tuple_of_ints(x: Chunks) -> TypeGuard[tuple[int, ...]]:
    return isinstance(x, tuple) and all(isinstance(c, int) for c in x)


def is_tuple_of_tuple_of_ints(x: Chunks) -> TypeGuard[ValidatedChunks]:
    return isinstance(x, tuple) and all(
        isinstance(item, tuple) and all(isinstance(c, int) for c in item) for item in x
    )


def is_validated_chunks(x: Chunks) -> TypeGuard[ValidatedChunks]:
    return is_tuple_of_tuple_of_ints(x)


def assert_chunks_match_shape(shape: tuple[int, ...], chunks: ValidatedChunks) -> None:
    if len(shape) != len(chunks):
        raise ValueError(f"Shape {shape} and chunks {chunks} must have same rank")
    if not all(sum(c) == s for s, c in zip(shape, chunks)):
        raise ValueError(f"Chunks {chunks} do not match shape {shape}")


def chunk_ranges(chunks: ValidatedChunks) -> tuple[tuple[tuple[int, int], ...], ...]:
    return tuple(
        tuple((end - width, end) for width, end in zip(axis_chunks, accumulate(axis_chunks)))
        for axis_chunks in chunks
    )


def iterate_chunk_ranges(chunks: ValidatedChunks):
    for block_indices, axis_ranges in zip(
        itertools.product(*(range(len(axis)) for axis in chunks)),
        itertools.product(*chunk_ranges(chunks)),
    ):
        yield block_indices, tuple(slice(start, end) for start, end in axis_ranges)


def fill_in_chunk_sizes(
    shape: tuple[int, ...], chunks: tuple[int | tuple[int, ...], ...]
) -> ValidatedChunks:
    validated: list[tuple[int, ...]] = []
    for size, chunk in zip(shape, chunks):
        if isinstance(chunk, tuple):
            validated.append(tuple(int(c) for c in chunk))
            continue
        if chunk == -1:
            validated.append((int(size),))
            continue
        repeats = size // int(chunk)
        axis_chunks = (int(chunk),) * repeats
        remainder = size % int(chunk)
        if remainder:
            axis_chunks += (remainder,)
        validated.append(axis_chunks)
    return tuple(validated)


def check_chunks_match_shape_length(shape: tuple[int, ...], chunks: Chunks) -> None:
    if isinstance(chunks, tuple) and len(shape) != len(chunks):
        raise ValueError(f"Shape {shape} and chunks {chunks} must have same rank")


def _auto_chunks(
    shape: tuple[int, ...],
    chunks: tuple[int | str | tuple[int, ...], ...],
    max_elements: str | int = "auto",
    dtype: Optional[np.dtype] = None,
) -> ValidatedChunks:
    if max_elements == "auto":
        if dtype is None:
            max_elements = 262144
        else:
            max_bytes = 8 * 1024 * 1024
            max_elements = max(1, max_bytes // np.dtype(dtype).itemsize)

    fixed_sizes = []
    auto_axes = []
    for i, (size, chunk) in enumerate(zip(shape, chunks)):
        if isinstance(chunk, tuple):
            fixed_sizes.append(sum(chunk))
        elif isinstance(chunk, int) and chunk > 0:
            fixed_sizes.append(chunk)
        else:
            fixed_sizes.append(1)
            auto_axes.append(i)

    fixed_product = int(np.prod(fixed_sizes)) if fixed_sizes else 1
    remaining = max(1, int(max_elements) // max(1, fixed_product))
    axis_auto = int(max(1, round(remaining ** (1 / max(1, len(auto_axes))))))

    resolved: list[int | tuple[int, ...]] = []
    for i, size in enumerate(shape):
        chunk = chunks[i]
        if isinstance(chunk, tuple):
            resolved.append(chunk)
        elif isinstance(chunk, int) and chunk > 0:
            resolved.append(chunk)
        else:
            resolved.append(min(size, axis_auto))

    return fill_in_chunk_sizes(shape, tuple(resolved))


def validate_chunks(
    shape: tuple[int, ...],
    chunks: Chunks,
    max_elements: int | str = "auto",
    dtype: Optional[np.dtype] = None,
) -> ValidatedChunks:
    check_chunks_match_shape_length(shape, chunks)

    if is_validated_chunks(chunks):
        validated = chunks
    elif chunks == -1:
        validated = tuple((int(s),) for s in shape)
    elif isinstance(chunks, int):
        validated = _auto_chunks(shape, ("auto",) * len(shape), max_elements=chunks, dtype=dtype)
    elif isinstance(chunks, str):
        if chunks != "auto":
            raise ValueError(f"Unsupported chunk specifier: {chunks}")
        validated = _auto_chunks(shape, ("auto",) * len(shape), max_elements=max_elements, dtype=dtype)
    elif isinstance(chunks, tuple) and any(isinstance(c, str) for c in chunks):
        validated = _auto_chunks(shape, chunks, max_elements=max_elements, dtype=dtype)
    elif isinstance(chunks, tuple):
        validated = fill_in_chunk_sizes(shape, chunks)
    else:
        raise ValueError(f"Unsupported chunks specification: {chunks}")

    assert_chunks_match_shape(shape, validated)
    return validated
