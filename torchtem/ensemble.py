from __future__ import annotations

import itertools
from abc import abstractmethod
from itertools import accumulate
from typing import Any, Callable, Generator, Optional

import numpy as np

from tspi.torchtem.axes import AxesMetadataList, AxisMetadata
from tspi.torchtem.chunks import Chunks, ValidatedChunks, chunk_ranges, validate_chunks
from tspi.torchtem.misc_utils import itemset


def interleave(args1, args2):
    output = []
    for a, b in zip(args1, args2):
        output.extend((a, b))
    return tuple(output)


def _wrap_with_array(x: Any, ndims: int | None = None) -> np.ndarray:
    if ndims is None:
        ndims = len(x.ensemble_shape)
    wrapped = np.zeros((1,) * ndims, dtype=object)
    itemset(wrapped, 0, x)
    return wrapped


def unpack_blockwise_args(args) -> tuple:
    return tuple(arg.item() if hasattr(arg, "item") else arg for arg in args)


class Ensemble:
    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def base_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def shape(self) -> tuple[int, ...]:
        return self.ensemble_shape + self.base_shape

    @property
    def base_axes_metadata(self) -> list[AxisMetadata]:
        return []

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        return []

    @property
    def axes_metadata(self) -> AxesMetadataList:
        return AxesMetadataList(self.ensemble_axes_metadata + self.base_axes_metadata, self.shape)

    @property
    @abstractmethod
    def _default_ensemble_chunks(self) -> Chunks:
        raise NotImplementedError

    def _validate_ensemble_chunks(
        self, chunks: Optional[Chunks] = None, limit: str | int = "auto"
    ) -> ValidatedChunks:
        if chunks is None:
            chunks = self._default_ensemble_chunks
        return validate_chunks(self.ensemble_shape, chunks, max_elements=limit)

    @abstractmethod
    def _partition_args(
        self, chunks: Optional[Chunks] = None, lazy: bool = True
    ) -> tuple:
        raise NotImplementedError

    @abstractmethod
    def _from_partitioned_args(self) -> Callable[..., np.ndarray]:
        raise NotImplementedError

    def ensemble_blocks(self, chunks: Optional[Chunks] = None) -> np.ndarray:
        chunks = self._validate_ensemble_chunks(chunks)
        args = self._partition_args(chunks, lazy=False)
        arg_dims = tuple(len(arg.shape) for arg in args)
        shape = tuple(len(axis_chunks) for axis_chunks in chunks)
        blocks = np.empty(shape if shape else (), dtype=object)

        for indices, slics, block in self.generate_blocks(chunks):
            if shape:
                blocks[indices] = block
            else:
                blocks = np.asarray(block, dtype=object)
        return blocks

    def generate_blocks(
        self, chunks: Chunks = 1
    ) -> Generator[tuple[tuple[int, ...], tuple[slice, ...], np.ndarray], None, None]:
        chunks = self._validate_ensemble_chunks(chunks)
        blocks = self._partition_args(chunks=chunks, lazy=False)
        shape = sum((block.shape for block in blocks), ())
        start_stops = chunk_ranges(chunks)
        assert tuple(len(cr) for cr in start_stops) == shape

        for indices, start_stop in zip(np.ndindex(shape), itertools.product(*start_stops)):
            block_indices: tuple[tuple[int, ...], ...] = ()
            j = 0
            for block in blocks:
                n = len(block.shape)
                block_indices += (tuple(indices[index] for index in range(j, j + n)),)
                j += n
            args = tuple(block[i] for i, block in zip(block_indices, blocks))
            slics = tuple(slice(start, stop) for start, stop in start_stop)
            yield indices, slics, self._from_partitioned_args()(*args)


class EmptyEnsemble(Ensemble):
    @property
    def _default_ensemble_chunks(self) -> Chunks:
        return ()

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        return []

    def _partition_args(
        self, chunks: Optional[Chunks] = None, lazy: bool = True
    ) -> tuple:
        return ()

    def _from_partitioned_args(self) -> type:
        return self.__class__

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        return ()


def concatenate_array_blocks(blocks: np.ndarray) -> np.ndarray:
    def _concatenate_items(items, axis: int):
        entries = tuple(items.tolist()) if hasattr(items, "tolist") else tuple(items)
        if len(entries) == 0:
            return np.array([])
        if not any(isinstance(entry, np.ndarray) for entry in entries):
            return np.asarray(entries)
        normalized = tuple(
            entry if isinstance(entry, np.ndarray) else np.asarray([entry]) for entry in entries
        )
        return np.concatenate(normalized, axis=axis)

    if blocks.ndim == 1:
        return _concatenate_items(blocks, axis=0)

    current = blocks
    while current.ndim > 1:
        new_blocks = np.empty(current.shape[:-1], dtype=object)
        for indices in np.ndindex(current.shape[:-1]):
            new_blocks[indices] = _concatenate_items(current[indices], axis=len(indices))
        current = new_blocks

    return _concatenate_items(current, axis=0)
