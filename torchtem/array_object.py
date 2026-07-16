from __future__ import annotations

import copy
import inspect
import json
from abc import ABCMeta
from typing import Any, Optional, Self, TypeVar, Union

import numpy as np
import torch

import tspi.torchtem.config as config
from tspi.torchtem.axes import (
    AxesMetadataList,
    AxisMetadata,
    OrdinalAxis,
    UnknownAxis,
    axis_from_dict,
    axis_to_dict,
)
from tspi.torchtem.backend import copy_to_device, get_array_module
from tspi.torchtem.chunks import Chunks
from tspi.torchtem.ensemble import EmptyEnsemble, Ensemble
from tspi.torchtem.misc_utils import CopyMixin, EqualityMixin, normalize_axes, number_to_tuple

try:
    import dask
    import dask.array as da
except ModuleNotFoundError:
    dask = None
    da = None

ArrayObjectType = TypeVar("ArrayObjectType", bound="ArrayObject")
ArrayItemType = Union[int, slice, list, np.ndarray, None]


class LazyArray:
    def __init__(self, array: np.ndarray, chunks: Chunks = "auto"):
        self._array = np.asarray(array)
        self.chunks = chunks

    @property
    def shape(self):
        return self._array.shape

    def compute(self):
        return self._array

    def __getitem__(self, item):
        return LazyArray(self._array[item], chunks=self.chunks)


def validate_lazy(lazy: Optional[bool]) -> bool:
    if lazy is None:
        return bool(config.get("dask.lazy"))
    if not isinstance(lazy, bool):
        raise ValueError("lazy must be a boolean")
    return lazy


def _validate_array_items(
    items: ArrayItemType | tuple[ArrayItemType, ...],
    shape: tuple[int, ...],
    keepdims: bool = False,
) -> tuple[ArrayItemType, ...]:
    if isinstance(items, (int, slice, type(None), list, np.ndarray)):
        items = (items,)
    elif not isinstance(items, tuple):
        raise NotImplementedError(f"Unsupported index type {type(items).__name__}")

    if keepdims:
        items = tuple(slice(item, item + 1) if isinstance(item, int) else item for item in items)

    if any(item is Ellipsis for item in items):
        raise NotImplementedError("Ellipsis indexing is not supported")

    if len(tuple(item for item in items if item is not None)) > len(shape):
        raise RuntimeError("too many indices for array")

    return items


class ComputableList(list):
    def compute(self, **kwargs):
        arrays = []
        for item in self:
            computed = item.compute(**kwargs)
            arrays.append(computed[0] if isinstance(computed, tuple) else computed)
        return arrays

    def to_zarr(self, url: str, compute: bool = True, overwrite: bool = False, **kwargs):
        import zarr

        arrays = self.compute() if compute else self
        mode = "w" if overwrite else "w-"
        root = zarr.open(url, mode=mode)
        for i, has_array in enumerate(arrays):
            array = has_array.array
            if isinstance(array, torch.Tensor):
                array = array.detach().cpu().numpy()
            elif da is not None and isinstance(array, da.Array):
                array = array.compute()
            elif isinstance(array, LazyArray):
                array = array.compute()
            root.create_array(f"array{i}", data=array, overwrite=overwrite, **kwargs)
            root.attrs[f"metadata{i}"] = has_array._metadata_to_dict()
        return url


class ArrayObject(Ensemble, EqualityMixin, CopyMixin, metaclass=ABCMeta):
    _base_dims: int = 0

    def __init__(
        self,
        array: np.ndarray | da.Array | torch.Tensor,
        ensemble_axes_metadata: list[AxisMetadata] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ):
        if ensemble_axes_metadata is None:
            ensemble_axes_metadata = []
        if metadata is None:
            metadata = {}
        self._array = array
        self._ensemble_axes_metadata = ensemble_axes_metadata
        self._metadata = metadata
        self._check_axes_metadata()
        super().__init__(**kwargs)

    @property
    def base_dims(self) -> int:
        return self._base_dims

    @property
    def ensemble_dims(self) -> int:
        return len(self.shape) - self.base_dims

    @property
    def base_axes_metadata(self) -> list[AxisMetadata]:
        return [UnknownAxis() for _ in range(self._base_dims)]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.array.shape)

    @property
    def base_shape(self) -> tuple[int, ...]:
        return self.shape[self.ensemble_dims :]

    @property
    def ensemble_shape(self) -> tuple[int, ...]:
        return self.shape[: self.ensemble_dims]

    @property
    def ensemble_axes_metadata(self) -> list[AxisMetadata]:
        return self._ensemble_axes_metadata

    @property
    def axes_metadata(self) -> AxesMetadataList:
        return AxesMetadataList(self.ensemble_axes_metadata + self.base_axes_metadata, self.shape)

    @property
    def array(self):
        return self._array

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def is_lazy(self) -> bool:
        is_dask = da is not None and isinstance(self.array, da.Array)
        return is_dask or isinstance(self.array, LazyArray)

    @property
    def device(self) -> str:
        if isinstance(self.array, torch.Tensor):
            return "gpu" if self.array.is_cuda else "cpu"
        return "cpu"

    def _check_axes_metadata(self) -> None:
        if len(self.shape) != len(self.axes_metadata):
            raise RuntimeError("number of array dimensions does not match axis metadata")
        for n, axis in zip(self.shape, self.axes_metadata):
            if isinstance(axis, OrdinalAxis) and len(axis) != n:
                raise RuntimeError("ordinal axis length does not match array dimension")

    def _copy_kwargs(self, exclude: tuple[str, ...] = ()) -> dict[str, Any]:
        parameters = inspect.signature(self.__class__).parameters
        keys = [
            key
            for key, value in parameters.items()
            if value.kind not in (value.VAR_POSITIONAL, value.VAR_KEYWORD) and key not in exclude
        ]
        return {key: copy.deepcopy(getattr(self, key)) for key in keys if hasattr(self, key)}

    def _with_array(self, array) -> Self:
        return self.__class__(array=array, **self._copy_kwargs(exclude=("array",)))

    @property
    def _default_ensemble_chunks(self) -> Chunks:
        return ("auto",) * len(self.ensemble_shape)

    def _partition_args(self, chunks: Optional[Chunks] = None, lazy: bool = True) -> tuple:
        return ()

    def _from_partitioned_args(self):
        return self.__class__

    def ensure_lazy(self, chunks: Chunks = "auto") -> Self:
        if self.is_lazy:
            return self
        if isinstance(self.array, torch.Tensor):
            source = self.array.detach().cpu().numpy()
        else:
            source = self.array
        if chunks == "auto":
            chunks = ("auto",) * len(self.ensemble_shape) + (-1,) * len(self.base_shape)
        if da is not None:
            array = da.from_array(source, chunks=chunks)
        else:
            array = LazyArray(source, chunks=chunks)
        return self._with_array(array)

    def lazy(self, chunks: Chunks = "auto") -> Self:
        return self.ensure_lazy(chunks)

    def compute(self, **kwargs) -> Self:
        if not self.is_lazy:
            return self
        if da is not None and isinstance(self.array, da.Array):
            array = dask.compute(self.array, **kwargs)[0]
        else:
            array = self.array.compute()
        return self._with_array(array)

    def copy_to_device(self, device: str) -> Self:
        return self._with_array(copy_to_device(self.array, device))

    def apply_func(self, func, **kwargs) -> Self:
        array = self.array
        if isinstance(array, LazyArray):
            array = array.compute()
        transformed = func(array, **kwargs)
        return self._with_array(transformed)

    def get_from_metadata(self, name: str, broadcastable: bool = False):
        axes_metadata_index = None
        data = None
        for i, (n, axis) in enumerate(zip(self.shape, self.ensemble_axes_metadata)):
            if axis.label == name:
                data = axis.coordinates(n)
                axes_metadata_index = i

        if axes_metadata_index is not None and broadcastable:
            return np.array(data)[
                (
                    *((None,) * axes_metadata_index),
                    slice(None),
                    *((None,) * (len(self.ensemble_shape) - 1 - axes_metadata_index)),
                )
            ]
        if axes_metadata_index is not None:
            if name in self.metadata:
                raise RuntimeError(
                    f"Could not resolve metadata for {name}, found in both ensemble axes metadata and metadata"
                )
            return data
        try:
            return self.metadata[name]
        except KeyError as exc:
            raise RuntimeError(f"Could not resolve metadata for {name}") from exc

    def apply_transform(self, transform) -> Self:
        array = self.array
        if isinstance(array, LazyArray):
            array = array.compute()
        if not isinstance(array, torch.Tensor):
            array = torch.as_tensor(array)
        result = transform.apply(array)
        if isinstance(result, list):
            raise TypeError("List-valued transforms are not supported by this Torch-native ArrayObject path")
        return self._with_array(result)

    def to_cpu(self) -> Self:
        return self.copy_to_device("cpu")

    def to_gpu(self, device: str = "gpu") -> Self:
        return self.copy_to_device(device)

    def to_zarr(self, url: str, compute: bool = True, overwrite: bool = False, **kwargs):
        return ComputableList([self]).to_zarr(url=url, compute=compute, overwrite=overwrite, **kwargs)

    @classmethod
    def _pack_kwargs(cls, kwargs):
        attrs = {}
        for key, value in kwargs.items():
            if key == "ensemble_axes_metadata":
                attrs[key] = [axis_to_dict(axis) for axis in value]
            else:
                attrs[key] = value
        return attrs

    @classmethod
    def _unpack_kwargs(cls, attrs):
        kwargs = {"ensemble_axes_metadata": []}
        for key, value in attrs.items():
            if key == "ensemble_axes_metadata":
                kwargs["ensemble_axes_metadata"] = [axis_from_dict(d) for d in value]
            elif key != "type":
                kwargs[key] = value
        return kwargs

    def _metadata_to_dict(self):
        metadata = copy.copy(self.metadata)
        metadata["axes"] = {f"axis_{i}": axis_to_dict(axis) for i, axis in enumerate(self.axes_metadata)}
        metadata["data_origin"] = "tspi_torchtem"
        metadata["type"] = self.__class__.__name__
        return metadata

    def _metadata_to_json(self):
        return json.dumps(self._metadata_to_dict())

    def __getitem__(self, items):
        items = _validate_array_items(items, self.shape)
        array = self.array[items]
        kwargs = self._copy_kwargs(exclude=("array", "ensemble_axes_metadata"))
        full_items = list(items) + [slice(None)] * (len(self.shape) - len(items))
        kwargs["ensemble_axes_metadata"] = [
            axis
            for i, axis in enumerate(self.ensemble_axes_metadata)
            if not isinstance(full_items[i], int)
        ]
        return self.__class__(array=array, **kwargs)
