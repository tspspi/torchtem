from __future__ import annotations

import dataclasses
from copy import copy
from dataclasses import dataclass
from numbers import Number
from typing import Any, Optional

import numpy as np

from tspi.torchtem.chunks import iterate_chunk_ranges, validate_chunks
from tspi.torchtem.misc_utils import safe_equality
from tspi.torchtem.units import format_units, get_conversion_factor, validate_units


def format_label(axis: AxisMetadata, units: Optional[str] = None) -> str:
    label = axis.tex_label or axis.label
    if len(label) == 0:
        return ""
    if units is None and axis.units is not None:
        units = axis.units
    units = format_units(units)
    return f"{label} [{units}]" if units else label


def latex_float(number: float, formatting: str) -> str:
    float_str = f"{number:>{formatting}}"
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return f"{base} \\times 10^{{{int(exponent)}}}"
    return float_str


def format_value(
    value: Number | tuple, formatting: str, tolerance: float = 1e-14
) -> str:
    if isinstance(value, (tuple, list, np.ndarray)):
        return ", ".join(format_value(v, formatting=formatting) for v in value)
    if isinstance(value, float):
        float_value = 0.0 if np.abs(value) < tolerance else value
        return f"{float_value:>{formatting}}"
    if isinstance(value, (int, str, np.number)):
        return str(value)
    raise ValueError(f"Cannot format value of type {type(value)}")


def format_title(
    axis: OrdinalAxis,
    formatting: Optional[str] = None,
    units: Optional[str] = None,
    include_label: bool = True,
) -> str:
    if formatting is None:
        formatting = ".3f"
    if units:
        value = axis.values[0] * get_conversion_factor(units, axis.units)
    else:
        value = axis.values[0]
    units = validate_units(units, axis.units)
    label = f"{axis.label} = " if include_label and axis.label else ""
    suffix = f" {units}" if units else ""
    if isinstance(value, tuple):
        return f"{label}{format_value(value, formatting)}{suffix}"
    return f"{label}{value:>{formatting}}{suffix}"


@dataclass(eq=False, repr=False, unsafe_hash=True)
class AxisMetadata:
    label: str = ""
    units: Optional[str] = None
    tex_label: Optional[str] = None
    tex_units: Optional[str] = None
    _default_type: str = "index"
    _concatenate: bool = True
    _ensemble_mean: bool = False
    _squeeze: bool = False

    def _tabular_repr_data(self, n: int):
        return [self.format_type(), self.format_label(), self.format_coordinates(n)]

    def format_coordinates(self, n: Optional[int] = None):
        return "-"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        return safe_equality(self, other)

    def coordinates(self, n: int) -> tuple:
        return tuple(np.arange(n))

    def format_type(self) -> str:
        return self.__class__.__name__

    def format_label(self, units: Optional[str] = None) -> str:
        return format_label(self, units=units)

    def format_title(self, *args: Any, **kwargs: Any) -> str:
        return f"{self.label}"

    def item_metadata(self, item, metadata=None) -> dict:
        return {}

    def to_ordinal_axis(self, n: int):
        values = tuple(range(n))
        return OrdinalAxis(
            label=self.label,
            tex_label=self.tex_label,
            units=self.units,
            values=values,
            _concatenate=self._concatenate,
        )

    def _to_blocks(self, chunks):
        validated = validate_chunks(shape=(1,), chunks=chunks)
        blocks = []
        for _, _ in iterate_chunk_ranges(validated):
            blocks.append(copy(self))
        return tuple(blocks)

    def copy(self):
        return copy(self)

    def to_dict(self) -> dict:
        data = dataclasses.asdict(self)
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                data[key] = tuple(value.tolist())
        data["type"] = self.__class__.__name__
        return data

    def concatenate(self, other: AxisMetadata):
        if not self._concatenate:
            raise RuntimeError("Axis does not support concatenation")
        if not self.__eq__(other):
            raise RuntimeError("Axis metadata mismatch")
        return self

    @staticmethod
    def from_dict(d: dict):
        cls = globals()[d["type"]]
        return cls(**{key: value for key, value in d.items() if key != "type"})

    def limits(self, n=None) -> tuple:
        coordinates = self.coordinates(n)
        return coordinates[0], coordinates[-1]


@dataclass(eq=False, repr=False, unsafe_hash=True)
class UnknownAxis(AxisMetadata):
    label: str = "unknown"


@dataclass(eq=False, repr=False, unsafe_hash=True)
class SampleAxis(AxisMetadata):
    pass


@dataclass(eq=False, repr=False, unsafe_hash=True)
class LinearAxis(AxisMetadata):
    sampling: float = 1.0
    units: str = ""
    offset: float = 0.0

    def format_coordinates(self, n: Optional[int] = None) -> str:
        if n is None:
            raise ValueError("n must be provided")
        coordinates = self.coordinates(n)
        if n > 3:
            return f"{coordinates[0]:.2f} {coordinates[1]:.2f} ... {coordinates[-1]:.2f}"
        return " ".join(f"{coord:.2f}" for coord in coordinates)

    def coordinates(self, n: int) -> tuple[float, ...]:
        return tuple(
            np.linspace(self.offset, self.offset + self.sampling * n, n, endpoint=False)
        )

    def to_ordinal_axis(self, n: int):
        return OrdinalAxis(
            label=self.label,
            tex_label=self.tex_label,
            units=self.units,
            values=tuple(self.coordinates(n)),
            _concatenate=self._concatenate,
        )

    def convert_units(self, units: str, **kwargs):
        new_copy = self.copy()
        new_copy.units = units
        conversion = get_conversion_factor(units, old_units=self.units, **kwargs)
        new_copy.sampling = new_copy.sampling * conversion
        new_copy.offset = new_copy.offset * conversion
        return new_copy


@dataclass(eq=False, repr=False, unsafe_hash=True)
class RealSpaceAxis(LinearAxis):
    sampling: float = 1.0
    units: str = "pixels"
    endpoint: bool = True


@dataclass(eq=False, repr=False, unsafe_hash=True)
class ReciprocalSpaceAxis(LinearAxis):
    sampling: float = 1.0
    units: str = "pixels"
    fftshift: bool = True
    _concatenate: bool = False


@dataclass(eq=False, repr=False, unsafe_hash=True)
class ScanAxis(RealSpaceAxis):
    _main: bool = True


@dataclass(eq=False, repr=False, unsafe_hash=True)
class OrdinalAxis(AxisMetadata):
    values: tuple = ()

    def format_title(
        self, formatting: Optional[str] = None, include_label: bool = True, **kwargs
    ) -> str:
        return format_title(
            self, formatting=formatting, include_label=include_label, **kwargs
        )

    def format_all_titles(self) -> list[str]:
        return [
            f"{self.label} = {value} [{self.units}]"
            if i == 0
            else f"{self.label} [{self.units}]"
            for i, value in enumerate(self.values)
        ]

    def to_ordinal_axis(self, n: int):
        if n != len(self):
            raise ValueError("Axis length mismatch")
        return self

    def concatenate(self, other: AxisMetadata):
        if not safe_equality(self, other, ("values",)):
            raise RuntimeError("Axis metadata mismatch")
        assert isinstance(other, OrdinalAxis)
        kwargs = dataclasses.asdict(self)
        kwargs["values"] = kwargs["values"] + other.values
        return self.__class__(**kwargs)

    def __len__(self) -> int:
        return len(self.values)

    def __post_init__(self):
        if not isinstance(self.values, tuple):
            if isinstance(self.values, Number):
                self.values = (self.values,)
            else:
                self.values = tuple(self.values)

    def item_metadata(self, item, metadata=None):
        return {self.label: self.values[item]}

    def __getitem__(self, item):
        kwargs = dataclasses.asdict(self)
        if isinstance(item, Number):
            kwargs["values"] = (kwargs["values"][item],)
        else:
            array = np.empty(len(kwargs["values"]), dtype=object)
            array[:] = kwargs["values"]
            kwargs["values"] = tuple(array[item])
        return self.__class__(**kwargs)

    def coordinates(self, n: int) -> tuple:
        return self.values

    def _to_blocks(self, chunks):
        validated = validate_chunks(shape=(len(self),), chunks=chunks)
        blocks = []
        for _, slic in iterate_chunk_ranges(validated):
            blocks.append(self[slic[0]])
        return tuple(blocks)


@dataclass(eq=False, repr=False, unsafe_hash=True)
class NonLinearAxis(OrdinalAxis):
    units: str = "unknown"

    def format_coordinates(self, n: Optional[int] = None):
        if len(self.values) > 3:
            return f"{self.values[0]:.2f} {self.values[1]:.2f} ... {self.values[-1]:.2f}"
        try:
            return " ".join(f"{value:.2f}" for value in self.values)
        except TypeError:
            return self.values


@dataclass(eq=False, repr=False, unsafe_hash=True)
class AxisAlignedTiltAxis(NonLinearAxis):
    units: str = "mrad"
    direction: str = "x"
    _ensemble_mean: bool = False

    @property
    def tilt(self):
        if self.direction == "x":
            return tuple((value, 0.0) for value in self.values)
        if self.direction == "y":
            return tuple((0.0, value) for value in self.values)
        raise RuntimeError(f"Invalid tilt direction {self.direction}")

    def item_metadata(self, item, metadata=None):
        key = f"base_tilt_{self.direction}"
        value = self.values[item]
        if metadata is not None and key in metadata:
            value += metadata[key]
        return {key: value}


@dataclass(eq=False, repr=False, unsafe_hash=True)
class WaveVectorAxis(OrdinalAxis):
    units: str = "1/Å"


@dataclass(eq=False, repr=False, unsafe_hash=True)
class TiltAxis(OrdinalAxis):
    units: str = "mrad"

    @property
    def tilt(self) -> tuple:
        return self.values

    def item_metadata(self, item, metadata=None):
        return {
            "base_tilt_x": self.values[item][0],
            "base_tilt_y": self.values[item][1],
        }


@dataclass(eq=False, repr=False, unsafe_hash=True)
class ThicknessAxis(NonLinearAxis):
    label: str = "thickness"
    units: str = "Å"


@dataclass(eq=False, repr=False, unsafe_hash=True)
class ParameterAxis(NonLinearAxis):
    label: str = ""


@dataclass(eq=False, repr=False, unsafe_hash=True)
class PositionsAxis(OrdinalAxis):
    label: str = "x, y"
    units: str = "Å"

    def format_title(
        self, formatting: Optional[str] = None, include_label: bool = True, **kwargs
    ) -> str:
        if formatting is None:
            formatting = ".3f"
        formatted = ", ".join(f"{value:>{formatting}}" for value in self.values[0])
        return f"{self.label} = {formatted} {self.units}" if include_label else f"{formatted} {self.units}"


@dataclass(eq=False, repr=False, unsafe_hash=True)
class FrozenPhononsAxis(AxisMetadata):
    label: str = "Frozen phonons"


@dataclass(eq=False, repr=False, unsafe_hash=True)
class PrismPlaneWavesAxis(AxisMetadata):
    pass


@dataclass(eq=False, repr=False, unsafe_hash=True)
class ScaleAxis:
    label: str = ""
    units: Optional[str] = None
    tex_label: str | None = None

    def format_label(self):
        axis = AxisMetadata(label=self.label, units=self.units, tex_label=self.tex_label)
        return format_label(axis)


@dataclass(eq=False, repr=False, unsafe_hash=True)
class AxesMetadataList:
    metadata: list[AxisMetadata]
    shape: tuple[int, ...]

    def __iter__(self):
        return iter(self.metadata)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, item):
        return self.metadata[item]


AxesParameterAxis = ParameterAxis


def axis_to_dict(axis: AxisMetadata):
    data = dataclasses.asdict(axis)
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            data[key] = tuple(value.tolist())
    data["type"] = axis.__class__.__name__
    return data


def axis_from_dict(d: dict):
    cls = globals()[d["type"]]
    return cls(**{key: value for key, value in d.items() if key != "type"})


def format_axes_metadata(axes_metadata, shape) -> str:
    lines = ["type | label | coordinates"]
    for axis, n in zip(axes_metadata, shape):
        axis_type, label, coordinates = axis._tabular_repr_data(n)
        lines.append(f"{axis_type} | {label} | {coordinates}")
    return "\n".join(lines)
