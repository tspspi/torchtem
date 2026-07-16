from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable

import torch

def _cell_c_vector(cell: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(cell, dtype=torch.float64)[2]


def _cell_c_length(cell: torch.Tensor) -> float:
    return float(torch.linalg.norm(_cell_c_vector(cell)).item())


def _fractional_positions(positions_A: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    positions_A = torch.as_tensor(positions_A, dtype=torch.float64)
    cell = torch.as_tensor(cell, dtype=torch.float64)
    return positions_A @ torch.linalg.inv(cell)


def _axial_positions_along_c(positions_A: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    frac = _fractional_positions(positions_A, cell)
    return frac[:, 2] * _cell_c_length(cell)


def crystal_slice_thicknesses(
    positions_A: torch.Tensor,
    cell: torch.Tensor,
    tolerance: float = 0.2,
) -> torch.Tensor:
    positions_A = torch.as_tensor(positions_A, dtype=torch.float64)
    cell = torch.as_tensor(cell, dtype=torch.float64)
    z = _axial_positions_along_c(positions_A, cell)
    c_length = _cell_c_length(cell)
    if z.numel() == 0:
        return torch.tensor([c_length], dtype=torch.float64)

    interior = (torch.unique(torch.floor(z / tolerance).to(torch.int64)).to(torch.float64) + 0.5) * tolerance
    interior = interior[(interior > 0.0) & (interior < c_length)]
    slice_positions = torch.cat(
        [
            torch.tensor([0.0], dtype=torch.float64),
            torch.sort(interior).values,
            torch.tensor([c_length], dtype=torch.float64),
        ]
    )
    thicknesses = slice_positions[1:] - slice_positions[:-1]
    if not torch.isclose(thicknesses.sum(), torch.tensor(c_length, dtype=torch.float64), atol=1e-8):
        raise RuntimeError("Crystal slice thicknesses do not sum to the cell c-axis length.")
    return thicknesses


def validate_slice_thickness(
    slice_thickness: float | torch.Tensor | Iterable[float],
    *,
    thickness: float | None = None,
    num_slices: int | None = None,
) -> tuple[float, ...]:
    if isinstance(slice_thickness, torch.Tensor) and slice_thickness.ndim == 0:
        slice_thickness = float(slice_thickness.item())

    if isinstance(slice_thickness, (int, float)):
        slice_thickness = float(slice_thickness)
        if slice_thickness <= 0.0:
            raise ValueError(f"slice_thickness must be positive, got {slice_thickness}")
        if thickness is not None:
            n = int(torch.ceil(torch.tensor(thickness / slice_thickness)).item())
            validated = (float(thickness) / n,) * n
        elif num_slices is not None:
            validated = (slice_thickness,) * int(num_slices)
        else:
            raise RuntimeError("Either thickness or num_slices must be given.")
    else:
        validated = tuple(float(d) for d in slice_thickness)

    if thickness is not None and not math_isclose(sum(validated), float(thickness)):
        raise RuntimeError("Sum of slice thicknesses must equal the cell thickness.")
    if num_slices is not None and len(validated) != int(num_slices):
        raise RuntimeError("Number of slice thicknesses must match num_slices.")
    return validated


def slice_limits(slice_thickness: Iterable[float]) -> list[tuple[float, float]]:
    cumulative = list(itertools.accumulate((0.0,) + tuple(slice_thickness)))
    return [(cumulative[i], cumulative[i + 1]) for i in range(len(cumulative) - 1)]


def math_isclose(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 1e-9) -> bool:
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


@dataclass
class SlicedAtoms:
    positions_A: torch.Tensor
    atomic_numbers: torch.Tensor
    cell: torch.Tensor
    slice_thickness: tuple[float, ...]

    def __post_init__(self) -> None:
        self.positions_A = torch.as_tensor(self.positions_A, dtype=torch.float64)
        self.atomic_numbers = torch.as_tensor(self.atomic_numbers, dtype=torch.int64)
        self.cell = torch.as_tensor(self.cell, dtype=torch.float64)
        if torch.linalg.det(self.cell).abs() <= 1e-12:
            raise RuntimeError("torchtem.SlicedAtoms requires a non-singular cell.")
        self.slice_thickness = validate_slice_thickness(
            self.slice_thickness,
            thickness=_cell_c_length(self.cell),
        )
        self._axial_positions_A = _axial_positions_along_c(self.positions_A, self.cell)

    @property
    def num_slices(self) -> int:
        return len(self.slice_thickness)

    @property
    def box(self) -> tuple[float, float, float]:
        lengths = torch.linalg.norm(self.cell, dim=1)
        return float(lengths[0]), float(lengths[1]), float(lengths[2])

    @property
    def limits(self) -> list[tuple[float, float]]:
        return slice_limits(self.slice_thickness)

    def get_atoms_in_slices(
        self,
        first_slice: int,
        last_slice: int | None = None,
        atomic_number: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if last_slice is None:
            last_slice = first_slice + 1
        if first_slice < 0 or last_slice > self.num_slices or first_slice >= last_slice:
            raise IndexError("Invalid slice selection.")
        lower = self.limits[first_slice][0]
        upper = self.limits[last_slice - 1][1]
        mask = (self._axial_positions_A >= lower) & (self._axial_positions_A < upper)
        if atomic_number is not None:
            mask = mask & (self.atomic_numbers == int(atomic_number))
        return self.positions_A[mask], self.atomic_numbers[mask]

    def generate_atoms_in_slices(
        self,
        first_slice: int = 0,
        last_slice: int | None = None,
        atomic_number: int | None = None,
    ):
        if last_slice is None:
            last_slice = self.num_slices
        for idx in range(first_slice, last_slice):
            yield self.get_atoms_in_slices(idx, idx + 1, atomic_number=atomic_number)
