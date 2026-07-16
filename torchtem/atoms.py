from __future__ import annotations

import math

import torch


def plane_to_axes(plane: str) -> tuple[int, int, int]:
    axis_map = {"x": 0, "y": 1, "z": 2}
    axes = tuple(axis_map[axis] for axis in plane)
    last_axis = ({0, 1, 2} - set(axes)).pop()
    return axes + (last_axis,)


def is_cell_orthogonal(cell: torch.Tensor, tol: float = 1e-12) -> bool:
    cell = torch.as_tensor(cell, dtype=torch.float64)
    off_diagonal = cell[~torch.eye(3, dtype=torch.bool, device=cell.device)]
    return bool(torch.all(off_diagonal.abs() <= tol))


def is_cell_valid(cell: torch.Tensor, tol: float = 1e-12) -> bool:
    cell = torch.as_tensor(cell, dtype=torch.float64)
    if abs(cell[0, 0] - torch.linalg.norm(cell[0]).item()) > tol:
        return False
    if abs(cell[1, 2].item()) > tol:
        return False
    if abs(cell[2, 2] - torch.linalg.norm(cell[2]).item()) > tol:
        return False
    return True


def is_cell_hexagonal(cell: torch.Tensor, tol: float = 1e-12) -> bool:
    cell = torch.as_tensor(cell, dtype=torch.float64)
    a = torch.linalg.norm(cell[0])
    b = torch.linalg.norm(cell[1])
    c = torch.linalg.norm(cell[2])
    angle = torch.arccos(torch.dot(cell[0], cell[1]) / (a * b))
    return bool(
        torch.isclose(a, b, atol=tol)
        and (
            torch.isclose(angle, torch.tensor(math.pi / 3, dtype=cell.dtype), atol=tol)
            or torch.isclose(angle, torch.tensor(2 * math.pi / 3, dtype=cell.dtype), atol=tol)
        )
        and torch.isclose(c, cell[2, 2], atol=tol)
    )


def standardize_cell(cell: torch.Tensor, positions: torch.Tensor, tol: float = 1e-12) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardize an orthogonal or nearly orthogonal cell for torchtem use."""
    cell = torch.as_tensor(cell, dtype=torch.float64).clone()
    positions = torch.as_tensor(positions, dtype=torch.float64).clone()

    if is_cell_orthogonal(cell, tol=tol):
        diagonal = torch.abs(torch.diag(cell))
        diagonal = torch.where(diagonal > tol, diagonal, torch.zeros_like(diagonal))
        return torch.diag(diagonal), positions

    if not is_cell_valid(cell, tol=tol):
        raise RuntimeError("Cell cannot be standardized into the current torchtem orthogonal form.")

    diagonal = torch.tensor(
        [torch.linalg.norm(cell[0]), torch.linalg.norm(cell[1]), torch.linalg.norm(cell[2])],
        dtype=torch.float64,
        device=cell.device,
    )
    return torch.diag(diagonal), positions
