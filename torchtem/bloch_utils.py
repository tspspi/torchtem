from __future__ import annotations

import itertools
from typing import Optional, Sequence

import numpy as np
import torch

from tspi.torchtem.physics import energy2wavelength


def reciprocal_cell(cell: torch.Tensor | np.ndarray) -> torch.Tensor:
    cell_tensor = torch.as_tensor(cell, dtype=torch.float64)
    return torch.linalg.pinv(cell_tensor).transpose(0, 1)


def calculate_g_vec(hkl: torch.Tensor | np.ndarray, cell: torch.Tensor | np.ndarray) -> torch.Tensor:
    hkl_tensor = torch.as_tensor(hkl, dtype=torch.float64)
    return hkl_tensor @ reciprocal_cell(cell)


def calculate_g_vec_length(
    hkl: torch.Tensor | np.ndarray, cell: torch.Tensor | np.ndarray
) -> torch.Tensor:
    return torch.linalg.norm(calculate_g_vec(hkl, cell), dim=-1)


def hkl_strings_to_array(hkl: list[str]) -> torch.Tensor:
    return torch.tensor([tuple(map(int, hkli.split(" "))) for hkli in hkl], dtype=torch.int64)


def generate_linear_combinations(
    vectors: torch.Tensor | np.ndarray,
    coefficients: Sequence[int],
    exclude_zero: bool = False,
) -> torch.Tensor:
    vectors_tensor = torch.as_tensor(vectors, dtype=torch.float64)
    combinations = torch.stack(
        [
            sum(int(c) * v for c, v in zip(coef_comb, vectors_tensor))
            for coef_comb in itertools.product(coefficients, repeat=len(vectors_tensor))
        ],
        dim=0,
    )
    if exclude_zero:
        combinations = combinations[~torch.all(combinations == 0, dim=1)]
    return combinations


def get_shortest_g_vec_length(cell: torch.Tensor | np.ndarray) -> float:
    combinations = generate_linear_combinations(
        reciprocal_cell(cell), [-1, 0, 1], exclude_zero=True
    )
    return float(torch.linalg.norm(combinations, dim=1).min().item())


def cell_bounds(cell: torch.Tensor | np.ndarray) -> torch.Tensor:
    cell_tensor = torch.as_tensor(cell, dtype=torch.float64)
    origin = torch.zeros(3, dtype=cell_tensor.dtype, device=cell_tensor.device)
    vertices = torch.stack(
        [
            origin,
            cell_tensor[0],
            cell_tensor[1],
            cell_tensor[2],
            cell_tensor[0] + cell_tensor[1],
            cell_tensor[0] + cell_tensor[2],
            cell_tensor[1] + cell_tensor[2],
            cell_tensor[0] + cell_tensor[1] + cell_tensor[2],
        ]
    )
    return vertices.max(dim=0).values - vertices.min(dim=0).values


def reciprocal_space_gpts(
    cell: torch.Tensor | np.ndarray,
    g_max: float,
) -> tuple[int, int, int]:
    dk = 1.0 / cell_bounds(cell)
    return tuple(int(torch.ceil(torch.as_tensor(g_max) / d).item()) * 2 + 1 for d in dk)


def make_hkl_grid(
    cell: torch.Tensor | np.ndarray,
    g_max: float,
    axes: tuple[int, ...] = (0, 1, 2),
) -> torch.Tensor:
    gpts = reciprocal_space_gpts(cell, g_max)
    freqs = tuple(torch.fft.fftfreq(n, d=1 / n).to(torch.int64) for n in gpts)
    freqs = tuple(freqs[axis] for axis in axes)
    hkl_grids = torch.meshgrid(*freqs, indexing="ij")
    hkl = torch.stack(hkl_grids, dim=-1).reshape(-1, len(axes))
    g_vec = calculate_g_vec(hkl, cell)
    return hkl[(g_vec.square().sum(dim=-1) <= g_max**2)]


def excitation_errors(
    g: torch.Tensor | np.ndarray, energy: float, use_wave_eq: bool = False
) -> torch.Tensor:
    g_tensor = torch.as_tensor(g, dtype=torch.float64)
    wavelength = energy2wavelength(energy, device=g_tensor.device, dtype=g_tensor.dtype)
    if use_wave_eq:
        return (-2 * g_tensor[..., 2] - wavelength * (g_tensor[..., 0].square() + g_tensor[..., 1].square())) / 2.0
    return (-2 * g_tensor[..., 2] - wavelength * g_tensor.square().sum(dim=-1)) / 2.0


def calculate_M_matrix(
    hkl: torch.Tensor | np.ndarray,
    cell: torch.Tensor | np.ndarray,
    energy: float,
) -> torch.Tensor:
    g = calculate_g_vec(hkl, cell)
    k0 = 1.0 / energy2wavelength(energy, device=g.device, dtype=g.dtype)
    return 1.0 / torch.sqrt(1.0 + g[:, 2] / k0)


def get_reflection_condition(hkl: torch.Tensor | np.ndarray, centering: str) -> torch.Tensor:
    hkl_tensor = torch.as_tensor(hkl, dtype=torch.int64)
    cl = centering.lower()
    if cl == "f":
        all_even = torch.all(hkl_tensor % 2 == 0, dim=1)
        all_odd = torch.all(hkl_tensor % 2 == 1, dim=1)
        return all_even | all_odd
    if cl == "i":
        return hkl_tensor.sum(dim=1) % 2 == 0
    if cl == "a":
        return (hkl_tensor[:, 1] + hkl_tensor[:, 2]) % 2 == 0
    if cl == "b":
        return (hkl_tensor[:, 0] + hkl_tensor[:, 2]) % 2 == 0
    if cl == "c":
        return (hkl_tensor[:, 0] + hkl_tensor[:, 1]) % 2 == 0
    if cl == "p":
        return torch.ones((hkl_tensor.shape[0],), dtype=torch.bool, device=hkl_tensor.device)
    raise ValueError("Invalid crystal centering type.")


def filter_reciprocal_space_vectors(
    hkl: torch.Tensor | np.ndarray,
    cell: torch.Tensor | np.ndarray,
    energy: float,
    sg_max: float,
    g_max: float,
    centering: str = "P",
    orientation_matrices: Optional[torch.Tensor | np.ndarray] = None,
) -> torch.Tensor:
    hkl_tensor = torch.as_tensor(hkl, dtype=torch.float64)
    g = hkl_tensor @ reciprocal_cell(cell)
    g_length = torch.linalg.norm(g, dim=-1)

    if orientation_matrices is None:
        mask = torch.abs(excitation_errors(g, energy, use_wave_eq=False)) <= sg_max
    else:
        R = torch.as_tensor(orientation_matrices, dtype=torch.float64)
        if R.ndim == 2:
            R = R.unsqueeze(0)
        wavelength = energy2wavelength(energy, device=g.device, dtype=g.dtype)
        b = 0.5 * wavelength * g.square().sum(dim=-1)
        mask = torch.zeros((g.shape[0],), dtype=torch.bool, device=g.device)
        for Ri in R.reshape(-1, 3, 3):
            sg = -g[:, 0] * Ri[2, 0] - g[:, 1] * Ri[2, 1] - g[:, 2] * Ri[2, 2] - b
            mask |= torch.abs(sg) < sg_max

    mask &= get_reflection_condition(hkl_tensor.to(torch.int64), centering)
    mask &= g_length <= g_max
    return mask


def ravel_hkl(hkl: torch.Tensor | np.ndarray, gpts: tuple[int, int, int]) -> torch.Tensor:
    hkl_tensor = torch.as_tensor(hkl, dtype=torch.int64)
    shift = torch.tensor((gpts[0] // 2, gpts[1] // 2, gpts[2] // 2), dtype=torch.int64)
    shifted = hkl_tensor + shift
    return shifted[:, 0] * gpts[1] * gpts[2] + shifted[:, 1] * gpts[2] + shifted[:, 2]


def retrieve_structure_factor_values(
    array: torch.Tensor,
    hkl_source: torch.Tensor | np.ndarray,
    hkl_destination: torch.Tensor | np.ndarray,
    gpts: tuple[int, int, int],
) -> torch.Tensor:
    source_idx = ravel_hkl(hkl_source, gpts)
    dest_idx = ravel_hkl(hkl_destination, gpts)
    lookup = {int(idx.item()): array[i] for i, idx in enumerate(source_idx)}
    return torch.stack([lookup[int(idx.item())] for idx in dest_idx], dim=0)


def are_vectors_orthogonal(
    v1: torch.Tensor | np.ndarray, v2: torch.Tensor | np.ndarray, tol: float = 1e-9
) -> bool:
    dot_product = torch.dot(torch.as_tensor(v1, dtype=torch.float64), torch.as_tensor(v2, dtype=torch.float64))
    return bool(torch.isclose(dot_product, torch.tensor(0.0, dtype=torch.float64), atol=tol))


def check_orthogonality(vectors: torch.Tensor | np.ndarray, tol: float = 1e-9) -> bool:
    vectors_tensor = torch.as_tensor(vectors, dtype=torch.float64)
    if vectors_tensor.shape[1] != 3:
        raise ValueError("Each vector must be 3-dimensional.")
    for i in range(vectors_tensor.shape[0]):
        for j in range(i + 1, vectors_tensor.shape[0]):
            if not are_vectors_orthogonal(vectors_tensor[i], vectors_tensor[j], tol):
                return False
    return True


def relative_positions_for_centering() -> dict[str, torch.Tensor]:
    return {
        "F": torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]], dtype=torch.float64),
        "I": torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=torch.float64),
        "A": torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=torch.float64),
        "B": torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=torch.float64),
        "C": torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]], dtype=torch.float64),
        "P": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float64),
    }


def wrapped_is_close(a: torch.Tensor | np.ndarray, b: torch.Tensor | np.ndarray) -> torch.Tensor:
    a_tensor = torch.as_tensor(a, dtype=torch.float64)
    b_tensor = torch.as_tensor(b, dtype=torch.float64)
    differences = torch.remainder(a_tensor.unsqueeze(1) - b_tensor.unsqueeze(0), 1.0)
    is_close_to_zero = torch.isclose(differences, torch.tensor(0.0, dtype=differences.dtype), atol=1e-8)
    is_close_to_one = torch.isclose(differences, torch.tensor(1.0, dtype=differences.dtype), atol=1e-8)
    return is_close_to_zero | is_close_to_one


def all_positions_have_relative_periodic_pair(
    positions: torch.Tensor | np.ndarray, relative_positions: torch.Tensor | np.ndarray
) -> bool:
    positions_tensor = torch.as_tensor(positions, dtype=torch.float64)
    relative_tensor = torch.as_tensor(relative_positions, dtype=torch.float64)
    basis_size = len(positions_tensor) / len(relative_tensor)
    if not np.isclose(basis_size, np.round(basis_size), atol=1e-6):
        return False
    num_match_total = 0
    for position in positions_tensor:
        shifted = torch.remainder(position + relative_tensor, 1.0)
        position_is_close = wrapped_is_close(shifted, positions_tensor)
        num_match_total += position_is_close.all(dim=2).sum().item()
    return num_match_total >= len(relative_tensor) * len(positions_tensor)


def auto_detect_centering(
    scaled_positions: torch.Tensor | np.ndarray,
    numbers: torch.Tensor | np.ndarray,
    cell: torch.Tensor | np.ndarray,
    centerings_to_check: Optional[set[str]] = None,
) -> str:
    if centerings_to_check is None:
        centerings_to_check = set(relative_positions_for_centering().keys())
    else:
        centerings_to_check = set(centerings_to_check)

    cell_tensor = torch.as_tensor(cell, dtype=torch.float64)
    if "P" in centerings_to_check:
        centerings_to_check.remove("P")
    if "F" in centerings_to_check or "I" in centerings_to_check:
        if not check_orthogonality(cell_tensor):
            centerings_to_check.discard("F")
            centerings_to_check.discard("I")
    if "A" in centerings_to_check and not check_orthogonality(cell_tensor[[0, 1]]):
        centerings_to_check.discard("A")
    if "B" in centerings_to_check and not check_orthogonality(cell_tensor[[0, 2]]):
        centerings_to_check.discard("B")
    if "C" in centerings_to_check and not check_orthogonality(cell_tensor[[1, 2]]):
        centerings_to_check.discard("C")

    positions_tensor = torch.as_tensor(scaled_positions, dtype=torch.float64)
    numbers_tensor = torch.as_tensor(numbers)
    relative_positions = relative_positions_for_centering()

    for number in torch.unique(numbers_tensor):
        centerings_to_check = {
            centering
            for centering in centerings_to_check
            if all_positions_have_relative_periodic_pair(
                positions_tensor[numbers_tensor == number], relative_positions[centering]
            )
        }

    if len(centerings_to_check) == 1:
        return next(iter(centerings_to_check))
    return "P"
