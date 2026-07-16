from __future__ import annotations

import torch
from torch import nn

from torchtem.constants import kappa
from torchtem.bloch_utils import calculate_M_matrix
from torchtem.matrix_exponential import expm
from torchtem.physics import energy2sigma, energy2wavelength


def reciprocal_cell(cell: torch.Tensor) -> torch.Tensor:
    return torch.linalg.pinv(cell).transpose(0, 1)


def calculate_g_vec(hkl: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    return hkl.to(cell.dtype) @ reciprocal_cell(cell)


def excitation_errors(g: torch.Tensor, energy_eV: float, use_wave_eq: bool = False) -> torch.Tensor:
    wavelength = energy2wavelength(energy_eV, device=g.device, dtype=g.dtype)
    if use_wave_eq:
        return (-2.0 * g[..., 2] - wavelength * (g[..., 0].square() + g[..., 1].square())) / 2.0
    return (-2.0 * g[..., 2] - wavelength * g.square().sum(dim=-1)) / 2.0


def plane_wave_coefficients(hkl: torch.Tensor) -> torch.Tensor:
    return torch.all(hkl == torch.tensor([0, 0, 0], device=hkl.device, dtype=hkl.dtype), dim=1).to(torch.complex128)


def calculate_structure_matrix(
    structure_factors: torch.Tensor,
    hkl: torch.Tensor,
    hkl_selected: torch.Tensor,
    cell: torch.Tensor,
    energy_eV: float,
    use_wave_eq: bool = False,
) -> torch.Tensor:
    hkl = hkl.to(torch.int64)
    hkl_selected = hkl_selected.to(torch.int64)
    cell = cell.to(torch.float64)
    structure_factors = structure_factors.to(torch.complex128)

    g = calculate_g_vec(hkl_selected.to(torch.float64), cell)
    mii = calculate_M_matrix(hkl_selected, cell, energy_eV)

    structure_factor_lookup = {
        tuple(hkli.tolist()): structure_factors[i] for i, hkli in enumerate(hkl)
    }
    diffs = hkl_selected[None, :, :] - hkl_selected[:, None, :]
    factors = torch.stack(
        [
            structure_factor_lookup.get(
                tuple(diff.tolist()),
                torch.tensor(0.0 + 0.0j, dtype=structure_factors.dtype, device=structure_factors.device),
            )
            for diff in diffs.reshape(-1, 3)
        ],
        dim=0,
    ).reshape(hkl_selected.shape[0], hkl_selected.shape[0])

    prefactor = energy2sigma(energy_eV, device=factors.device, dtype=torch.float64) / (
        kappa * energy2wavelength(energy_eV, device=factors.device, dtype=torch.float64) * torch.pi
    )
    A = factors * prefactor.to(torch.complex128)
    A = A * mii[None].to(torch.complex128) * mii[:, None].to(torch.complex128)

    sg = excitation_errors(g, energy_eV, use_wave_eq=use_wave_eq)
    diag = (
        2.0
        / energy2wavelength(energy_eV, device=g.device, dtype=g.dtype)
        * sg
        * mii.to(g.dtype)
    ).to(torch.complex128)
    A = A.clone()
    indices = torch.arange(A.shape[0], device=A.device)
    A[indices, indices] = diag
    return A


def calculate_scattering_matrix(
    structure_matrix: torch.Tensor,
    hkl: torch.Tensor,
    cell: torch.Tensor,
    thickness_A: float,
    energy_eV: float,
    method: str = "expm",
) -> torch.Tensor:
    if method != "expm":
        raise ValueError(f"Unknown Bloch scattering matrix method: {method}")

    wavelength = energy2wavelength(energy_eV, device=structure_matrix.device, dtype=torch.float64)
    S = expm(1.0j * torch.pi * thickness_A * wavelength * structure_matrix)
    mii = calculate_M_matrix(hkl, cell, energy_eV).to(torch.complex128)
    M = torch.diag(mii)
    M_inv = torch.diag(1.0 / mii)
    return M @ S @ M_inv


def calculate_dynamical_scattering(
    structure_matrix: torch.Tensor,
    hkl: torch.Tensor,
    cell: torch.Tensor,
    energy_eV: float,
    thickness_A: float | torch.Tensor,
) -> torch.Tensor:
    thickness_tensor = torch.as_tensor(
        thickness_A, device=structure_matrix.device, dtype=torch.float64
    )
    mii = calculate_M_matrix(hkl, cell, energy_eV).to(torch.complex128)
    eigenvalues, eigenvectors = torch.linalg.eigh(structure_matrix)
    gamma = eigenvalues * energy2wavelength(
        energy_eV, device=structure_matrix.device, dtype=torch.float64
    ) / 2.0

    C = eigenvectors.clone()
    diag_indices = torch.arange(C.shape[0], device=C.device)
    C[diag_indices, diag_indices] = C[diag_indices, diag_indices] / mii
    C_inv = torch.conj(C.transpose(0, 1))

    initial = plane_wave_coefficients(hkl)
    alpha = C_inv @ initial

    if thickness_tensor.ndim == 0:
        phases = torch.exp(2.0j * torch.pi * thickness_tensor.to(torch.complex128) * gamma)
        return C @ (phases * alpha)

    waves = []
    for thickness in thickness_tensor.reshape(-1):
        phases = torch.exp(2.0j * torch.pi * thickness.to(torch.complex128) * gamma)
        waves.append(C @ (phases * alpha))
    return torch.stack(waves, dim=0)


class StructureFactorModel(nn.Module):
    """Differentiable structure-factor model from per-reflection values."""

    def __init__(self, hkl: torch.Tensor, structure_factors: torch.Tensor, cell: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("hkl", hkl.to(torch.int64))
        self.register_buffer("cell", cell.to(torch.float64))
        self.structure_factors = nn.Parameter(structure_factors.to(torch.complex128))

    def retrieve(self, differences: torch.Tensor) -> torch.Tensor:
        values = []
        for diff in differences:
            mask = torch.all(self.hkl == diff.to(self.hkl.dtype), dim=1)
            if torch.any(mask):
                values.append(self.structure_factors[torch.argmax(mask.to(torch.int64))])
            else:
                values.append(torch.tensor(0.0 + 0.0j, device=self.structure_factors.device, dtype=self.structure_factors.dtype))
        return torch.stack(values, dim=0)


class BlochWaveModel(nn.Module):
    """Minimal differentiable Bloch-wave / scattering-matrix model."""

    def __init__(
        self,
        *,
        hkl: torch.Tensor,
        cell: torch.Tensor,
        structure_factor_model: StructureFactorModel,
        energy_eV: float,
        use_wave_eq: bool = False,
    ) -> None:
        super().__init__()
        self.register_buffer("hkl", hkl.to(torch.int64))
        self.register_buffer("cell", cell.to(torch.float64))
        self.structure_factor_model = structure_factor_model
        self.energy_eV = float(energy_eV)
        self.use_wave_eq = bool(use_wave_eq)

    def calculate_structure_matrix(self) -> torch.Tensor:
        return calculate_structure_matrix(
            self.structure_factor_model.structure_factors,
            self.hkl,
            self.hkl,
            self.cell,
            self.energy_eV,
            use_wave_eq=self.use_wave_eq,
        )

    def calculate_scattering_matrix(self, thickness_A: float) -> torch.Tensor:
        return calculate_scattering_matrix(
            self.calculate_structure_matrix(),
            self.hkl,
            self.cell,
            thickness_A,
            self.energy_eV,
            method="expm",
        )

    def calculate_dynamical_scattering(
        self,
        thickness_A: float | torch.Tensor,
        *,
        method: str = "expm",
    ) -> torch.Tensor:
        if method == "expm":
            thickness_tensor = torch.as_tensor(
                thickness_A, device=self.cell.device, dtype=torch.float64
            )
            initial = plane_wave_coefficients(self.hkl)
            if thickness_tensor.ndim == 0:
                return self.calculate_scattering_matrix(float(thickness_tensor.item())) @ initial
            return torch.stack(
                [
                    self.calculate_scattering_matrix(float(thickness.item())) @ initial
                    for thickness in thickness_tensor.reshape(-1)
                ],
                dim=0,
            )

        if method != "decomposition":
            raise ValueError(f"Unknown Bloch dynamical scattering method: {method}")

        return calculate_dynamical_scattering(
            self.calculate_structure_matrix(),
            self.hkl,
            self.cell,
            self.energy_eV,
            thickness_A,
        )

    def dynamical_scattering(
        self,
        thickness_A: float | torch.Tensor,
        *,
        method: str = "expm",
    ) -> torch.Tensor:
        return self.calculate_dynamical_scattering(thickness_A, method=method)
