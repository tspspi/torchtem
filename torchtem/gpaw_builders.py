from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from tspi.torchtem.charge_density import (
    integrate_gradient_fourier,
    interpolate_slice,
    spatial_frequencies,
)
from tspi.torchtem.fft_support import fft_crop, fft_interpolate
from tspi.torchtem.iam import IAMPotentialBuilder
from tspi.torchtem.magnetism import _real_space_mesh
from tspi.torchtem.slicing import validate_slice_thickness

BOHR_MAGNETON = 9.2740100657e-24 * 1e20  # A * Å^2
VACUUM_PERMEABILITY = 1.25663706127e-6 * 1e10  # T * Å / A


def _axis_length(cell: torch.Tensor, axis: int) -> float:
    return float(torch.linalg.norm(cell[axis]).item())


def curl_fourier(vector_field: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    """Curl of a 3D vector field using Fourier-space differentiation."""
    if vector_field.shape[0] != 3:
        raise ValueError("vector_field must have leading dimension 3")
    F_hat = torch.fft.fftn(vector_field, dim=(-3, -2, -1))
    ky, kx, kz = spatial_frequencies(
        tuple(vector_field.shape[1:]), cell, device=vector_field.device, dtype=torch.float64
    )
    curl_y_hat = 2.0j * math.pi * (kz * F_hat[0] - ky * F_hat[2])
    curl_x_hat = 2.0j * math.pi * (ky * F_hat[1] - kx * F_hat[0])
    curl_z_hat = 2.0j * math.pi * (kx * F_hat[2] - kz * F_hat[1])
    curl_hat = torch.stack((curl_x_hat, curl_y_hat, curl_z_hat), dim=0)
    return torch.fft.ifftn(curl_hat, dim=(-3, -2, -1)).real


def apply_rotation_matrix(vector_field: torch.Tensor, rotation_matrix: torch.Tensor) -> torch.Tensor:
    shape = vector_field.shape[1:]
    reshaped = vector_field.reshape(3, -1)
    rotated = rotation_matrix.to(device=vector_field.device, dtype=vector_field.dtype) @ reshaped
    return rotated.reshape((3,) + shape)


def rotate_vector_field(
    vector_field: torch.Tensor, euler_angles_xyz: tuple[float, float, float]
) -> torch.Tensor:
    ax, ay, az = [torch.as_tensor(v, dtype=vector_field.dtype, device=vector_field.device) for v in euler_angles_xyz]
    cx, sx = torch.cos(ax), torch.sin(ax)
    cy, sy = torch.cos(ay), torch.sin(ay)
    cz, sz = torch.cos(az), torch.sin(az)
    rx = torch.stack(
        [
            torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)]),
            torch.stack([torch.zeros_like(cx), cx, -sx]),
            torch.stack([torch.zeros_like(cx), sx, cx]),
        ]
    )
    ry = torch.stack(
        [
            torch.stack([cy, torch.zeros_like(cy), sy]),
            torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)]),
            torch.stack([-sy, torch.zeros_like(cy), cy]),
        ]
    )
    rz = torch.stack(
        [
            torch.stack([cz, -sz, torch.zeros_like(cz)]),
            torch.stack([sz, cz, torch.zeros_like(cz)]),
            torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)]),
        ]
    )
    return apply_rotation_matrix(vector_field, rz @ ry @ rx)


def auto_rotation_matrix_for_vector_field(vector_field: torch.Tensor) -> torch.Tensor:
    ax = vector_field[0].to(torch.float64)
    ay = vector_field[1].to(torch.float64)
    sxx = torch.sum(ax * ax)
    syy = torch.sum(ay * ay)
    if sxx >= syy:
        return torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            dtype=torch.float64,
            device=vector_field.device,
        )
    return torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=torch.float64,
        device=vector_field.device,
    )


def calculate_magnetic_vector_potential(spin_density: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
    m = torch.stack(
        [torch.zeros_like(spin_density), torch.zeros_like(spin_density), spin_density], dim=0
    )
    j = BOHR_MAGNETON * curl_fourier(m, cell)
    A = -VACUUM_PERMEABILITY * integrate_gradient_fourier(j, cell)
    return A


def integrate_slice_fft(
    array: torch.Tensor,
    gpts: tuple[int, int],
    a: float,
    b: float,
    thickness_A: float,
) -> torch.Tensor:
    """Source-style orthogonal-slice integration with reciprocal-space cropping."""
    if array.ndim != 3:
        raise ValueError("array must have shape (ny, nx, nz)")
    dz = float(thickness_A) / array.shape[2]
    na = int(math.floor(float(a) / dz))
    nb = int(math.floor(float(b) / dz))
    nb = max(nb, na + 1)
    na = max(0, min(na, array.shape[2] - 1))
    nb = max(na + 1, min(nb, array.shape[2]))

    slice_array = array[..., na:nb].sum(dim=-1) * dz
    new_shape = (nb - na,) + tuple(gpts)
    old_shape = (nb - na,) + tuple(slice_array.shape)
    slice_array = torch.fft.fftn(slice_array.to(torch.complex128))
    slice_array = fft_crop(slice_array, gpts)
    slice_array = torch.fft.ifftn(slice_array).real * (
        math.prod(new_shape) / math.prod(old_shape)
    )
    return slice_array


@dataclass
class GPAWLikeState:
    cell: torch.Tensor
    valence_potential_3d: torch.Tensor | None = None
    spin_density_3d: torch.Tensor | None = None


class GPAWPotentialBuilder(nn.Module):
    """Tensor-native analogue of the GPAW electrostatic potential builder."""

    def __init__(
        self,
        *,
        valence_potential_3d: torch.Tensor,
        cell: torch.Tensor,
        slice_thickness: float | tuple[float, ...],
        gpts: tuple[int, int] | None = None,
        sampling: tuple[float, float] | None = None,
        positions_A: torch.Tensor | None = None,
        symbols: list[str] | None = None,
        parametrization: str = "lobato",
        projection: str = "real_space",
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        super().__init__()
        self.valence_potential_3d = nn.Parameter(valence_potential_3d.to(device=device, dtype=dtype))
        self.register_buffer("cell", torch.as_tensor(cell, device=device, dtype=dtype))
        self.slice_thickness = validate_slice_thickness(
            slice_thickness, thickness=_axis_length(self.cell, 2)
        )
        if gpts is None:
            gpts = tuple(int(n) for n in valence_potential_3d.shape[:2])
        self.gpts = gpts
        if sampling is None:
            sampling = (
                _axis_length(self.cell, 0) / gpts[0],
                _axis_length(self.cell, 1) / gpts[1],
            )
        self.sampling = sampling
        self.projection = projection
        self.dtype = dtype

        if positions_A is not None and symbols is not None:
            self.core_correction = IAMPotentialBuilder(
                gpts=gpts,
                sampling=sampling,
                positions_A=positions_A,
                symbols=symbols,
                cell=self.cell,
                slice_thickness=self.slice_thickness,
                parametrization=parametrization,
                projection="infinite" if parametrization != "peng" else "finite",
                dtype=dtype,
                device=device,
            )
        else:
            self.core_correction = None

    @property
    def num_slices(self) -> int:
        return len(self.slice_thickness)

    @property
    def slice_limits(self) -> list[tuple[float, float]]:
        lower = 0.0
        limits = []
        for dz in self.slice_thickness:
            upper = lower + dz
            limits.append((lower, upper))
            lower = upper
        return limits

    def valence_slices(self) -> torch.Tensor:
        if self.projection == "fft":
            return torch.stack(
                [
                    integrate_slice_fft(
                        self.valence_potential_3d,
                        self.gpts,
                        a,
                        b,
                        _axis_length(self.cell, 2),
                    )
                    for a, b in self.slice_limits
                ],
                dim=0,
            )

        return torch.stack(
            [
                interpolate_slice(
                    self.valence_potential_3d,
                    self.cell,
                    gpts=self.gpts,
                    sampling=self.sampling,
                    a=a,
                    b=b,
                )
                for a, b in self.slice_limits
            ],
            dim=0,
        )

    def forward(self) -> torch.Tensor:
        slices = self.valence_slices()
        if self.core_correction is not None:
            slices = slices + self.core_correction()
        return slices


class GPAWVectorPotentialBuilder(nn.Module):
    """Vector-potential builder from a precomputed spin density."""

    def __init__(
        self,
        *,
        spin_density_3d: torch.Tensor,
        cell: torch.Tensor,
        slice_thickness: float | tuple[float, ...],
        gpts: tuple[int, int] | None = None,
        sampling: tuple[float, float] | None = None,
        projection: str = "fft",
        rotate_field: tuple[float, float, float] | str | None = "auto",
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        super().__init__()
        self.spin_density_3d = nn.Parameter(spin_density_3d.to(device=device, dtype=dtype))
        self.register_buffer("cell", torch.as_tensor(cell, device=device, dtype=dtype))
        self.slice_thickness = validate_slice_thickness(
            slice_thickness, thickness=_axis_length(self.cell, 2)
        )
        if gpts is None:
            gpts = tuple(int(n) for n in spin_density_3d.shape[:2])
        self.gpts = gpts
        if sampling is None:
            sampling = (
                _axis_length(self.cell, 0) / gpts[0],
                _axis_length(self.cell, 1) / gpts[1],
            )
        self.sampling = sampling
        self.projection = projection
        self.rotate_field = rotate_field
        self.dtype = dtype

    @property
    def num_slices(self) -> int:
        return len(self.slice_thickness)

    @property
    def slice_limits(self) -> list[tuple[float, float]]:
        lower = 0.0
        limits = []
        for dz in self.slice_thickness:
            upper = lower + dz
            limits.append((lower, upper))
            lower = upper
        return limits

    def vector_potential_3d(self) -> torch.Tensor:
        array = calculate_magnetic_vector_potential(self.spin_density_3d, self.cell)
        if self.rotate_field == "auto":
            array = apply_rotation_matrix(array, auto_rotation_matrix_for_vector_field(array))
        elif self.rotate_field is not None:
            array = rotate_vector_field(array, self.rotate_field)
        return array

    def forward(self) -> torch.Tensor:
        array = self.vector_potential_3d()
        if self.projection == "fft":
            return torch.stack(
                [
                    torch.stack(
                        [
                            integrate_slice_fft(
                                array[c],
                                self.gpts,
                                a,
                                b,
                                _axis_length(self.cell, 2),
                            )
                            for c in range(3)
                        ],
                        dim=0,
                    )
                    for a, b in self.slice_limits
                ],
                dim=0,
            )

        return torch.stack(
            [
                torch.stack(
                    [
                        interpolate_slice(
                            array[c],
                            self.cell,
                            gpts=self.gpts,
                            sampling=self.sampling,
                            a=a,
                            b=b,
                        )
                        for c in range(3)
                    ],
                    dim=0,
                )
                for a, b in self.slice_limits
            ],
            dim=0,
        )


class GPAWMagneticFieldBuilder(nn.Module):
    """Magnetic-field builder from a precomputed spin density."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.vector_potential_builder = GPAWVectorPotentialBuilder(**kwargs)

    def forward(self) -> torch.Tensor:
        vector_slices = self.vector_potential_builder()
        return torch.stack(
            [
                torch.stack(
                    _slice_curl_2d(vector_slices[i], self.vector_potential_builder.sampling),
                    dim=0,
                )
                for i in range(vector_slices.shape[0])
            ],
            dim=0,
        )


def _slice_curl_2d(
    vector_slice: torch.Tensor, sampling: tuple[float, float]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ay = vector_slice[0]
    ax = vector_slice[1]
    az = vector_slice[2]
    d_az_dy = (torch.roll(az, -1, dims=-2) - torch.roll(az, 1, dims=-2)) / (2.0 * sampling[0])
    d_az_dx = (torch.roll(az, -1, dims=-1) - torch.roll(az, 1, dims=-1)) / (2.0 * sampling[1])
    d_ax_dy = (torch.roll(ax, -1, dims=-2) - torch.roll(ax, 1, dims=-2)) / (2.0 * sampling[0])
    d_ay_dx = (torch.roll(ay, -1, dims=-1) - torch.roll(ay, 1, dims=-1)) / (2.0 * sampling[1])
    bx = d_az_dy
    by = -d_az_dx
    bz = d_ax_dy - d_ay_dx
    return bx, by, bz
