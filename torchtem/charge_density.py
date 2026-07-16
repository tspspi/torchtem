from __future__ import annotations

import math

import torch
from torch import nn

from tspi.torchtem.atoms import is_cell_orthogonal
from tspi.torchtem.slicing import validate_slice_thickness


def _is_cell_nonsingular(cell: torch.Tensor, tol: float = 1e-12) -> bool:
    cell = torch.as_tensor(cell, dtype=torch.float64)
    return bool(torch.linalg.det(cell).abs() > tol)


def spatial_frequencies_orthorhombic(
    shape: tuple[int, int, int],
    cell: torch.Tensor,
    *,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cell = torch.as_tensor(cell, dtype=dtype, device=device)
    if not is_cell_orthogonal(cell):
        raise RuntimeError("Only orthorhombic cells are currently supported.")
    lengths = torch.diag(cell)
    ky = torch.fft.fftfreq(shape[0], d=float(lengths[0] / shape[0]), device=device).to(dtype)
    kx = torch.fft.fftfreq(shape[1], d=float(lengths[1] / shape[1]), device=device).to(dtype)
    kz = torch.fft.fftfreq(shape[2], d=float(lengths[2] / shape[2]), device=device).to(dtype)
    return torch.meshgrid(ky, kx, kz, indexing="ij")


def spatial_frequencies_meshgrid(
    shape: tuple[int, int, int],
    cell: torch.Tensor,
    *,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cell = torch.as_tensor(cell, dtype=dtype, device=device)
    reciprocal = torch.linalg.inv(cell).transpose(0, 1)

    k0 = torch.fft.fftfreq(shape[0], d=1.0 / shape[0], device=device).to(dtype)
    k1 = torch.fft.fftfreq(shape[1], d=1.0 / shape[1], device=device).to(dtype)
    k2 = torch.fft.fftfreq(shape[2], d=1.0 / shape[2], device=device).to(dtype)
    g0, g1, g2 = torch.meshgrid(k0, k1, k2, indexing="ij")
    kp = torch.stack((g0, g1, g2), dim=-1).reshape(-1, 3)
    ky, kx, kz = (kp @ reciprocal).transpose(0, 1)
    return ky.reshape(shape), kx.reshape(shape), kz.reshape(shape)


def spatial_frequencies(
    shape: tuple[int, int, int],
    cell: torch.Tensor,
    *,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cell = torch.as_tensor(cell, dtype=dtype, device=device)
    if is_cell_orthogonal(cell):
        return spatial_frequencies_orthorhombic(shape, cell, device=device, dtype=dtype)
    if not _is_cell_nonsingular(cell):
        raise RuntimeError("Only non-singular cells are currently supported.")
    return spatial_frequencies_meshgrid(shape, cell, device=device, dtype=dtype)


def spatial_frequencies_squared(
    shape: tuple[int, int, int],
    cell: torch.Tensor,
    *,
    device=None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    ky, kx, kz = spatial_frequencies(shape, cell, device=device, dtype=dtype)
    return ky.square() + kx.square() + kz.square()


def integrate_gradient_fourier(
    array: torch.Tensor,
    cell: torch.Tensor,
    *,
    in_space: str = "real",
    out_space: str = "real",
) -> torch.Tensor:
    if in_space == "real":
        array = torch.fft.fftn(array, dim=(-3, -2, -1))

    k2 = spatial_frequencies_squared(
        tuple(array.shape[-3:]),
        cell,
        device=array.device,
        dtype=torch.float64,
    )
    denominator = -(2.0**2) * math.pi**2 * k2
    denominator = denominator.to(array.dtype)
    denominator = denominator.clone()
    denominator[0, 0, 0] = 1.0
    integrated = array / denominator

    if out_space == "real":
        integrated = torch.fft.ifftn(integrated, dim=(-3, -2, -1)).real
    return integrated


def fourier_space_delta(
    ky: torch.Tensor,
    kx: torch.Tensor,
    kz: torch.Tensor,
    position_A: torch.Tensor,
) -> torch.Tensor:
    return torch.exp(
        -2.0j
        * math.pi
        * (ky * position_A[0] + kx * position_A[1] + kz * position_A[2])
    )


def fourier_space_gaussian(k2: torch.Tensor, width_A: float) -> torch.Tensor:
    a = math.sqrt(1.0 / (2.0 * width_A**2)) / (2.0 * math.pi)
    return torch.exp(-(1.0 / (4.0 * a**2)) * k2)


def add_point_charges_fourier(
    array: torch.Tensor,
    positions_A: torch.Tensor,
    atomic_numbers: torch.Tensor,
    cell: torch.Tensor,
    *,
    broadening_A: float = 0.05,
) -> torch.Tensor:
    ky, kx, kz = spatial_frequencies(
        tuple(array.shape[-3:]), cell, device=array.device, dtype=torch.float64
    )
    k2 = ky.square() + kx.square() + kz.square()
    broadening = (
        fourier_space_gaussian(k2, broadening_A) if broadening_A > 0.0 else torch.ones_like(k2)
    ).to(array.dtype)

    cell = torch.as_tensor(cell, dtype=torch.float64, device=array.device)
    pixel_volume = torch.abs(torch.linalg.det(cell)) / float(
        array.shape[-3] * array.shape[-2] * array.shape[-1]
    )

    out = array.clone()
    for position, number in zip(positions_A.to(torch.float64), atomic_numbers.to(torch.float64)):
        scale = number / pixel_volume
        out = out + scale.to(array.dtype) * broadening * fourier_space_delta(ky, kx, kz, position)
    return out


def _trilinear_periodic_sample(array: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
    shape = array.shape
    base = torch.floor(coordinates).to(torch.int64)
    frac = coordinates - base.to(coordinates.dtype)

    i0 = torch.remainder(base[:, 0], shape[0])
    i1 = torch.remainder(base[:, 0] + 1, shape[0])
    j0 = torch.remainder(base[:, 1], shape[1])
    j1 = torch.remainder(base[:, 1] + 1, shape[1])
    k0 = torch.remainder(base[:, 2], shape[2])
    k1 = torch.remainder(base[:, 2] + 1, shape[2])

    wx0 = 1.0 - frac[:, 0]
    wx1 = frac[:, 0]
    wy0 = 1.0 - frac[:, 1]
    wy1 = frac[:, 1]
    wz0 = 1.0 - frac[:, 2]
    wz1 = frac[:, 2]

    c000 = array[i0, j0, k0]
    c001 = array[i0, j0, k1]
    c010 = array[i0, j1, k0]
    c011 = array[i0, j1, k1]
    c100 = array[i1, j0, k0]
    c101 = array[i1, j0, k1]
    c110 = array[i1, j1, k0]
    c111 = array[i1, j1, k1]

    return (
        c000 * wx0 * wy0 * wz0
        + c001 * wx0 * wy0 * wz1
        + c010 * wx0 * wy1 * wz0
        + c011 * wx0 * wy1 * wz1
        + c100 * wx1 * wy0 * wz0
        + c101 * wx1 * wy0 * wz1
        + c110 * wx1 * wy1 * wz0
        + c111 * wx1 * wy1 * wz1
    )


def interpolate_between_cells(
    array: torch.Tensor,
    new_shape: tuple[int, int, int],
    old_cell: torch.Tensor,
    new_cell: torch.Tensor,
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    old_cell = torch.as_tensor(old_cell, dtype=torch.float64, device=array.device)
    new_cell = torch.as_tensor(new_cell, dtype=torch.float64, device=array.device)
    offset_tensor = torch.as_tensor(offset, dtype=torch.float64, device=array.device)

    x = torch.linspace(0.0, 1.0, new_shape[0] + 1, device=array.device, dtype=torch.float64)[:-1]
    y = torch.linspace(0.0, 1.0, new_shape[1] + 1, device=array.device, dtype=torch.float64)[:-1]
    z = torch.linspace(0.0, 1.0, new_shape[2] + 1, device=array.device, dtype=torch.float64)[:-1]
    gx, gy, gz = torch.meshgrid(x, y, z, indexing="ij")
    coordinates = torch.stack((gx, gy, gz), dim=-1).reshape(-1, 3)
    coordinates = coordinates @ new_cell + offset_tensor

    mapped = coordinates @ torch.linalg.inv(old_cell)
    mapped = torch.remainder(mapped, 1.0)
    mapped = mapped * torch.tensor(array.shape, dtype=torch.float64, device=array.device)
    return _trilinear_periodic_sample(array, mapped).reshape(new_shape)


def interpolate_slice(
    volume: torch.Tensor,
    cell: torch.Tensor,
    *,
    gpts: tuple[int, int],
    sampling: tuple[float, float],
    a: float,
    b: float,
) -> torch.Tensor:
    cell = torch.as_tensor(cell, dtype=torch.float64, device=volume.device)
    if not _is_cell_nonsingular(cell):
        raise RuntimeError("Only non-singular cells are currently supported.")
    dz_charge = torch.linalg.norm(cell[2]).item() / volume.shape[-1]
    nz = max(int(math.ceil((b - a) / dz_charge)), 2)
    c_axis = cell[2]
    c_length = torch.linalg.norm(c_axis)
    c_unit = c_axis / c_length
    slice_box = torch.stack(
        (
            torch.tensor([gpts[0] * sampling[0], 0.0, 0.0], dtype=torch.float64, device=volume.device),
            torch.tensor([0.0, gpts[1] * sampling[1], 0.0], dtype=torch.float64, device=volume.device),
            c_unit * (b - a),
        ),
        dim=0,
    )
    slice_array = interpolate_between_cells(
        volume,
        (gpts[0], gpts[1], nz),
        cell,
        slice_box,
        offset=tuple((c_unit * a).tolist()),
    )
    dz = (b - a) / nz
    return slice_array.sum(dim=-1) * dz


class ChargeDensityPotential(nn.Module):
    """Differentiable charge-density potential builder for non-singular cells."""

    def __init__(
        self,
        *,
        charge_density: torch.Tensor,
        cell: torch.Tensor,
        slice_thickness: float | tuple[float, ...],
        positions_A: torch.Tensor | None = None,
        atomic_numbers: torch.Tensor | None = None,
        gpts: tuple[int, int] | None = None,
        sampling: tuple[float, float] | None = None,
        broadening_A: float = 0.05,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        super().__init__()
        self.charge_density = nn.Parameter(charge_density.to(device=device, dtype=dtype))
        self.register_buffer("cell", torch.as_tensor(cell, device=device, dtype=dtype))
        if positions_A is not None:
            self.positions_A = nn.Parameter(positions_A.to(device=device, dtype=dtype))
        else:
            self.positions_A = None
        if atomic_numbers is not None:
            self.register_buffer("atomic_numbers", atomic_numbers.to(device=device, dtype=dtype))
        else:
            self.atomic_numbers = None
        self.slice_thickness = validate_slice_thickness(
            slice_thickness, thickness=float(torch.linalg.norm(self.cell[2]).item())
        )
        if gpts is None:
            gpts = tuple(int(n) for n in charge_density.shape[:2])
        self.gpts = gpts
        if sampling is None:
            a_length = float(torch.linalg.norm(self.cell[0]).item())
            b_length = float(torch.linalg.norm(self.cell[1]).item())
            sampling = (
                a_length / gpts[0],
                b_length / gpts[1],
            )
        self.sampling = sampling
        self.broadening_A = float(broadening_A)
        self.dtype = dtype
        self.device_hint = device

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

    def electrostatic_potential_3d(self) -> torch.Tensor:
        charge_fourier = -torch.fft.fftn(self.charge_density, dim=(-3, -2, -1))
        if self.positions_A is not None and self.atomic_numbers is not None:
            charge_fourier = add_point_charges_fourier(
                charge_fourier,
                self.positions_A,
                self.atomic_numbers,
                self.cell,
                broadening_A=self.broadening_A,
            )
        potential = -integrate_gradient_fourier(
            charge_fourier, self.cell, in_space="fourier", out_space="real"
        )
        potential = potential - potential.amin()
        return potential

    def forward(self) -> torch.Tensor:
        potential = self.electrostatic_potential_3d()
        slices = [
            interpolate_slice(
                potential,
                self.cell,
                gpts=self.gpts,
                sampling=self.sampling,
                a=a,
                b=b,
            )
            for a, b in self.slice_limits
        ]
        return torch.stack(slices, dim=0)
