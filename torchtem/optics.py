from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from torchtem.physics import energy2wavelength, polar_spatial_frequencies


@dataclass
class AberrationCoefficients:
    C10: float = 0.0
    C12: float = 0.0
    phi12: float = 0.0
    C21: float = 0.0
    phi21: float = 0.0
    C23: float = 0.0
    phi23: float = 0.0
    C30: float = 0.0
    C32: float = 0.0
    phi32: float = 0.0
    C34: float = 0.0
    phi34: float = 0.0
    C41: float = 0.0
    phi41: float = 0.0
    C43: float = 0.0
    phi43: float = 0.0
    C45: float = 0.0
    phi45: float = 0.0
    C50: float = 0.0
    C52: float = 0.0
    phi52: float = 0.0
    C54: float = 0.0
    phi54: float = 0.0
    C56: float = 0.0
    phi56: float = 0.0


class CTF(nn.Module):
    """Torch implementation of abTEM-style aberration and aperture transfer."""

    def __init__(
        self,
        *,
        energy_eV: float,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        semiangle_cutoff_mrad: float,
        soft_edge_mrad: float = 0.0,
        focal_spread_A: float = 0.0,
        angular_spread_mrad: float = 0.0,
        flip_phase: bool = False,
        wiener_snr: float = 0.0,
        aberrations: AberrationCoefficients | None = None,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        super().__init__()
        self.energy_eV = float(energy_eV)
        self.gpts = gpts
        self.sampling = sampling
        self.semiangle_cutoff_mrad = semiangle_cutoff_mrad
        self.soft_edge_mrad = soft_edge_mrad
        self.focal_spread_A = float(focal_spread_A)
        self.angular_spread_mrad = float(angular_spread_mrad)
        self.flip_phase = bool(flip_phase)
        self.wiener_snr = float(wiener_snr)
        self.dtype = dtype
        self.device_hint = device

        coeffs = aberrations or AberrationCoefficients()
        for name, value in coeffs.__dict__.items():
            self.register_parameter(name, nn.Parameter(torch.tensor(float(value), dtype=dtype)))

    @property
    def wavelength(self) -> torch.Tensor:
        return energy2wavelength(self.energy_eV, device=self.device_hint, dtype=self.dtype)

    def _angular_coordinates_rad(self) -> tuple[torch.Tensor, torch.Tensor]:
        k, phi = polar_spatial_frequencies(
            self.gpts, self.sampling, device=self.device_hint, dtype=self.dtype
        )
        alpha_rad = self.wavelength * k
        return alpha_rad, phi

    def evaluate(self, alpha_rad: torch.Tensor, phi_rad: torch.Tensor) -> torch.Tensor:
        alpha = torch.as_tensor(alpha_rad, dtype=self.dtype, device=self.device_hint)
        phi = torch.as_tensor(phi_rad, dtype=self.dtype, device=alpha.device)

        coeff = {name: getattr(self, name).to(alpha.device) for name in AberrationCoefficients.__annotations__}
        phase = torch.zeros_like(alpha)
        phase = phase + 0.5 * alpha.square() * (
            coeff["C10"] + coeff["C12"] * torch.cos(2.0 * (phi - coeff["phi12"]))
        )
        phase = phase + (alpha**3 / 3.0) * (
            coeff["C21"] * torch.cos(phi - coeff["phi21"])
            + coeff["C23"] * torch.cos(3.0 * (phi - coeff["phi23"]))
        )
        phase = phase + (alpha**4 / 4.0) * (
            coeff["C30"]
            + coeff["C32"] * torch.cos(2.0 * (phi - coeff["phi32"]))
            + coeff["C34"] * torch.cos(4.0 * (phi - coeff["phi34"]))
        )
        phase = phase + (alpha**5 / 5.0) * (
            coeff["C41"] * torch.cos(phi - coeff["phi41"])
            + coeff["C43"] * torch.cos(3.0 * (phi - coeff["phi43"]))
            + coeff["C45"] * torch.cos(5.0 * (phi - coeff["phi45"]))
        )
        phase = phase + (alpha**6 / 6.0) * (
            coeff["C50"]
            + coeff["C52"] * torch.cos(2.0 * (phi - coeff["phi52"]))
            + coeff["C54"] * torch.cos(4.0 * (phi - coeff["phi54"]))
            + coeff["C56"] * torch.cos(6.0 * (phi - coeff["phi56"]))
        )

        transfer = torch.exp(
            -1.0j * ((2.0 * torch.pi / self.wavelength.to(alpha.device)) * phase).to(torch.complex128)
        )

        if self.angular_spread_mrad != 0.0:
            angular_spread = torch.as_tensor(
                self.angular_spread_mrad / 1e3, dtype=self.dtype, device=alpha.device
            )
            dchi_dk = (
                2.0
                * torch.pi
                / self.wavelength.to(alpha.device)
                * (
                    (coeff["C12"] * torch.cos(2.0 * (phi - coeff["phi12"])) + coeff["C10"]) * alpha
                    + (
                        coeff["C23"] * torch.cos(3.0 * (phi - coeff["phi23"]))
                        + coeff["C21"] * torch.cos(phi - coeff["phi21"])
                    )
                    * alpha.square()
                    + (
                        coeff["C34"] * torch.cos(4.0 * (phi - coeff["phi34"]))
                        + coeff["C32"] * torch.cos(2.0 * (phi - coeff["phi32"]))
                        + coeff["C30"]
                    )
                    * alpha.pow(3)
                    + (
                        coeff["C45"] * torch.cos(5.0 * (phi - coeff["phi45"]))
                        + coeff["C43"] * torch.cos(3.0 * (phi - coeff["phi43"]))
                        + coeff["C41"] * torch.cos(phi - coeff["phi41"])
                    )
                    * alpha.pow(4)
                    + (
                        coeff["C56"] * torch.cos(6.0 * (phi - coeff["phi56"]))
                        + coeff["C54"] * torch.cos(4.0 * (phi - coeff["phi54"]))
                        + coeff["C52"] * torch.cos(2.0 * (phi - coeff["phi52"]))
                        + coeff["C50"]
                    )
                    * alpha.pow(5)
                )
            )
            dchi_dphi = (
                -2.0
                * torch.pi
                / self.wavelength.to(alpha.device)
                * (
                    coeff["C12"] * torch.sin(2.0 * (phi - coeff["phi12"])) * alpha
                    + (
                        coeff["C23"] * torch.sin(3.0 * (phi - coeff["phi23"]))
                        + coeff["C21"] / 3.0 * torch.sin(phi - coeff["phi21"])
                    )
                    * alpha.square()
                    + (
                        coeff["C34"] * torch.sin(4.0 * (phi - coeff["phi34"]))
                        + 0.5 * coeff["C32"] * torch.sin(2.0 * (phi - coeff["phi32"]))
                    )
                    * alpha.pow(3)
                    + (
                        coeff["C45"] * torch.sin(5.0 * (phi - coeff["phi45"]))
                        + 3.0 / 5.0 * coeff["C43"] * torch.sin(3.0 * (phi - coeff["phi43"]))
                        + 1.0 / 5.0 * coeff["C41"] * torch.sin(phi - coeff["phi41"])
                    )
                    * alpha.pow(4)
                    + (
                        coeff["C56"] * torch.sin(6.0 * (phi - coeff["phi56"]))
                        + 2.0 / 3.0 * coeff["C54"] * torch.sin(4.0 * (phi - coeff["phi54"]))
                        + 1.0 / 3.0 * coeff["C52"] * torch.sin(2.0 * (phi - coeff["phi52"]))
                    )
                    * alpha.pow(5)
                )
            )
            transfer = transfer * torch.exp(
                -torch.sign(angular_spread)
                * (angular_spread / 2.0).square()
                * (dchi_dk.square() + dchi_dphi.square())
            ).to(torch.complex128)

        if self.focal_spread_A != 0.0:
            focal_spread = torch.as_tensor(self.focal_spread_A, dtype=self.dtype, device=alpha.device)
            transfer = transfer * torch.exp(
                -((0.5 * torch.pi / self.wavelength.to(alpha.device) * focal_spread * alpha.square()) ** 2)
            ).to(torch.complex128)

        if self.semiangle_cutoff_mrad != float("inf"):
            cutoff_rad = torch.as_tensor(self.semiangle_cutoff_mrad / 1e3, dtype=self.dtype, device=alpha.device)
            if self.soft_edge_mrad <= 0.0:
                aperture = (torch.as_tensor(alpha_rad, dtype=self.dtype, device=alpha.device) <= cutoff_rad).to(torch.complex128)
            else:
                softness = torch.as_tensor(self.soft_edge_mrad / 1e3, dtype=self.dtype, device=alpha.device)
                aperture = torch.sigmoid((cutoff_rad - torch.as_tensor(alpha_rad, dtype=self.dtype, device=alpha.device)) / softness).to(torch.complex128)
            transfer = transfer * aperture

        if self.wiener_snr != 0.0:
            snr = torch.as_tensor(self.wiener_snr, dtype=self.dtype, device=alpha.device)
            transfer = ((1.0 + 1.0 / snr) * transfer.square()) / (transfer.square() + 1.0 / snr)
        elif self.flip_phase:
            transfer = transfer.real.to(torch.complex128) - 1.0j * transfer.imag.abs().to(torch.complex128)

        return transfer

    def angular_coordinates(self) -> tuple[torch.Tensor, torch.Tensor]:
        alpha_rad, phi = self._angular_coordinates_rad()
        return 1e3 * alpha_rad, phi

    def evaluate_on_grid(
        self,
        *,
        gpts: tuple[int, int] | None = None,
        sampling: tuple[float, float] | None = None,
    ) -> torch.Tensor:
        if gpts is None:
            gpts = self.gpts
        if sampling is None:
            sampling = self.sampling

        if gpts == self.gpts and sampling == self.sampling:
            return self()

        k, phi = polar_spatial_frequencies(
            gpts, sampling, device=self.device_hint, dtype=self.dtype
        )
        alpha_rad = self.wavelength * k
        return self.evaluate(alpha_rad, phi)

    def aberration_phase(self) -> torch.Tensor:
        alpha, phi = self._angular_coordinates_rad()
        coeff = {name: getattr(self, name) for name in AberrationCoefficients.__annotations__}
        phase = torch.zeros_like(alpha)
        phase = phase + 0.5 * alpha.square() * (
            coeff["C10"] + coeff["C12"] * torch.cos(2.0 * (phi - coeff["phi12"]))
        )
        phase = phase + (alpha**3 / 3.0) * (
            coeff["C21"] * torch.cos(phi - coeff["phi21"])
            + coeff["C23"] * torch.cos(3.0 * (phi - coeff["phi23"]))
        )
        phase = phase + (alpha**4 / 4.0) * (
            coeff["C30"]
            + coeff["C32"] * torch.cos(2.0 * (phi - coeff["phi32"]))
            + coeff["C34"] * torch.cos(4.0 * (phi - coeff["phi34"]))
        )
        phase = phase + (alpha**5 / 5.0) * (
            coeff["C41"] * torch.cos(phi - coeff["phi41"])
            + coeff["C43"] * torch.cos(3.0 * (phi - coeff["phi43"]))
            + coeff["C45"] * torch.cos(5.0 * (phi - coeff["phi45"]))
        )
        phase = phase + (alpha**6 / 6.0) * (
            coeff["C50"]
            + coeff["C52"] * torch.cos(2.0 * (phi - coeff["phi52"]))
            + coeff["C54"] * torch.cos(4.0 * (phi - coeff["phi54"]))
            + coeff["C56"] * torch.cos(6.0 * (phi - coeff["phi56"]))
        )
        return (2.0 * torch.pi / self.wavelength) * phase

    def temporal_envelope(self) -> torch.Tensor:
        alpha, _ = self._angular_coordinates_rad()
        focal_spread = torch.as_tensor(
            self.focal_spread_A, dtype=self.dtype, device=alpha.device
        )
        return torch.exp(
            -((0.5 * torch.pi / self.wavelength * focal_spread * alpha.square()) ** 2)
        )

    def spatial_envelope(self) -> torch.Tensor:
        alpha, phi = self._angular_coordinates_rad()
        angular_spread = torch.as_tensor(
            self.angular_spread_mrad / 1e3, dtype=self.dtype, device=alpha.device
        )
        coeff = {name: getattr(self, name) for name in AberrationCoefficients.__annotations__}

        dchi_dk = (
            2.0
            * torch.pi
            / self.wavelength
            * (
                (coeff["C12"] * torch.cos(2.0 * (phi - coeff["phi12"])) + coeff["C10"]) * alpha
                + (
                    coeff["C23"] * torch.cos(3.0 * (phi - coeff["phi23"]))
                    + coeff["C21"] * torch.cos(phi - coeff["phi21"])
                )
                * alpha.square()
                + (
                    coeff["C34"] * torch.cos(4.0 * (phi - coeff["phi34"]))
                    + coeff["C32"] * torch.cos(2.0 * (phi - coeff["phi32"]))
                    + coeff["C30"]
                )
                * alpha.pow(3)
                + (
                    coeff["C45"] * torch.cos(5.0 * (phi - coeff["phi45"]))
                    + coeff["C43"] * torch.cos(3.0 * (phi - coeff["phi43"]))
                    + coeff["C41"] * torch.cos(phi - coeff["phi41"])
                )
                * alpha.pow(4)
                + (
                    coeff["C56"] * torch.cos(6.0 * (phi - coeff["phi56"]))
                    + coeff["C54"] * torch.cos(4.0 * (phi - coeff["phi54"]))
                    + coeff["C52"] * torch.cos(2.0 * (phi - coeff["phi52"]))
                    + coeff["C50"]
                )
                * alpha.pow(5)
            )
        )

        dchi_dphi = (
            -2.0
            * torch.pi
            / self.wavelength
            * (
                coeff["C12"] * torch.sin(2.0 * (phi - coeff["phi12"])) * alpha
                + (
                    coeff["C23"] * torch.sin(3.0 * (phi - coeff["phi23"]))
                    + coeff["C21"] / 3.0 * torch.sin(phi - coeff["phi21"])
                )
                * alpha.square()
                + (
                    coeff["C34"] * torch.sin(4.0 * (phi - coeff["phi34"]))
                    + 0.5 * coeff["C32"] * torch.sin(2.0 * (phi - coeff["phi32"]))
                )
                * alpha.pow(3)
                + (
                    coeff["C45"] * torch.sin(5.0 * (phi - coeff["phi45"]))
                    + 3.0 / 5.0 * coeff["C43"] * torch.sin(3.0 * (phi - coeff["phi43"]))
                    + 1.0 / 5.0 * coeff["C41"] * torch.sin(phi - coeff["phi41"])
                )
                * alpha.pow(4)
                + (
                    coeff["C56"] * torch.sin(6.0 * (phi - coeff["phi56"]))
                    + 2.0 / 3.0 * coeff["C54"] * torch.sin(4.0 * (phi - coeff["phi54"]))
                    + 1.0 / 3.0 * coeff["C52"] * torch.sin(2.0 * (phi - coeff["phi52"]))
                )
                * alpha.pow(5)
            )
        )

        return torch.exp(
            -torch.sign(angular_spread) * (angular_spread / 2.0).square() * (dchi_dk.square() + dchi_dphi.square())
        )

    def aperture(self) -> torch.Tensor:
        alpha, _ = self.angular_coordinates()
        if self.soft_edge_mrad <= 0.0:
            return (alpha <= self.semiangle_cutoff_mrad).to(self.dtype)
        softness = torch.as_tensor(self.soft_edge_mrad, dtype=self.dtype, device=alpha.device)
        cutoff = torch.as_tensor(self.semiangle_cutoff_mrad, dtype=self.dtype, device=alpha.device)
        return torch.sigmoid((cutoff - alpha) / softness)

    def forward(self) -> torch.Tensor:
        phase = self.aberration_phase()
        transfer = torch.exp(-1.0j * phase.to(torch.complex128))
        transfer = transfer * self.aperture().to(torch.complex128)
        if self.angular_spread_mrad != 0.0:
            transfer = transfer * self.spatial_envelope().to(torch.complex128)
        if self.focal_spread_A != 0.0:
            transfer = transfer * self.temporal_envelope().to(torch.complex128)
        if self.wiener_snr != 0.0:
            snr = torch.as_tensor(self.wiener_snr, dtype=self.dtype, device=phase.device)
            transfer = ((1.0 + 1.0 / snr) * transfer.square()) / (transfer.square() + 1.0 / snr)
        elif self.flip_phase:
            transfer = transfer.real.to(torch.complex128) - 1.0j * transfer.imag.abs().to(torch.complex128)
        return transfer


class Probe(nn.Module):
    """Build a convergent probe from a reciprocal-space CTF."""

    def __init__(self, ctf: CTF) -> None:
        super().__init__()
        self.ctf = ctf

    def forward(self, amplitude: torch.Tensor | None = None) -> torch.Tensor:
        reciprocal = self.ctf()
        if amplitude is not None:
            reciprocal = reciprocal * amplitude.to(reciprocal.dtype)
        norm = torch.sqrt(torch.sum(torch.abs(reciprocal).square()))
        reciprocal = reciprocal / norm.clamp_min(torch.finfo(self.ctf.dtype).eps)
        return torch.fft.ifft2(reciprocal)


class PlaneWave(nn.Module):
    """Build a uniform incident plane wave on the simulation grid."""

    def __init__(
        self,
        *,
        gpts: tuple[int, int],
        amplitude: complex = 1.0 + 0.0j,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        super().__init__()
        self.gpts = gpts
        self.amplitude = complex(amplitude)
        self.dtype = dtype
        self.device_hint = device

    def forward(self) -> torch.Tensor:
        amplitude = torch.as_tensor(self.amplitude, dtype=torch.complex128, device=self.device_hint)
        return torch.full(self.gpts, amplitude, dtype=torch.complex128, device=self.device_hint)
