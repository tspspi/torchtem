from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence

import torch
from torch import nn

from tspi.torchtem.assembly import (
    BatchDetectorReadout,
    DetectorReadout,
    LayerStack,
    PotentialSliceInteraction,
    ScanLayer,
    apply_detector_recursive,
)
from tspi.torchtem.bloch import BlochWaveModel, StructureFactorModel, calculate_g_vec
from tspi.torchtem.coherence import CoherenceAverager, CoherenceEnsemble
from tspi.torchtem.detectors import (
    AnnularDetector,
    FlexibleAnnularDetector,
    ImageDetector,
    PixelatedDetector,
    SegmentedDetector,
    WavesDetector,
)
from tspi.torchtem.fft_support import fft_interpolate
from tspi.torchtem.inelastic import PlasmonScatteringEvents
from tspi.torchtem.magnetism import PauliVectorPotentialInteraction
from tspi.torchtem.mcf import DiagonalMCF
from tspi.torchtem.multislice import MultisliceSystem
from tspi.torchtem.optics import AberrationCoefficients, CTF, PlaneWave, Probe
from tspi.torchtem.pipelines import (
    CenterOfMassConfig,
    GaussianBlurConfig,
    IntegrateGradientConfig,
    MTFConfig,
    PipelineStepConfig,
    PoissonNoiseConfig,
    ScanDistortionConfig,
    build_measurement_pipeline,
)
from tspi.torchtem.realspace_multislice import RealSpaceMultisliceSystem
from tspi.torchtem.results import SimulationResult, SimulationSeriesResult
from tspi.torchtem.physics import spatial_frequencies
from tspi.torchtem.prism import SMatrixBuilder, smatrix_wave_vectors
from tspi.torchtem.scan import CustomScan, GridScan, fft_shift_wave
from tspi.torchtem.tilt import apply_beam_tilt


@dataclass
class PlaneWaveSourceConfig:
    gpts: tuple[int, int]
    sampling: tuple[float, float]
    energy_eV: float
    amplitude: complex = 1.0 + 0.0j
    tilt_mrad: tuple[float, float] = (0.0, 0.0)


@dataclass
class ProbeSourceConfig:
    energy_eV: float
    gpts: tuple[int, int]
    sampling: tuple[float, float]
    semiangle_cutoff_mrad: float
    soft_edge_mrad: float = 0.0
    focal_spread_A: float = 0.0
    angular_spread_mrad: float = 0.0
    flip_phase: bool = False
    wiener_snr: float = 0.0
    aberrations: AberrationCoefficients | None = None
    tilt_mrad: tuple[float, float] = (0.0, 0.0)


@dataclass
class MultisliceConfig:
    energy_eV: float
    gpts: tuple[int, int]
    sampling: tuple[float, float]
    slice_thickness_A: float
    num_slices: int = 1
    backend: str = "fourier"
    series_order: int = 1
    max_terms: int = 12
    tolerance: float = 1e-12
    prism_interpolation: int = 1
    prism_cutoff_mrad: float | None = None


@dataclass
class GridScanConfig:
    start_A: tuple[float, float] | float
    end_A: tuple[float, float] | float
    shape: tuple[int, int]


@dataclass
class CustomScanConfig:
    positions_A: torch.Tensor


@dataclass
class AnnularDetectorConfig:
    inner_mrad: float
    outer_mrad: float
    offset_mrad: tuple[float, float] = (0.0, 0.0)


@dataclass
class FlexibleAnnularDetectorConfig:
    step_size_mrad: float = 1.0
    inner_mrad: float = 0.0
    outer_mrad: float | None = None
    offset_mrad: tuple[float, float] = (0.0, 0.0)


@dataclass
class PixelatedDetectorConfig:
    reciprocal_space: bool = True
    gpts: tuple[int, int] | None = None
    max_angle_mrad: float | None = None


@dataclass
class ImageDetectorConfig:
    output: str = "intensity"
    semiangle_cutoff_mrad: float | None = None
    soft_edge_mrad: float = 0.0
    focal_spread_A: float = 0.0
    angular_spread_mrad: float = 0.0
    flip_phase: bool = False
    wiener_snr: float = 0.0
    aberrations: AberrationCoefficients | None = None


@dataclass
class WavesDetectorConfig:
    gpts: tuple[int, int] | None = None
    reciprocal_space: bool = False


@dataclass
class SegmentedDetectorConfig:
    inner_mrad: float
    outer_mrad: float
    nbins_radial: int
    nbins_azimuthal: int
    rotation_rad: float = 0.0
    offset_mrad: tuple[float, float] = (0.0, 0.0)


@dataclass
class CoherenceConfig:
    mode: str = "ensemble"
    defocus_offsets_A: torch.Tensor | None = None
    source_offsets_A: torch.Tensor | None = None
    weights: torch.Tensor | None = None
    eigenvectors: int | tuple[int, ...] | None = None
    focal_spread_A: float = 0.0
    source_size_A: float = 0.0
    rectangular_offset_A: tuple[float, float] = (0.0, 0.0)


@dataclass
class InelasticConfig:
    operator: nn.Module
    mode: str = "channels"


@dataclass
class MagneticConfig:
    vector_potential: nn.Module
    mode: str = "pauli"
    prefactor: complex = 1.0j


@dataclass
class BlochConfig:
    hkl: torch.Tensor
    cell: torch.Tensor
    structure_factors: torch.Tensor
    thickness_A: float
    use_wave_eq: bool = False


@dataclass(frozen=True)
class ParameterVectorEntry:
    name: str
    shape: tuple[int, ...]
    start: int
    stop: int


@dataclass
class ExperimentConfig:
    source: PlaneWaveSourceConfig | ProbeSourceConfig
    multislice: MultisliceConfig
    potential: nn.Module | None
    bloch: BlochConfig | None = None
    detector: (
        AnnularDetectorConfig
        | FlexibleAnnularDetectorConfig
        | PixelatedDetectorConfig
        | ImageDetectorConfig
        | WavesDetectorConfig
        | SegmentedDetectorConfig
        | Mapping[
            str,
            AnnularDetectorConfig
            | FlexibleAnnularDetectorConfig
            | PixelatedDetectorConfig
            | ImageDetectorConfig
            | WavesDetectorConfig
            | SegmentedDetectorConfig,
        ]
        | None
    ) = None
    scan: GridScanConfig | CustomScanConfig | None = None
    coherence: CoherenceConfig | None = None
    postprocess: Sequence[PipelineStepConfig] | Mapping[str, Sequence[PipelineStepConfig]] | None = None
    inelastic: InelasticConfig | None = None
    magnetic: MagneticConfig | None = None


@dataclass
class HRTEMConfig:
    source: PlaneWaveSourceConfig | ProbeSourceConfig
    multislice: MultisliceConfig
    potential: nn.Module | None
    bloch: BlochConfig | None = None
    detector: (
        PixelatedDetectorConfig
        | ImageDetectorConfig
        | WavesDetectorConfig
        | Mapping[
            str,
            AnnularDetectorConfig
            | FlexibleAnnularDetectorConfig
            | PixelatedDetectorConfig
            | ImageDetectorConfig
            | WavesDetectorConfig
            | SegmentedDetectorConfig,
        ]
        | None
    ) = None
    coherence: CoherenceConfig | None = None
    postprocess: Sequence[PipelineStepConfig] | Mapping[str, Sequence[PipelineStepConfig]] | None = None
    inelastic: InelasticConfig | None = None
    magnetic: MagneticConfig | None = None


@dataclass
class STEMConfig:
    source: ProbeSourceConfig
    multislice: MultisliceConfig
    potential: nn.Module | None
    detector: (
        AnnularDetectorConfig
        | FlexibleAnnularDetectorConfig
        | PixelatedDetectorConfig
        | ImageDetectorConfig
        | WavesDetectorConfig
        | SegmentedDetectorConfig
        | Mapping[
            str,
            AnnularDetectorConfig
            | FlexibleAnnularDetectorConfig
            | PixelatedDetectorConfig
            | ImageDetectorConfig
            | WavesDetectorConfig
            | SegmentedDetectorConfig,
        ]
    )
    scan: GridScanConfig | CustomScanConfig
    bloch: BlochConfig | None = None
    coherence: CoherenceConfig | None = None
    postprocess: Sequence[PipelineStepConfig] | Mapping[str, Sequence[PipelineStepConfig]] | None = None
    inelastic: InelasticConfig | None = None
    magnetic: MagneticConfig | None = None


@dataclass
class DiffractionConfig:
    source: PlaneWaveSourceConfig | ProbeSourceConfig
    multislice: MultisliceConfig
    potential: nn.Module | None = None
    bloch: BlochConfig | None = None
    detector: PixelatedDetectorConfig | ImageDetectorConfig | WavesDetectorConfig = field(default_factory=PixelatedDetectorConfig)
    scan: GridScanConfig | CustomScanConfig | None = None
    coherence: CoherenceConfig | None = None
    postprocess: Sequence[PipelineStepConfig] | Mapping[str, Sequence[PipelineStepConfig]] | None = None
    inelastic: InelasticConfig | None = None
    magnetic: MagneticConfig | None = None


class NamedDetectorReadout(nn.Module):
    """Apply multiple named detector branches to one wave or wave batch."""

    def __init__(
        self,
        detectors: Mapping[str, nn.Module],
        *,
        batched: bool,
    ) -> None:
        super().__init__()
        self.detectors = nn.ModuleDict(dict(detectors))
        self.batched = bool(batched)

    def _apply_single(self, wave: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: apply_detector_recursive(detector, wave) for name, detector in self.detectors.items()}

    def forward(self, waves: torch.Tensor) -> dict[str, torch.Tensor]:
        if (not self.batched) or waves.ndim == 2:
            return self._apply_single(waves)
        return self._apply_single(waves)


class NamedPostprocess(nn.Module):
    """Apply named postprocessing pipelines to matching named outputs."""

    def __init__(self, pipelines: Mapping[str, nn.Module]) -> None:
        super().__init__()
        self.pipelines = nn.ModuleDict(dict(pipelines))

    def forward(self, outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            name: self.pipelines[name](value) if name in self.pipelines else value
            for name, value in outputs.items()
        }


class InelasticInteraction(nn.Module):
    """Apply an inelastic operator to a propagated elastic wave."""

    def __init__(self, config: InelasticConfig) -> None:
        super().__init__()
        self.operator = config.operator
        self.mode = str(config.mode)

    def _apply_single(self, wave: torch.Tensor) -> torch.Tensor:
        if self.mode == "channels":
            return self.operator(wave)
        if self.mode == "weighted_intensity":
            if not hasattr(self.operator, "weighted_intensity"):
                raise ValueError("weighted_intensity mode requires an operator with weighted_intensity(...)")
            return self.operator.weighted_intensity(wave)
        raise ValueError(f"Unknown inelastic mode: {self.mode}")

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        if wave.ndim == 2:
            return self._apply_single(wave)
        return torch.stack([self._apply_single(wave[i]) for i in range(wave.shape[0])], dim=0)


class MagneticInteraction(nn.Module):
    """Apply a projected magnetic interaction to an elastic wave or wave batch."""

    def __init__(self, config: MagneticConfig, *, sampling: tuple[float, float]) -> None:
        super().__init__()
        self.vector_potential = config.vector_potential
        self.mode = str(config.mode)
        if self.mode != "pauli":
            raise ValueError(f"Unknown magnetic mode: {self.mode}")
        self.interaction = PauliVectorPotentialInteraction(
            sampling=sampling,
            prefactor=config.prefactor,
        )

    def _apply_single(self, wave: torch.Tensor) -> torch.Tensor:
        vector_potential = self.vector_potential()
        return self.interaction(vector_potential, wave)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        if wave.ndim == 2:
            return self._apply_single(wave)
        return torch.stack([self._apply_single(wave[i]) for i in range(wave.shape[0])], dim=0)


class PrismScanInteraction(nn.Module):
    """Build and scan a PRISM S-matrix from projected potential slices."""

    def __init__(
        self,
        *,
        probe: Probe,
        potential: nn.Module,
        scan: nn.Module | None,
        config: MultisliceConfig,
        num_slices: int,
    ) -> None:
        super().__init__()
        self.probe = probe
        self.potential = potential
        self.scan = scan
        self.config = config
        self.num_slices = int(num_slices)
        cutoff = (
            float(config.prism_cutoff_mrad)
            if config.prism_cutoff_mrad is not None
            else float(probe.ctf.semiangle_cutoff_mrad)
        )
        extent_A = (config.gpts[0] * config.sampling[0], config.gpts[1] * config.sampling[1])
        parent_wave_vectors = smatrix_wave_vectors(
            cutoff,
            extent_A,
            config.gpts,
            config.energy_eV,
            interpolation=config.prism_interpolation,
        )
        self.register_buffer("parent_wave_vectors", parent_wave_vectors)
        self.smatrix_builder = SMatrixBuilder(
            multislice_system=MultisliceSystem(
                energy_eV=config.energy_eV,
                gpts=config.gpts,
                sampling=config.sampling,
                slice_thickness_A=config.slice_thickness_A,
            ),
            parent_wave_vectors=parent_wave_vectors,
            extent_A=extent_A,
            gpts=config.gpts,
            energy_eV=config.energy_eV,
            interpolation=config.prism_interpolation,
        )

    def potential_slices(self) -> torch.Tensor:
        potential = self.potential()
        if potential.ndim == 3:
            return potential
        return potential.unsqueeze(0).repeat(self.num_slices, 1, 1) / self.num_slices

    def ctf_coefficients(self) -> torch.Tensor:
        transfer = self.probe.ctf()
        ky, kx = spatial_frequencies(
            self.config.gpts,
            self.config.sampling,
            device=transfer.device,
            dtype=torch.float64,
        )
        coeffs = []
        for vec in self.parent_wave_vectors:
            iy = torch.argmin(torch.abs(ky - vec[0]))
            ix = torch.argmin(torch.abs(kx - vec[1]))
            coeffs.append(transfer[iy, ix])
        return torch.stack(coeffs, dim=0)

    def forward(self, incident_wave: torch.Tensor) -> torch.Tensor:
        del incident_wave
        smatrix = self.smatrix_builder(self.potential_slices())
        coefficients = self.ctf_coefficients()
        if self.scan is None:
            return smatrix.reduce_to_waves(coefficients.to(torch.complex128))
        positions = self.scan().to(device=smatrix.array.device, dtype=torch.float64)
        return smatrix.scan(positions, coefficients)


class BlochDiffractionPattern(nn.Module):
    """Generate a diffraction-pattern image from the differentiable Bloch model."""

    def __init__(
        self,
        *,
        config: BlochConfig,
        energy_eV: float,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
    ) -> None:
        super().__init__()
        structure_factor_model = StructureFactorModel(
            config.hkl,
            config.structure_factors,
            config.cell,
        )
        self.model = BlochWaveModel(
            hkl=config.hkl,
            cell=config.cell,
            structure_factor_model=structure_factor_model,
            energy_eV=energy_eV,
            use_wave_eq=config.use_wave_eq,
        )
        self.register_buffer("hkl", config.hkl.to(torch.int64))
        self.register_buffer("cell", config.cell.to(torch.float64))
        self.thickness_A = float(config.thickness_A)
        self.gpts = gpts
        self.sampling = sampling

    def forward(self, incident_wave: torch.Tensor) -> torch.Tensor:
        del incident_wave
        coeffs = self.model.dynamical_scattering(self.thickness_A)
        return self._rasterize_intensity(coeffs)

    def _projected_reciprocal_indices(self, coeffs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g = calculate_g_vec(self.hkl.to(torch.float64), self.cell)
        ky_axis, kx_axis = spatial_frequencies(
            self.gpts,
            self.sampling,
            device=coeffs.device,
            dtype=torch.float64,
        )
        iy = torch.stack([torch.argmin(torch.abs(ky_axis - gy)) for gy in g[:, 0]])
        ix = torch.stack([torch.argmin(torch.abs(kx_axis - gx)) for gx in g[:, 1]])
        return iy, ix

    def _rasterize_intensity(self, coeffs: torch.Tensor) -> torch.Tensor:
        image = torch.zeros(self.gpts, device=coeffs.device, dtype=torch.float64)
        intensity = coeffs.abs().square().to(torch.float64)
        iy, ix = self._projected_reciprocal_indices(coeffs)
        flat_index = iy * self.gpts[1] + ix
        image = image.reshape(-1)
        image.scatter_add_(0, flat_index, intensity)
        return image.reshape(self.gpts)


class BlochReciprocalWaves(nn.Module):
    """Rasterize Bloch scattering amplitudes onto the reciprocal-space image grid."""

    def __init__(
        self,
        *,
        config: BlochConfig,
        energy_eV: float,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
    ) -> None:
        super().__init__()
        self.pattern = BlochDiffractionPattern(
            config=config,
            energy_eV=energy_eV,
            gpts=gpts,
            sampling=sampling,
        )

    def forward(self, incident_wave: torch.Tensor) -> torch.Tensor:
        del incident_wave
        coeffs = self.pattern.model.dynamical_scattering(self.pattern.thickness_A)
        iy, ix = self.pattern._projected_reciprocal_indices(coeffs)
        image = torch.zeros(self.pattern.gpts, device=coeffs.device, dtype=torch.complex128)
        flat_index = iy * self.pattern.gpts[1] + ix
        image = image.reshape(-1)
        image.scatter_add_(0, flat_index, coeffs.to(torch.complex128))
        return image.reshape(self.pattern.gpts)


class CoherenceReadout(nn.Module):
    """Apply coherence averaging to a single wave or a leading batch of waves."""

    def __init__(
        self,
        *,
        coherence: CoherenceAverager,
        sampling: tuple[float, float],
        detector: nn.Module | Mapping[str, nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.coherence = coherence
        self.sampling = sampling
        if detector is None or isinstance(detector, nn.Module):
            self.detector = detector
            self.named_detectors = None
        else:
            self.detector = None
            self.named_detectors = nn.ModuleDict(dict(detector))

    def _apply_single(self, wave: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        coherent_members = self.coherence.apply_source_offsets(wave, self.sampling)
        if self.named_detectors is None:
            return self.coherence.average_intensity(coherent_members, self.detector)
        return {
            name: self.coherence.average_intensity(coherent_members, detector)
            for name, detector in self.named_detectors.items()
        }

    def forward(self, waves: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        if waves.ndim == 2:
            return self._apply_single(waves)
        if self.named_detectors is None:
            return torch.stack([self._apply_single(waves[i]) for i in range(waves.shape[0])], dim=0)

        outputs = [self._apply_single(waves[i]) for i in range(waves.shape[0])]
        return {
            name: torch.stack([output[name] for output in outputs], dim=0)
            for name in outputs[0]
        }


class DiagonalMCFSource(nn.Module):
    """Generate coherent source modes from a diagonal mixed-coherence model."""

    def __init__(self, mcf: DiagonalMCF) -> None:
        super().__init__()
        self.mcf = mcf

    def forward(self) -> torch.Tensor:
        return self.mcf.real_space_modes()


class ProbeCoherenceEnsembleSource(nn.Module):
    """Generate source-side coherent probe members from defocus/source ensembles."""

    def __init__(self, probe: Probe, ensemble: CoherenceEnsemble) -> None:
        super().__init__()
        self.probe = probe
        if ensemble.defocus_offsets_A is not None:
            self.register_buffer("defocus_offsets_A", ensemble.defocus_offsets_A.to(torch.float64))
        else:
            self.defocus_offsets_A = None
        if ensemble.source_offsets_A is not None:
            self.register_buffer("source_offsets_A", ensemble.source_offsets_A.to(torch.float64))
        else:
            self.source_offsets_A = None

    def _num_members(self) -> int:
        n = 1
        if self.defocus_offsets_A is not None:
            n = max(n, int(self.defocus_offsets_A.shape[0]))
        if self.source_offsets_A is not None:
            n = max(n, int(self.source_offsets_A.shape[0]))
        return n

    def _broadcast_members(self, values: torch.Tensor | None, width: int) -> torch.Tensor:
        members = self._num_members()
        if values is None:
            return torch.zeros((members, width), dtype=torch.float64, device=self.probe.ctf.wavelength.device)
        if values.ndim == 1:
            values = values[:, None]
        if values.shape[0] == members:
            return values
        if values.shape[0] == 1:
            return values.expand(members, width)
        raise ValueError("Coherence ensemble member counts must match or be singleton-expandable")

    def _apply_member_source_offsets(
        self,
        reciprocal: torch.Tensor,
        source_offsets_A: torch.Tensor,
    ) -> torch.Tensor:
        if torch.all(source_offsets_A == 0):
            return torch.fft.ifft2(reciprocal)

        ky, kx = spatial_frequencies(
            self.probe.ctf.gpts,
            self.probe.ctf.sampling,
            device=reciprocal.device,
            dtype=torch.float64,
        )
        ky, kx = torch.meshgrid(ky, kx, indexing="ij")
        phase = torch.exp(
            -2.0j
            * torch.pi
            * (
                source_offsets_A[:, 0, None, None].to(ky.dtype) * ky[None]
                + source_offsets_A[:, 1, None, None].to(kx.dtype) * kx[None]
            )
        )
        return torch.fft.ifft2(reciprocal * phase)

    def forward(self) -> torch.Tensor:
        members = self._num_members()
        defocus_offsets = self._broadcast_members(self.defocus_offsets_A, 1).squeeze(-1)
        source_offsets = self._broadcast_members(self.source_offsets_A, 2)

        reciprocal = self.probe.ctf()
        reciprocal = reciprocal.unsqueeze(0).expand(members, *reciprocal.shape)

        if torch.any(defocus_offsets != 0):
            alpha_rad, _ = self.probe.ctf._angular_coordinates_rad()
            wavelength = self.probe.ctf.wavelength.to(device=reciprocal.device, dtype=self.probe.ctf.dtype)
            delta_C10 = -defocus_offsets.to(device=reciprocal.device, dtype=self.probe.ctf.dtype)
            delta_phase = (
                2.0
                * torch.pi
                / wavelength
                * 0.5
                * alpha_rad.square().unsqueeze(0)
                * delta_C10[:, None, None]
            )
            reciprocal = reciprocal * torch.exp(-1.0j * delta_phase.to(torch.complex128))

        return self._apply_member_source_offsets(reciprocal, source_offsets.to(device=reciprocal.device))


class SourceTiltLayer(nn.Module):
    """Apply a microscope beam tilt to a source wave or coherent source-mode stack."""

    def __init__(
        self,
        *,
        energy_eV: float,
        sampling: tuple[float, float],
        tilt_mrad: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        super().__init__()
        self.energy_eV = float(energy_eV)
        self.sampling = sampling
        self.register_parameter(
            "tilt_y_mrad",
            nn.Parameter(torch.tensor(float(tilt_mrad[0]), dtype=torch.float64)),
        )
        self.register_parameter(
            "tilt_x_mrad",
            nn.Parameter(torch.tensor(float(tilt_mrad[1]), dtype=torch.float64)),
        )

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        tilt = torch.stack((self.tilt_y_mrad, self.tilt_x_mrad))
        return apply_beam_tilt(
            wave,
            tilt_mrad=tilt,
            energy_eV=self.energy_eV,
            sampling=self.sampling,
        )


class StaticWaveShiftLayer(nn.Module):
    """Apply one fixed real-space shift to a wave or coherent source-mode stack."""

    def __init__(
        self,
        *,
        position_A: tuple[float, float],
        sampling: tuple[float, float],
    ) -> None:
        super().__init__()
        self.sampling = sampling
        self.register_parameter(
            "position_A",
            nn.Parameter(torch.tensor(position_A, dtype=torch.float64)),
        )

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        return fft_shift_wave(wave, self.position_A, self.sampling)


class ModeCoherenceReadout(nn.Module):
    """Reduce propagated coherent modes to incoherent detector outputs."""

    def __init__(
        self,
        detector: nn.Module | Mapping[str, nn.Module] | None = None,
        *,
        weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if detector is None or isinstance(detector, nn.Module):
            self.detector = detector
            self.named_detectors = None
        else:
            self.detector = None
            self.named_detectors = nn.ModuleDict(dict(detector))
        if weights is not None:
            self.register_buffer("weights", weights.to(torch.float64) / weights.to(torch.float64).sum())
        else:
            self.weights = None

    def _reduce_outputs(self, outputs: torch.Tensor, mode_dim: int) -> torch.Tensor:
        if self.weights is None:
            return outputs.sum(dim=mode_dim)
        shape = [1] * outputs.ndim
        shape[mode_dim] = self.weights.shape[0]
        return (outputs * self.weights.to(outputs.dtype).reshape(shape)).sum(dim=mode_dim)

    def forward(self, waves: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.named_detectors is None:
            if self.detector is None:
                outputs = waves.abs().square()
            else:
                outputs = apply_detector_recursive(self.detector, waves)
            mode_dim = max(waves.ndim - 3, 0)
            return self._reduce_outputs(outputs, mode_dim)
        mode_dim = max(waves.ndim - 3, 0)
        return {
            name: self._reduce_outputs(apply_detector_recursive(detector, waves), mode_dim)
            for name, detector in self.named_detectors.items()
        }


class BlochDetectorReadout(nn.Module):
    """Apply reciprocal-intensity detector geometry to a Bloch diffraction image."""

    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        self.detector = detector

    def _dummy_wave(self, image: torch.Tensor) -> torch.Tensor:
        return torch.zeros(image.shape[-2:], device=image.device, dtype=torch.complex128)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        dummy = self._dummy_wave(image)
        intensity = image.abs().square() if torch.is_complex(image) else image

        if isinstance(self.detector, PixelatedDetector):
            if isinstance(self.detector, WavesDetector):
                raise ValueError("Internal detector dispatch error")
            if not self.detector.reciprocal_space:
                raise ValueError("Bloch reciprocal-image readout only supports reciprocal-space pixelated detectors")
            output = intensity
            if self.detector.max_angle_mrad is not None:
                output = output * self.detector.detector_region(dummy).to(output.dtype)
            if self.detector.gpts is not None and tuple(output.shape[-2:]) != self.detector.gpts:
                output = torch.nn.functional.interpolate(
                    output.unsqueeze(0).unsqueeze(0),
                    size=self.detector.gpts,
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
            return output

        if isinstance(self.detector, WavesDetector):
            if not self.detector.reciprocal_space:
                raise ValueError("Bloch reciprocal-wave readout only supports reciprocal_space=True")
            output = image.to(torch.complex128)
            if self.detector.gpts is not None and tuple(output.shape[-2:]) != self.detector.gpts:
                output = fft_interpolate(output, self.detector.gpts, normalization="values")
            return output

        if isinstance(self.detector, AnnularDetector):
            return torch.sum(intensity * self.detector.detector_region(dummy).to(intensity.dtype))

        if isinstance(self.detector, FlexibleAnnularDetector):
            bins = self.detector.radial_bin_map(dummy)
            nbins = self.detector.nbins_radial(dummy)
            out = torch.zeros(nbins, device=image.device, dtype=intensity.dtype)
            valid = bins >= 0
            out.scatter_add_(0, bins[valid], intensity[valid])
            return out

        if isinstance(self.detector, SegmentedDetector):
            bins = self.detector.bins(dummy)
            out = torch.zeros(
                self.detector.nbins_radial * self.detector.nbins_azimuthal,
                device=image.device,
                dtype=intensity.dtype,
            )
            valid = bins >= 0
            out.scatter_add_(0, bins[valid], intensity[valid])
            return out.reshape(self.detector.nbins_radial, self.detector.nbins_azimuthal)

        raise ValueError(f"Unsupported Bloch detector type: {type(self.detector).__name__}")


class NamedBlochDetectorReadout(nn.Module):
    """Apply multiple reciprocal-intensity detector branches to one Bloch diffraction image."""

    def __init__(self, detectors: Mapping[str, nn.Module]) -> None:
        super().__init__()
        self.detectors = nn.ModuleDict(
            {name: BlochDetectorReadout(detector) for name, detector in detectors.items()}
        )

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: detector(image) for name, detector in self.detectors.items()}


def build_source(
    config: PlaneWaveSourceConfig | ProbeSourceConfig,
) -> nn.Module:
    if isinstance(config, PlaneWaveSourceConfig):
        return PlaneWave(
            gpts=config.gpts,
            amplitude=config.amplitude,
        )

    ctf = CTF(
        energy_eV=config.energy_eV,
        gpts=config.gpts,
        sampling=config.sampling,
        semiangle_cutoff_mrad=config.semiangle_cutoff_mrad,
        soft_edge_mrad=config.soft_edge_mrad,
        focal_spread_A=config.focal_spread_A,
        angular_spread_mrad=config.angular_spread_mrad,
        flip_phase=config.flip_phase,
        wiener_snr=config.wiener_snr,
        aberrations=config.aberrations,
    )
    return Probe(ctf)


def source_tilt_from_config(
    config: PlaneWaveSourceConfig | ProbeSourceConfig,
) -> tuple[float, float]:
    return (float(config.tilt_mrad[0]), float(config.tilt_mrad[1]))


def build_scan(config: GridScanConfig | CustomScanConfig) -> nn.Module:
    if isinstance(config, GridScanConfig):
        return GridScan(start_A=config.start_A, end_A=config.end_A, shape=config.shape)
    return CustomScan(config.positions_A)


def build_detector(
    config: AnnularDetectorConfig | FlexibleAnnularDetectorConfig | PixelatedDetectorConfig | ImageDetectorConfig | WavesDetectorConfig | SegmentedDetectorConfig,
    *,
    energy_eV: float,
    gpts: tuple[int, int],
    sampling: tuple[float, float],
) -> nn.Module:
    if isinstance(config, AnnularDetectorConfig):
        return AnnularDetector(
            energy_eV=energy_eV,
            gpts=gpts,
            sampling=sampling,
            inner_mrad=config.inner_mrad,
            outer_mrad=config.outer_mrad,
            offset_mrad=config.offset_mrad,
        )
    if isinstance(config, FlexibleAnnularDetectorConfig):
        return FlexibleAnnularDetector(
            energy_eV=energy_eV,
            gpts=gpts,
            sampling=sampling,
            step_size_mrad=config.step_size_mrad,
            inner_mrad=config.inner_mrad,
            outer_mrad=config.outer_mrad,
            offset_mrad=config.offset_mrad,
        )
    if isinstance(config, PixelatedDetectorConfig):
        return PixelatedDetector(
            energy_eV=energy_eV,
            sampling=sampling,
            reciprocal_space=config.reciprocal_space,
            gpts=config.gpts,
            max_angle_mrad=config.max_angle_mrad,
        )
    if isinstance(config, ImageDetectorConfig):
        ctf = None
        if config.semiangle_cutoff_mrad is not None:
            ctf = CTF(
                energy_eV=energy_eV,
                gpts=gpts,
                sampling=sampling,
                semiangle_cutoff_mrad=config.semiangle_cutoff_mrad,
                soft_edge_mrad=config.soft_edge_mrad,
                focal_spread_A=config.focal_spread_A,
                angular_spread_mrad=config.angular_spread_mrad,
                flip_phase=config.flip_phase,
                wiener_snr=config.wiener_snr,
                aberrations=config.aberrations,
            )
        return ImageDetector(ctf=ctf, output=config.output)
    if isinstance(config, WavesDetectorConfig):
        return WavesDetector(gpts=config.gpts, reciprocal_space=config.reciprocal_space)
    return SegmentedDetector(
        energy_eV=energy_eV,
        gpts=gpts,
        sampling=sampling,
        inner_mrad=config.inner_mrad,
        outer_mrad=config.outer_mrad,
        nbins_radial=config.nbins_radial,
        nbins_azimuthal=config.nbins_azimuthal,
        rotation_rad=config.rotation_rad,
        offset_mrad=config.offset_mrad,
    )


def build_detectors(
    config: (
        AnnularDetectorConfig
        | FlexibleAnnularDetectorConfig
        | PixelatedDetectorConfig
        | ImageDetectorConfig
        | WavesDetectorConfig
        | SegmentedDetectorConfig
        | Mapping[
            str,
            AnnularDetectorConfig
            | FlexibleAnnularDetectorConfig
            | PixelatedDetectorConfig
            | ImageDetectorConfig
            | WavesDetectorConfig
            | SegmentedDetectorConfig,
        ]
    ),
    *,
    energy_eV: float,
    gpts: tuple[int, int],
    sampling: tuple[float, float],
) -> nn.Module | dict[str, nn.Module]:
    if isinstance(config, Mapping):
        return {
            name: build_detector(
                detector_config,
                energy_eV=energy_eV,
                gpts=gpts,
                sampling=sampling,
            )
            for name, detector_config in config.items()
        }
    return build_detector(config, energy_eV=energy_eV, gpts=gpts, sampling=sampling)


def build_multislice(config: MultisliceConfig) -> nn.Module:
    if config.backend in {"fourier", "prism", "bloch"}:
        return MultisliceSystem(
            energy_eV=config.energy_eV,
            gpts=config.gpts,
            sampling=config.sampling,
            slice_thickness_A=config.slice_thickness_A,
        )
    if config.backend == "realspace":
        return RealSpaceMultisliceSystem(
            energy_eV=config.energy_eV,
            sampling=config.sampling,
            slice_thickness_A=config.slice_thickness_A,
            series_order=config.series_order,
            max_terms=config.max_terms,
            tolerance=config.tolerance,
        )
    raise ValueError(f"Unknown multislice backend: {config.backend}")


def build_postprocess(
    config: Sequence[PipelineStepConfig] | Mapping[str, Sequence[PipelineStepConfig]],
    *,
    gpts: tuple[int, int],
    sampling: tuple[float, float],
    output_names: Sequence[str] | None = None,
) -> nn.Module:
    if isinstance(config, Mapping):
        return NamedPostprocess(
            {
                name: build_measurement_pipeline(steps, gpts=gpts, sampling=sampling)
                for name, steps in config.items()
            }
        )
    if output_names is not None:
        return NamedPostprocess(
            {
                name: build_measurement_pipeline(config, gpts=gpts, sampling=sampling)
                for name in output_names
            }
        )
    return build_measurement_pipeline(config, gpts=gpts, sampling=sampling)


def _is_supported_bloch_detector_config(config) -> bool:
    if isinstance(config, PixelatedDetectorConfig):
        return bool(config.reciprocal_space)
    if isinstance(config, WavesDetectorConfig):
        return bool(config.reciprocal_space)
    return isinstance(
        config,
        (AnnularDetectorConfig, FlexibleAnnularDetectorConfig, SegmentedDetectorConfig),
    )


def _bloch_requires_reciprocal_waves(config) -> bool:
    if config is None:
        return False
    if isinstance(config, Mapping):
        return any(_bloch_requires_reciprocal_waves(item) for item in config.values())
    return isinstance(config, WavesDetectorConfig) and bool(config.reciprocal_space)


def infer_mode(config: ExperimentConfig) -> str:
    if config.inelastic is not None:
        return "inelastic"
    if config.magnetic is not None:
        return "magnetic"
    if config.multislice.backend == "bloch":
        return "diffraction"
    if config.detector is not None:
        if isinstance(config.detector, PixelatedDetectorConfig):
            return "diffraction" if config.scan is not None else "hrtem"
        if isinstance(config.detector, ImageDetectorConfig):
            return "hrtem"
        if isinstance(config.detector, Mapping):
            values = tuple(config.detector.values())
            if all(isinstance(item, PixelatedDetectorConfig) for item in values):
                return "diffraction" if config.scan is not None else "hrtem"
            if all(isinstance(item, ImageDetectorConfig) for item in values):
                return "hrtem"
    if config.scan is not None and isinstance(config.source, ProbeSourceConfig):
        return "stem"
    return "hrtem" if config.scan is None else "stem"


def detector_names_from_config(config: ExperimentConfig) -> tuple[str, ...] | None:
    if config.detector is None:
        return ("exit_wave",)
    if isinstance(config.detector, Mapping):
        return tuple(str(name) for name in config.detector)
    return ("detector",)


class ExperimentBuilder(nn.Module):
    """Build a differentiable TEM/STEM stack from structured parameter objects."""

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        self.source = build_source(config.source)
        self.multislice = build_multislice(config.multislice)
        self.potential = config.potential
        self.stack = self._build_stack()

    @classmethod
    def from_hrtem(cls, config: HRTEMConfig) -> "ExperimentBuilder":
        return cls(
            ExperimentConfig(
                source=config.source,
                multislice=config.multislice,
                potential=config.potential,
                bloch=config.bloch,
                detector=config.detector,
                coherence=config.coherence,
                postprocess=config.postprocess,
                inelastic=config.inelastic,
                magnetic=config.magnetic,
            )
        )

    @classmethod
    def from_stem(cls, config: STEMConfig) -> "ExperimentBuilder":
        return cls(
            ExperimentConfig(
                source=config.source,
                multislice=config.multislice,
                potential=config.potential,
                bloch=config.bloch,
                detector=config.detector,
                scan=config.scan,
                coherence=config.coherence,
                postprocess=config.postprocess,
                inelastic=config.inelastic,
                magnetic=config.magnetic,
            )
        )

    @classmethod
    def from_diffraction(cls, config: DiffractionConfig) -> "ExperimentBuilder":
        return cls(
            ExperimentConfig(
                source=config.source,
                multislice=config.multislice,
                potential=config.potential,
                bloch=config.bloch,
                detector=config.detector,
                scan=config.scan,
                coherence=config.coherence,
                postprocess=config.postprocess,
                inelastic=config.inelastic,
                magnetic=config.magnetic,
            )
        )

    def _build_stack(self) -> LayerStack:
        layers: list[tuple[str, nn.Module]] = []
        scan_module = build_scan(self.config.scan) if self.config.scan is not None else None
        source: nn.Module = self.source

        if self.config.multislice.backend == "bloch":
            if self.config.bloch is None:
                raise ValueError("Bloch backend requires a BlochConfig")
            if self.config.detector is not None and not (
                _is_supported_bloch_detector_config(self.config.detector)
                or (
                    isinstance(self.config.detector, Mapping)
                    and all(_is_supported_bloch_detector_config(item) for item in self.config.detector.values())
                )
            ):
                raise ValueError(
                    "Bloch backend currently supports reciprocal-intensity detector families only "
                    "(pixelated reciprocal, annular, flexible-annular, segmented)"
                )
            layers.append(
                (
                    "interaction",
                    (
                        BlochReciprocalWaves(
                            config=self.config.bloch,
                            energy_eV=self.config.multislice.energy_eV,
                            gpts=self.config.multislice.gpts,
                            sampling=self.config.multislice.sampling,
                        )
                        if _bloch_requires_reciprocal_waves(self.config.detector)
                        else BlochDiffractionPattern(
                            config=self.config.bloch,
                            energy_eV=self.config.multislice.energy_eV,
                            gpts=self.config.multislice.gpts,
                            sampling=self.config.multislice.sampling,
                        )
                    ),
                )
            )
        elif self.config.multislice.backend == "prism":
            if not isinstance(self.source, Probe):
                raise ValueError("PRISM backend requires a probe-forming source")
            if self.config.potential is None:
                raise ValueError("PRISM backend requires a potential module")
            layers.append(
                (
                    "interaction",
                    PrismScanInteraction(
                        probe=self.source,
                        potential=self.potential,
                        scan=scan_module,
                        config=self.config.multislice,
                        num_slices=self.config.multislice.num_slices,
                    ),
                )
            )
        else:
            if scan_module is not None:
                layers.append(
                    (
                        "scan",
                        ScanLayer(
                            scan=scan_module,
                            shift_wave=fft_shift_wave,
                            sampling=self.config.multislice.sampling,
                        ),
                    )
                )

            if self.config.potential is None:
                raise ValueError("Multislice backends require a potential module")
            layers.append(
                (
                    "interaction",
                    PotentialSliceInteraction(
                        multislice=self.multislice,
                        potential=self.potential,
                        num_slices=self.config.multislice.num_slices,
                    ),
                )
            )

        if self.config.inelastic is not None:
            layers.append(("inelastic", InelasticInteraction(self.config.inelastic)))

        if self.config.magnetic is not None:
            layers.append(
                (
                    "magnetic",
                    MagneticInteraction(
                        self.config.magnetic,
                        sampling=self.config.multislice.sampling,
                    ),
                )
            )

        detector: nn.Module | dict[str, nn.Module] | None = None
        if self.config.detector is not None:
            detector = build_detectors(
                self.config.detector,
                energy_eV=self.config.multislice.energy_eV,
                gpts=self.config.multislice.gpts,
                sampling=self.config.multislice.sampling,
            )

        detector_consumed = False
        if self.config.coherence is not None:
            if self.config.coherence.mode == "diagonal_mcf":
                if self.config.multislice.backend in {"prism", "bloch"}:
                    raise ValueError("Diagonal MCF coherence is currently supported only for multislice backends")
                if not isinstance(source, Probe):
                    raise ValueError("Diagonal MCF coherence requires a probe-forming source")
                if self.config.coherence.eigenvectors is None:
                    raise ValueError("Diagonal MCF coherence requires eigenvectors to be specified")
                mcf = DiagonalMCF(
                    source.ctf,
                    eigenvectors=self.config.coherence.eigenvectors,
                    focal_spread_A=self.config.coherence.focal_spread_A,
                    source_size_A=self.config.coherence.source_size_A,
                    rectangular_offset_A=self.config.coherence.rectangular_offset_A,
                )
                source = DiagonalMCFSource(mcf)
                layers.append(("coherence", ModeCoherenceReadout(detector)))
                detector_consumed = True
            else:
                ensemble = CoherenceEnsemble(
                    defocus_offsets_A=self.config.coherence.defocus_offsets_A,
                    source_offsets_A=self.config.coherence.source_offsets_A,
                    weights=self.config.coherence.weights,
                )
                coherence = CoherenceAverager(ensemble)
                if isinstance(source, Probe) and (
                    self.config.coherence.defocus_offsets_A is not None
                    or self.config.coherence.source_offsets_A is not None
                ):
                    if self.config.multislice.backend in {"prism", "bloch"}:
                        raise ValueError(
                            "Probe ensemble coherence is currently supported only for multislice backends"
                        )
                    source = ProbeCoherenceEnsembleSource(source, ensemble)
                    layers.append(("coherence", ModeCoherenceReadout(detector, weights=coherence.weights)))
                    detector_consumed = True
                else:
                    layers.append(
                        (
                            "coherence",
                            CoherenceReadout(
                                coherence=coherence,
                                sampling=self.config.multislice.sampling,
                                detector=detector,
                            ),
                        )
                    )
                    detector_consumed = True

        if self.config.multislice.backend != "bloch":
            layers.insert(
                0,
                (
                    "source_tilt",
                    SourceTiltLayer(
                        energy_eV=self.config.multislice.energy_eV,
                        sampling=self.config.multislice.sampling,
                        tilt_mrad=source_tilt_from_config(self.config.source),
                    ),
                ),
            )
        if (
            self.config.multislice.backend not in {"bloch", "prism"}
            and self.config.scan is None
            and isinstance(self.config.source, ProbeSourceConfig)
        ):
            centered_position = (
                0.5 * self.config.multislice.gpts[0] * self.config.multislice.sampling[0],
                0.5 * self.config.multislice.gpts[1] * self.config.multislice.sampling[1],
            )
            layers.insert(
                1 if self.config.multislice.backend != "bloch" else 0,
                (
                    "source_position",
                    StaticWaveShiftLayer(
                        position_A=centered_position,
                        sampling=self.config.multislice.sampling,
                    ),
                ),
            )
        if detector is not None and not detector_consumed:
            detector_layer: nn.Module
            if self.config.multislice.backend == "bloch" and isinstance(detector, dict):
                detector_layer = NamedBlochDetectorReadout(detector)
            elif self.config.multislice.backend == "bloch":
                detector_layer = BlochDetectorReadout(detector)
            elif isinstance(detector, dict):
                detector_layer = NamedDetectorReadout(
                    detector,
                    batched=self.config.scan is not None,
                )
            elif self.config.scan is None:
                detector_layer = DetectorReadout(detector)
            else:
                detector_layer = BatchDetectorReadout(detector)
            layers.append(("detector", detector_layer))

        if self.config.postprocess is not None:
            layers.append(
                (
                    "postprocess",
                    build_postprocess(
                        self.config.postprocess,
                        gpts=self.config.multislice.gpts,
                        sampling=self.config.multislice.sampling,
                        output_names=tuple(detector.keys()) if isinstance(detector, dict) else None,
                    ),
                )
            )

        return LayerStack(source=source, layers=layers)

    def parameter_tensor_dict(self) -> dict[str, torch.Tensor]:
        return {name: param for name, param in self.named_parameters()}

    def _is_tem_parameter_name(self, name: str) -> bool:
        tem_prefixes = (
            "source.",
            "stack.layers.source_tilt.",
            "stack.layers.source_position.",
            "stack.layers.scan.",
            "stack.layers.detector.",
            "stack.layers.coherence.",
        )
        return any(name.startswith(prefix) for prefix in tem_prefixes)

    def tem_parameter_tensor_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: param
            for name, param in self.parameter_tensor_dict().items()
            if self._is_tem_parameter_name(name)
        }

    def tem_parameter_layout(self) -> tuple[ParameterVectorEntry, ...]:
        entries: list[ParameterVectorEntry] = []
        offset = 0
        for name, param in self.tem_parameter_tensor_dict().items():
            width = int(param.numel())
            entries.append(
                ParameterVectorEntry(
                    name=name,
                    shape=tuple(param.shape),
                    start=offset,
                    stop=offset + width,
                )
            )
            offset += width
        return tuple(entries)

    def tem_parameter_names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.tem_parameter_layout())

    def tem_parameter_vector(self) -> torch.Tensor:
        parameters = tuple(self.tem_parameter_tensor_dict().values())
        if not parameters:
            return torch.empty(0, dtype=torch.float64)
        return torch.cat([parameter.reshape(-1) for parameter in parameters], dim=0)

    def set_tem_parameter_vector(
        self,
        values: torch.Tensor | Sequence[float | complex],
    ) -> None:
        vector = torch.as_tensor(values, dtype=torch.float64).reshape(-1)
        layout = self.tem_parameter_layout()
        expected = 0 if not layout else layout[-1].stop
        if vector.numel() != expected:
            raise ValueError(
                f"TEM parameter vector has {vector.numel()} entries, expected {expected}"
            )

        for entry in layout:
            value = vector[entry.start : entry.stop].reshape(entry.shape)
            self.set_parameter(entry.name, value)

    def _resolve_parameter_name(self, name: str) -> str:
        parameters = self.parameter_tensor_dict()
        if name in parameters:
            return name

        candidates = [key for key in parameters if key.endswith(f".{name}")]
        if not candidates and name.startswith("detector."):
            detector_suffix = name.removeprefix("detector.")
            candidates = [key for key in parameters if key.endswith(f".detectors.{detector_suffix}")]
        if not candidates and name.startswith("source."):
            source_suffix = name.removeprefix("source.")
            candidates = [key for key in parameters if key.endswith(f".source_tilt.{source_suffix}")]

        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise KeyError(f"Ambiguous parameter name '{name}'. Matches: {candidates}")
        raise KeyError(f"Unknown parameter name '{name}'")

    def set_parameter(self, name: str, value: torch.Tensor | float | complex) -> None:
        resolved_name = self._resolve_parameter_name(name)
        module_name, _, param_name = resolved_name.rpartition(".")
        target = self.get_submodule(module_name) if module_name else self
        parameter = getattr(target, param_name)
        with torch.no_grad():
            parameter.copy_(torch.as_tensor(value, device=parameter.device, dtype=parameter.dtype))

    def forward(
        self,
        *,
        return_intermediates: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        return self.stack(return_intermediates=return_intermediates)

    def run(
        self,
        *,
        return_intermediates: bool = False,
        include_exit_wave: bool = False,
    ) -> SimulationResult:
        exit_wave = None
        if not return_intermediates and not include_exit_wave:
            outputs = self.forward(return_intermediates=False)
            intermediates = None
        else:
            states = self.forward(return_intermediates=True)
            outputs = states[next(reversed(states))]
            exit_wave = states.get("interaction")
            intermediates = states if return_intermediates else None
        return SimulationResult(
            outputs=outputs,
            detector_names=detector_names_from_config(self.config),
            exit_wave=exit_wave,
            intermediates=intermediates,
            metadata={
                "mode": infer_mode(self.config),
                "scanned": self.config.scan is not None,
                "coherent_average": self.config.coherence is not None,
                "coherence_model": None if self.config.coherence is None else self.config.coherence.mode,
                "inelastic": self.config.inelastic is not None,
                "magnetic": self.config.magnetic is not None,
                "num_slices": self.config.multislice.num_slices,
                "propagation_backend": self.config.multislice.backend,
                "postprocessed": self.config.postprocess is not None,
                "includes_exit_wave": exit_wave is not None,
            },
        )

    def run_parameter_series(
        self,
        parameter_name: str,
        values: torch.Tensor | Sequence[float | complex],
    ) -> SimulationSeriesResult:
        resolved_name = self._resolve_parameter_name(parameter_name)
        module_name, _, param_name = resolved_name.rpartition(".")
        target = self.get_submodule(module_name) if module_name else self
        parameter = getattr(target, param_name)
        values_tensor = torch.as_tensor(values, device=parameter.device, dtype=parameter.dtype).reshape(-1)
        original = parameter.detach().clone()

        def _stack_outputs(items: list[torch.Tensor | Mapping[str, torch.Tensor]]):
            first = items[0]
            if isinstance(first, Mapping):
                return {
                    name: torch.stack([item[name] for item in items], dim=0)
                    for name in first
                }
            return torch.stack(items, dim=0)

        try:
            outputs: list[torch.Tensor | Mapping[str, torch.Tensor]] = []
            for value in values_tensor:
                self.set_parameter(parameter_name, value)
                outputs.append(self.forward(return_intermediates=False))
        finally:
            self.set_parameter(parameter_name, original)

        return SimulationSeriesResult(
            outputs=_stack_outputs(outputs),
            parameter_name=resolved_name,
            parameter_values=values_tensor.detach().clone(),
            detector_names=detector_names_from_config(self.config),
            metadata={
                "mode": infer_mode(self.config),
                "scanned": self.config.scan is not None,
                "coherent_average": self.config.coherence is not None,
                "coherence_model": None if self.config.coherence is None else self.config.coherence.mode,
                "inelastic": self.config.inelastic is not None,
                "magnetic": self.config.magnetic is not None,
                "num_slices": self.config.multislice.num_slices,
                "propagation_backend": self.config.multislice.backend,
                "postprocessed": self.config.postprocess is not None,
                "series_parameter_name": resolved_name,
            },
        )

    def run_tem_parameter_series(
        self,
        values: torch.Tensor | Sequence[Sequence[float | complex]],
    ) -> SimulationSeriesResult:
        original = self.tem_parameter_vector().detach().clone()
        values_tensor = torch.as_tensor(values, dtype=torch.float64)
        if values_tensor.ndim == 1:
            values_tensor = values_tensor.unsqueeze(0)
        values_tensor = values_tensor.reshape(values_tensor.shape[0], -1)

        if values_tensor.shape[1] != original.numel():
            raise ValueError(
                f"TEM parameter series expects vectors of length {original.numel()}, "
                f"received {values_tensor.shape[1]}"
            )

        def _stack_outputs(items: list[torch.Tensor | Mapping[str, torch.Tensor]]):
            first = items[0]
            if isinstance(first, Mapping):
                return {
                    name: torch.stack([item[name] for item in items], dim=0)
                    for name in first
                }
            return torch.stack(items, dim=0)

        try:
            outputs: list[torch.Tensor | Mapping[str, torch.Tensor]] = []
            for value in values_tensor:
                self.set_tem_parameter_vector(value)
                outputs.append(self.forward(return_intermediates=False))
        finally:
            self.set_tem_parameter_vector(original)

        layout = self.tem_parameter_layout()
        return SimulationSeriesResult(
            outputs=_stack_outputs(outputs),
            parameter_name="tem_parameter_vector",
            parameter_values=values_tensor.detach().clone(),
            detector_names=detector_names_from_config(self.config),
            metadata={
                "mode": infer_mode(self.config),
                "scanned": self.config.scan is not None,
                "coherent_average": self.config.coherence is not None,
                "coherence_model": None if self.config.coherence is None else self.config.coherence.mode,
                "inelastic": self.config.inelastic is not None,
                "magnetic": self.config.magnetic is not None,
                "num_slices": self.config.multislice.num_slices,
                "propagation_backend": self.config.multislice.backend,
                "postprocessed": self.config.postprocess is not None,
                "series_parameter_name": "tem_parameter_vector",
                "series_parameter_layout": [
                    {
                        "name": entry.name,
                        "shape": entry.shape,
                        "start": entry.start,
                        "stop": entry.stop,
                    }
                    for entry in layout
                ],
            },
        )
