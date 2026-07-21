from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import torch
from torch import nn

from torchtem.detectors import PixelatedDetector
from torchtem.multislice import MultisliceSystem
from torchtem.optics import evaluate_ctf_from_coefficients
from torchtem.physics import energy2wavelength, polar_spatial_frequencies
from torchtem.scan import fft_shift_wave


@dataclass(frozen=True)
class TEMControlEntry:
    name: str
    start: int
    stop: int


@dataclass(frozen=True)
class TEMStageEntry:
    name: str
    start: int
    stop: int
    parameter_names: tuple[str, ...]


class FixedStackTEMControls(nn.Module):
    """Example differentiable TEM stack control model for a fixed microscope topology.

    The stack is intentionally fixed:
    condenser-1 current, condenser-2 current, condenser stigmator x/y,
    objective focus, objective stigmator x/y, and a simple corrector block.

    The calibration constants below are example values intended to demonstrate how
    a microscope state vector can be mapped into physical aberration coefficients.
    Real microscopes should replace these with measured calibrations.
    """

    CONTROL_NAMES = (
        "c1_current",
        "c2_current",
        "condenser_stig_dx",
        "condenser_stig_dy",
        "objective_focus",
        "objective_stig_dx",
        "objective_stig_dy",
        "corrector_c30",
        "corrector_c32_x",
        "corrector_c32_y",
        "corrector_c34_x",
        "corrector_c34_y",
    )
    STAGE_NAMES = (
        ("condenser", ("c1_current", "c2_current", "condenser_stig_dx", "condenser_stig_dy")),
        ("objective", ("objective_focus", "objective_stig_dx", "objective_stig_dy")),
        (
            "corrector",
            ("corrector_c30", "corrector_c32_x", "corrector_c32_y", "corrector_c34_x", "corrector_c34_y"),
        ),
    )

    def __init__(
        self,
        initial_vector: torch.Tensor | Sequence[float] | None = None,
        *,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        super().__init__()
        if initial_vector is None:
            initial_vector = torch.zeros(len(self.CONTROL_NAMES), dtype=dtype, device=device)
        vector = torch.as_tensor(initial_vector, dtype=dtype, device=device).reshape(-1)
        if vector.numel() != len(self.CONTROL_NAMES):
            raise ValueError(
                f"Expected {len(self.CONTROL_NAMES)} control parameters, got {vector.numel()}"
            )
        self.control_vector = nn.Parameter(vector.clone())

    @classmethod
    def from_named_values(
        cls,
        updates: Mapping[str, torch.Tensor | float],
        *,
        base_vector: torch.Tensor | Sequence[float] | None = None,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> "FixedStackTEMControls":
        controls = cls(initial_vector=base_vector, dtype=dtype, device=device)
        with torch.no_grad():
            controls.control_vector.copy_(controls.updated_vector(updates).detach())
        return controls

    @classmethod
    def from_stage_values(
        cls,
        stage_values: Mapping[str, torch.Tensor | Sequence[float]],
        *,
        base_vector: torch.Tensor | Sequence[float] | None = None,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> "FixedStackTEMControls":
        controls = cls(initial_vector=base_vector, dtype=dtype, device=device)
        with torch.no_grad():
            controls.control_vector.copy_(
                controls.vector_from_stage_values(stage_values).detach()
            )
        return controls

    @classmethod
    def layout(cls) -> tuple[TEMControlEntry, ...]:
        return tuple(TEMControlEntry(name=name, start=i, stop=i + 1) for i, name in enumerate(cls.CONTROL_NAMES))

    @classmethod
    def stage_layout(cls) -> tuple[TEMStageEntry, ...]:
        entries = []
        for stage_name, parameter_names in cls.STAGE_NAMES:
            start = cls.index(parameter_names[0])
            stop = cls.index(parameter_names[-1]) + 1
            entries.append(
                TEMStageEntry(
                    name=stage_name,
                    start=start,
                    stop=stop,
                    parameter_names=tuple(parameter_names),
                )
            )
        return tuple(entries)

    @classmethod
    def zero_vector(
        cls,
        *,
        batch_shape: tuple[int, ...] = (),
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> torch.Tensor:
        return torch.zeros(batch_shape + (len(cls.CONTROL_NAMES),), dtype=dtype, device=device)

    def vector(self) -> torch.Tensor:
        return self.control_vector

    @classmethod
    def _coerce_vector(
        cls,
        values: torch.Tensor | Sequence[float],
        *,
        dtype: torch.dtype,
        device: torch.device | None,
    ) -> torch.Tensor:
        values = torch.as_tensor(values, dtype=dtype, device=device)
        if values.shape[-1] != len(cls.CONTROL_NAMES):
            raise ValueError(
                f"Expected last dimension {len(cls.CONTROL_NAMES)}, got {values.shape[-1]}"
            )
        return values

    @classmethod
    def entry(cls, name: str) -> TEMControlEntry:
        for item in cls.layout():
            if item.name == name:
                return item
        raise KeyError(f"Unknown TEM control name: {name}")

    @classmethod
    def index(cls, name: str) -> int:
        return cls.entry(name).start

    @classmethod
    def stage_entry(cls, name: str) -> TEMStageEntry:
        for item in cls.stage_layout():
            if item.name == name:
                return item
        raise KeyError(f"Unknown TEM stage name: {name}")

    @classmethod
    def assemble_named_tensor(
        cls,
        updates: Mapping[str, torch.Tensor | float],
        *,
        base_vector: torch.Tensor | Sequence[float] | None = None,
        batch_shape: tuple[int, ...] = (),
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> torch.Tensor:
        vector = (
            cls.zero_vector(batch_shape=batch_shape, dtype=dtype, device=device)
            if base_vector is None
            else torch.as_tensor(base_vector, dtype=dtype, device=device)
        )
        if vector.shape[-1] != len(cls.CONTROL_NAMES):
            raise ValueError(
                f"Expected last dimension {len(cls.CONTROL_NAMES)}, got {vector.shape[-1]}"
            )
        if batch_shape and vector.shape[:-1] == ():
            vector = torch.broadcast_to(vector, batch_shape + (len(cls.CONTROL_NAMES),))
        updated = vector.clone()
        batch_shape = vector.shape[:-1]
        for name, value in updates.items():
            entry = cls.entry(name)
            replacement = torch.as_tensor(value, dtype=updated.dtype, device=updated.device)
            try:
                replacement = torch.broadcast_to(replacement, batch_shape)
            except RuntimeError as exc:
                raise ValueError(
                    f"Control update {name!r} with shape {tuple(replacement.shape)} "
                    f"cannot broadcast to batch shape {tuple(batch_shape)}"
                ) from exc
            updated[..., entry.start] = replacement
        return updated

    @classmethod
    def assemble_stage_tensor(
        cls,
        stage_values: Mapping[str, torch.Tensor | Sequence[float]],
        *,
        base_vector: torch.Tensor | Sequence[float] | None = None,
        batch_shape: tuple[int, ...] = (),
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> torch.Tensor:
        vector = (
            cls.zero_vector(batch_shape=batch_shape, dtype=dtype, device=device)
            if base_vector is None
            else torch.as_tensor(base_vector, dtype=dtype, device=device)
        )
        if vector.shape[-1] != len(cls.CONTROL_NAMES):
            raise ValueError(
                f"Expected last dimension {len(cls.CONTROL_NAMES)}, got {vector.shape[-1]}"
            )
        if batch_shape and vector.shape[:-1] == ():
            vector = torch.broadcast_to(vector, batch_shape + (len(cls.CONTROL_NAMES),))
        updated = vector.clone()
        batch_shape = vector.shape[:-1]
        for stage_name, values in stage_values.items():
            stage = cls.stage_entry(stage_name)
            replacement = torch.as_tensor(values, dtype=updated.dtype, device=updated.device)
            expected_tail = (stage.stop - stage.start,)
            try:
                replacement = torch.broadcast_to(replacement, batch_shape + expected_tail)
            except RuntimeError as exc:
                raise ValueError(
                    f"Stage update {stage_name!r} with shape {tuple(replacement.shape)} "
                    f"cannot broadcast to shape {tuple(batch_shape + expected_tail)}"
                ) from exc
            updated[..., stage.start:stage.stop] = replacement
        return updated

    @classmethod
    def named_parameter_grid(
        cls,
        parameter_values: Mapping[str, torch.Tensor | Sequence[float]],
        *,
        base_vector: torch.Tensor | Sequence[float] | None = None,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> torch.Tensor:
        names = list(parameter_values)
        if not names:
            raise ValueError("parameter_values must not be empty")

        one_dimensional_values = []
        grid_shape = []
        for name in names:
            cls.entry(name)
            values = torch.as_tensor(parameter_values[name], dtype=dtype, device=device)
            if values.ndim != 1:
                raise ValueError(
                    f"Grid values for {name!r} must be one-dimensional, got shape {tuple(values.shape)}"
                )
            one_dimensional_values.append(values)
            grid_shape.append(values.shape[0])

        grid = cls.zero_vector(batch_shape=tuple(grid_shape), dtype=dtype, device=device)
        if base_vector is not None:
            base = torch.as_tensor(base_vector, dtype=dtype, device=device)
            if base.shape != (len(cls.CONTROL_NAMES),):
                raise ValueError(
                    f"base_vector must have shape {(len(cls.CONTROL_NAMES),)}, got {tuple(base.shape)}"
                )
            grid = grid + base

        mesh = torch.meshgrid(*one_dimensional_values, indexing="ij")
        for name, values in zip(names, mesh):
            entry = cls.entry(name)
            grid[..., entry.start] = values
        return grid

    def named_values(
        self,
        vector: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        values = self._coerce_vector(
            self.control_vector if vector is None else vector,
            dtype=self.control_vector.dtype,
            device=self.control_vector.device,
        )
        return {
            entry.name: values[..., entry.start:entry.stop].squeeze(-1)
            for entry in self.layout()
        }

    def stage_values(
        self,
        vector: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        values = self._coerce_vector(
            self.control_vector if vector is None else vector,
            dtype=self.control_vector.dtype,
            device=self.control_vector.device,
        )
        return {
            entry.name: values[..., entry.start:entry.stop]
            for entry in self.stage_layout()
        }

    def vector_from_stage_values(
        self,
        stage_values: Mapping[str, torch.Tensor | Sequence[float]],
        *,
        base_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        vector = self.control_vector if base_vector is None else base_vector
        return self.assemble_stage_tensor(
            stage_values,
            base_vector=vector,
            dtype=self.control_vector.dtype,
            device=self.control_vector.device,
        )

    def updated_vector(
        self,
        updates: Mapping[str, torch.Tensor | float],
        vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = self.control_vector if vector is None else vector
        return self.assemble_named_tensor(
            updates,
            base_vector=values,
            dtype=self.control_vector.dtype,
            device=self.control_vector.device,
        )

    def updated_stage_vector(
        self,
        stage_name: str,
        values: torch.Tensor | Sequence[float],
        vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        current = (
            self.control_vector
            if vector is None
            else torch.as_tensor(
                vector,
                dtype=self.control_vector.dtype,
                device=self.control_vector.device,
            )
        )
        if current.shape[-1] != len(self.CONTROL_NAMES):
            raise ValueError(
                f"Expected last dimension {len(self.CONTROL_NAMES)}, got {current.shape[-1]}"
            )

        stage = self.stage_entry(stage_name)
        replacement = torch.as_tensor(
            values,
            dtype=current.dtype,
            device=current.device,
        )
        batch_shape = current.shape[:-1]
        expected_tail = (stage.stop - stage.start,)
        if replacement.shape == expected_tail:
            replacement = torch.broadcast_to(replacement, batch_shape + expected_tail)
        else:
            try:
                replacement = torch.broadcast_to(replacement, batch_shape + expected_tail)
            except RuntimeError as exc:
                raise ValueError(
                    f"Stage update {stage_name!r} with shape {tuple(replacement.shape)} "
                    f"cannot broadcast to shape {tuple(batch_shape + expected_tail)}"
                ) from exc

        updated = current.clone()
        updated[..., stage.start:stage.stop] = replacement
        return updated

    @classmethod
    def decode_values(
        cls,
        values: torch.Tensor | Sequence[float],
        *,
        dtype: torch.dtype = torch.float64,
        device: torch.device | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        values = cls._coerce_vector(values, dtype=dtype, device=device)

        c1 = values[..., 0]
        c2 = values[..., 1]
        cond_dx = values[..., 2]
        cond_dy = values[..., 3]
        obj_focus = values[..., 4]
        obj_dx = values[..., 5]
        obj_dy = values[..., 6]
        corr_c30 = values[..., 7]
        corr_c32_x = values[..., 8]
        corr_c32_y = values[..., 9]
        corr_c34_x = values[..., 10]
        corr_c34_y = values[..., 11]

        source_astig_x = 24.0 * cond_dx + 7.0 * cond_dy
        source_astig_y = 7.0 * cond_dx - 24.0 * cond_dy
        source_c12 = torch.sqrt(source_astig_x.square() + source_astig_y.square() + 1e-18)
        source_phi12 = 0.5 * torch.atan2(source_astig_y, source_astig_x + 1e-18)

        image_astig_x = 18.0 * obj_dx - 900.0 * corr_c32_x
        image_astig_y = 18.0 * obj_dy - 900.0 * corr_c32_y
        image_c12 = torch.sqrt(image_astig_x.square() + image_astig_y.square() + 1e-18)
        image_phi12 = 0.5 * torch.atan2(image_astig_y, image_astig_x + 1e-18)

        image_c32_x = -2200.0 * corr_c32_x
        image_c32_y = -2200.0 * corr_c32_y
        image_c32 = torch.sqrt(image_c32_x.square() + image_c32_y.square() + 1e-18)
        image_phi32 = 0.5 * torch.atan2(image_c32_y, image_c32_x + 1e-18)

        image_c34_x = -1400.0 * corr_c34_x
        image_c34_y = -1400.0 * corr_c34_y
        image_c34 = torch.sqrt(image_c34_x.square() + image_c34_y.square() + 1e-18)
        image_phi34 = 0.25 * torch.atan2(image_c34_y, image_c34_x + 1e-18)

        source = {
            "C10": -65.0 * c1 - 28.0 * c2,
            "C12": source_c12,
            "phi12": source_phi12,
            "C21": 14.0 * (c1 - c2),
            "phi21": torch.zeros_like(c1),
            "C23": torch.zeros_like(c1),
            "phi23": torch.zeros_like(c1),
            "C30": torch.zeros_like(c1),
            "C32": torch.zeros_like(c1),
            "phi32": torch.zeros_like(c1),
            "C34": torch.zeros_like(c1),
            "phi34": torch.zeros_like(c1),
            "C41": torch.zeros_like(c1),
            "phi41": torch.zeros_like(c1),
            "C43": torch.zeros_like(c1),
            "phi43": torch.zeros_like(c1),
            "C45": torch.zeros_like(c1),
            "phi45": torch.zeros_like(c1),
            "C50": torch.zeros_like(c1),
            "C52": torch.zeros_like(c1),
            "phi52": torch.zeros_like(c1),
            "C54": torch.zeros_like(c1),
            "phi54": torch.zeros_like(c1),
            "C56": torch.zeros_like(c1),
            "phi56": torch.zeros_like(c1),
        }

        image = {
            "C10": 110.0 * obj_focus + 6.0 * (c1 - c2),
            "C12": image_c12,
            "phi12": image_phi12,
            "C21": torch.zeros_like(c1),
            "phi21": torch.zeros_like(c1),
            "C23": torch.zeros_like(c1),
            "phi23": torch.zeros_like(c1),
            "C30": 1.2e4 - 8.5e3 * corr_c30,
            "C32": image_c32,
            "phi32": image_phi32,
            "C34": image_c34,
            "phi34": image_phi34,
            "C41": torch.zeros_like(c1),
            "phi41": torch.zeros_like(c1),
            "C43": torch.zeros_like(c1),
            "phi43": torch.zeros_like(c1),
            "C45": torch.zeros_like(c1),
            "phi45": torch.zeros_like(c1),
            "C50": torch.zeros_like(c1),
            "C52": torch.zeros_like(c1),
            "phi52": torch.zeros_like(c1),
            "C54": torch.zeros_like(c1),
            "phi54": torch.zeros_like(c1),
            "C56": torch.zeros_like(c1),
            "phi56": torch.zeros_like(c1),
        }
        return {"source": source, "image": image}

    def decode(self, vector: torch.Tensor | None = None) -> dict[str, dict[str, torch.Tensor]]:
        return self.decode_values(
            self.control_vector if vector is None else vector,
            dtype=self.control_vector.dtype,
            device=self.control_vector.device,
        )


class ActiveFixedStackTEMControls(FixedStackTEMControls):
    """Train only a selected subset of fixed-stack TEM controls.

    This keeps a full base microscope state fixed and exposes a smaller trainable
    subspace for inverse fitting. Optional bounds are enforced with a sigmoid map.
    """

    def __init__(
        self,
        *,
        base_vector: torch.Tensor | Sequence[float] | None = None,
        active_names: Sequence[str] = (),
        active_stages: Sequence[str] = (),
        bounds: Mapping[str, tuple[float, float]] | None = None,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        nn.Module.__init__(self)
        base = self._coerce_vector(
            self.zero_vector(dtype=dtype, device=device) if base_vector is None else base_vector,
            dtype=dtype,
            device=device,
        ).reshape(-1)

        selected_names: list[str] = []
        for name in active_names:
            self.entry(name)
            if name not in selected_names:
                selected_names.append(name)
        for stage_name in active_stages:
            for name in self.stage_entry(stage_name).parameter_names:
                if name not in selected_names:
                    selected_names.append(name)
        if not selected_names:
            raise ValueError("At least one active control name or stage must be provided")

        bounds = {} if bounds is None else dict(bounds)
        active_indices = torch.tensor([self.index(name) for name in selected_names], dtype=torch.long, device=device)
        lower = torch.empty(len(selected_names), dtype=dtype, device=device)
        upper = torch.empty(len(selected_names), dtype=dtype, device=device)
        bounded = torch.zeros(len(selected_names), dtype=torch.bool, device=device)
        latent_values = []
        eps = torch.finfo(dtype).eps

        for i, name in enumerate(selected_names):
            value = base[active_indices[i]]
            if name in bounds:
                low, high = bounds[name]
                if not high > low:
                    raise ValueError(f"Invalid bounds for {name!r}: lower={low}, upper={high}")
                lower[i] = low
                upper[i] = high
                bounded[i] = True
                scaled = ((value - low) / (high - low)).clamp(min=eps, max=1.0 - eps)
                latent_values.append(torch.logit(scaled))
            else:
                lower[i] = 0.0
                upper[i] = 0.0
                latent_values.append(value)

        self.register_buffer("base_vector", base.clone())
        self.register_buffer("active_indices", active_indices)
        self.register_buffer("active_lower", lower)
        self.register_buffer("active_upper", upper)
        self.register_buffer("active_bounded", bounded)
        self.active_names = tuple(selected_names)
        self.control_vector = nn.Parameter(torch.stack(latent_values))

    def _transformed_active_values(self) -> torch.Tensor:
        if not bool(self.active_bounded.any()):
            return self.control_vector
        transformed = self.control_vector.clone()
        bounded = self.active_bounded
        transformed[bounded] = self.active_lower[bounded] + (
            self.active_upper[bounded] - self.active_lower[bounded]
        ) * torch.sigmoid(self.control_vector[bounded])
        return transformed

    def vector(self) -> torch.Tensor:
        vector = self.base_vector.clone()
        vector[self.active_indices] = self._transformed_active_values()
        return vector

    def active_values(self) -> dict[str, torch.Tensor]:
        values = self._transformed_active_values()
        return {name: values[i] for i, name in enumerate(self.active_names)}

    def named_values(
        self,
        vector: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return super().named_values(vector=self.vector() if vector is None else vector)

    def stage_values(
        self,
        vector: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return super().stage_values(vector=self.vector() if vector is None else vector)

    def vector_from_stage_values(
        self,
        stage_values: Mapping[str, torch.Tensor | Sequence[float]],
        *,
        base_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().vector_from_stage_values(
            stage_values,
            base_vector=self.vector() if base_vector is None else base_vector,
        )

    def updated_vector(
        self,
        updates: Mapping[str, torch.Tensor | float],
        vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().updated_vector(
            updates,
            vector=self.vector() if vector is None else vector,
        )

    def updated_stage_vector(
        self,
        stage_name: str,
        values: torch.Tensor | Sequence[float],
        vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().updated_stage_vector(
            stage_name,
            values,
            vector=self.vector() if vector is None else vector,
        )

    def decode(self, vector: torch.Tensor | None = None) -> dict[str, dict[str, torch.Tensor]]:
        return self.decode_values(
            self.vector() if vector is None else vector,
            dtype=self.base_vector.dtype,
            device=self.base_vector.device,
        )


class FixedStackTEMSimulator(nn.Module):
    """Differentiable fixed-stack TEM model with vectorized control parameters."""

    def __init__(
        self,
        *,
        controls: FixedStackTEMControls,
        potential: nn.Module,
        energy_eV: float,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        slice_thickness_A: float,
        num_slices: int,
        source_semiangle_cutoff_mrad: float = 5.0,
        image_semiangle_cutoff_mrad: float = 35.0,
        source_soft_edge_mrad: float = 0.0,
        image_soft_edge_mrad: float = 0.0,
        center_illumination: bool = True,
    ) -> None:
        super().__init__()
        self.controls = controls
        self.potential = potential
        self.energy_eV = float(energy_eV)
        self.gpts = gpts
        self.sampling = sampling
        self.slice_thickness_A = float(slice_thickness_A)
        self.num_slices = int(num_slices)
        self.source_semiangle_cutoff_mrad = float(source_semiangle_cutoff_mrad)
        self.image_semiangle_cutoff_mrad = float(image_semiangle_cutoff_mrad)
        self.source_soft_edge_mrad = float(source_soft_edge_mrad)
        self.image_soft_edge_mrad = float(image_soft_edge_mrad)
        self.center_illumination = bool(center_illumination)
        self.multislice = MultisliceSystem(
            energy_eV=self.energy_eV,
            gpts=self.gpts,
            sampling=self.sampling,
            slice_thickness_A=self.slice_thickness_A,
        )

    @property
    def wavelength(self) -> torch.Tensor:
        return energy2wavelength(self.energy_eV, dtype=torch.float64)

    def _infer_named_batch_shape(
        self,
        updates: Mapping[str, torch.Tensor | float],
    ) -> tuple[int, ...]:
        batch_shape: tuple[int, ...] = ()
        for value in updates.values():
            tensor = torch.as_tensor(
                value,
                dtype=self.controls.control_vector.dtype,
                device=self.controls.control_vector.device,
            )
            batch_shape = torch.broadcast_shapes(batch_shape, tuple(tensor.shape))
        return batch_shape

    def _infer_stage_batch_shape(
        self,
        stage_values: Mapping[str, torch.Tensor | Sequence[float]],
    ) -> tuple[int, ...]:
        batch_shape: tuple[int, ...] = ()
        for stage_name, value in stage_values.items():
            stage = self.controls.stage_entry(stage_name)
            tensor = torch.as_tensor(
                value,
                dtype=self.controls.control_vector.dtype,
                device=self.controls.control_vector.device,
            )
            tail = stage.stop - stage.start
            if tensor.ndim == 0:
                raise ValueError(
                    f"Stage update {stage_name!r} must have at least one dimension of size {tail}"
                )
            if tensor.shape[-1] != tail:
                if tensor.ndim == 1 and tensor.numel() == tail:
                    current_batch_shape = ()
                else:
                    raise ValueError(
                        f"Stage update {stage_name!r} has trailing shape {tuple(tensor.shape)} "
                        f"but expected last dimension {tail}"
                    )
            else:
                current_batch_shape = tuple(tensor.shape[:-1])
            batch_shape = torch.broadcast_shapes(batch_shape, current_batch_shape)
        return batch_shape

    def control_layout(self) -> tuple[TEMControlEntry, ...]:
        return self.controls.layout()

    def control_vector(self) -> torch.Tensor:
        return self.controls.vector()

    @staticmethod
    def image_contrast(
        image: torch.Tensor,
        reference_mean: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if reference_mean is None:
            reference_mean = image.mean(dim=(-2, -1), keepdim=True)
        return image / reference_mean.clamp_min(torch.finfo(image.dtype).eps) - 1.0

    @staticmethod
    def normalized_diffraction(diffraction: torch.Tensor) -> torch.Tensor:
        return diffraction / diffraction.sum(dim=(-2, -1), keepdim=True).clamp_min(
            torch.finfo(diffraction.dtype).eps
        )

    @staticmethod
    def _broadcast_wave_target(
        predicted_wave: torch.Tensor,
        target_wave: torch.Tensor,
        *,
        wave_name: str,
    ) -> torch.Tensor:
        if predicted_wave.shape[-2:] != target_wave.shape[-2:]:
            raise ValueError(
                f"Target {wave_name} shape {tuple(target_wave.shape[-2:])} does not match "
                f"candidate {wave_name} shape {tuple(predicted_wave.shape[-2:])}"
            )
        candidate_batch_shape = predicted_wave.shape[:-2]
        expected_shape = candidate_batch_shape + predicted_wave.shape[-2:]
        if target_wave.shape == predicted_wave.shape[-2:]:
            return torch.broadcast_to(target_wave, expected_shape)
        try:
            return torch.broadcast_to(target_wave, expected_shape)
        except RuntimeError as exc:
            raise ValueError(
                f"Target {wave_name} with shape {tuple(target_wave.shape)} cannot broadcast "
                f"to candidate batch {wave_name} shape {tuple(expected_shape)}"
            ) from exc

    @staticmethod
    def align_wave_phase(
        predicted_wave: torch.Tensor,
        target_wave: torch.Tensor,
    ) -> torch.Tensor:
        overlap = torch.sum(
            torch.conj(target_wave) * predicted_wave,
            dim=(-2, -1),
            keepdim=True,
        )
        scale = overlap.abs().clamp_min(torch.finfo(overlap.real.dtype).eps)
        phase = overlap / scale
        return predicted_wave * torch.conj(phase)

    @staticmethod
    def _broadcast_exit_wave_target(
        predicted_exit_wave: torch.Tensor,
        target_exit_wave: torch.Tensor,
    ) -> torch.Tensor:
        return FixedStackTEMSimulator._broadcast_wave_target(
            predicted_exit_wave,
            target_exit_wave,
            wave_name="exit wave",
        )

    @staticmethod
    def _broadcast_source_wave_target(
        predicted_source_wave: torch.Tensor,
        target_source_wave: torch.Tensor,
    ) -> torch.Tensor:
        return FixedStackTEMSimulator._broadcast_wave_target(
            predicted_source_wave,
            target_source_wave,
            wave_name="source wave",
        )

    @staticmethod
    def align_exit_wave_phase(
        predicted_exit_wave: torch.Tensor,
        target_exit_wave: torch.Tensor,
    ) -> torch.Tensor:
        return FixedStackTEMSimulator.align_wave_phase(predicted_exit_wave, target_exit_wave)

    @staticmethod
    def align_source_wave_phase(
        predicted_source_wave: torch.Tensor,
        target_source_wave: torch.Tensor,
    ) -> torch.Tensor:
        return FixedStackTEMSimulator.align_wave_phase(predicted_source_wave, target_source_wave)

    @staticmethod
    def _broadcast_metric_target(
        predicted_metric: torch.Tensor,
        target_metric: torch.Tensor,
        *,
        metric_name: str,
    ) -> torch.Tensor:
        if predicted_metric.shape[-2:] != target_metric.shape[-2:]:
            raise ValueError(
                f"Target {metric_name} shape {tuple(target_metric.shape[-2:])} does not match "
                f"candidate {metric_name} shape {tuple(predicted_metric.shape[-2:])}"
            )

        candidate_batch_shape = predicted_metric.shape[:-2]
        expected_target_shape = candidate_batch_shape + predicted_metric.shape[-2:]
        if target_metric.shape == predicted_metric.shape[-2:]:
            return torch.broadcast_to(target_metric, expected_target_shape)
        try:
            return torch.broadcast_to(target_metric, expected_target_shape)
        except RuntimeError as exc:
            raise ValueError(
                f"Target {metric_name} with shape {tuple(target_metric.shape)} cannot broadcast "
                f"to candidate batch {metric_name} shape {tuple(expected_target_shape)}"
            ) from exc

    @staticmethod
    def _broadcast_reference_mean(
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        target_mean = target.mean(dim=(-2, -1), keepdim=True)
        candidate_batch_shape = predicted.shape[:-2]
        expected_shape = candidate_batch_shape + (1, 1)
        if target_mean.shape == (1, 1) or target_mean.shape == expected_shape:
            return torch.broadcast_to(target_mean, expected_shape)
        target_batch_shape = target_mean.shape[:-2]
        if target_batch_shape == ():
            return torch.broadcast_to(target_mean, expected_shape)
        try:
            return torch.broadcast_to(target_mean, expected_shape)
        except RuntimeError as exc:
            raise ValueError(
                f"Target image mean with shape {tuple(target_mean.shape)} cannot broadcast "
                f"to candidate batch reference shape {tuple(expected_shape)}"
            ) from exc

    def vector_from_named_values(
        self,
        updates: Mapping[str, torch.Tensor | float],
        *,
        base_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if base_vector is None:
            return self.controls.assemble_named_tensor(
                updates,
                base_vector=self.controls.vector(),
                batch_shape=self._infer_named_batch_shape(updates),
                dtype=self.controls.control_vector.dtype,
                device=self.controls.control_vector.device,
            )
        return self.controls.updated_vector(updates, vector=base_vector)

    def vector_from_stage_values(
        self,
        stage_values: Mapping[str, torch.Tensor | Sequence[float]],
        *,
        base_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if base_vector is None:
            return self.controls.assemble_stage_tensor(
                stage_values,
                base_vector=self.controls.vector(),
                batch_shape=self._infer_stage_batch_shape(stage_values),
                dtype=self.controls.control_vector.dtype,
                device=self.controls.control_vector.device,
            )
        return self.controls.vector_from_stage_values(stage_values, base_vector=base_vector)

    def potential_slices(self) -> torch.Tensor:
        potential = self.potential()
        if potential.ndim == 3:
            return potential
        return potential.unsqueeze(0).repeat(self.num_slices, 1, 1) / self.num_slices

    def _angular_grid(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        k, phi = polar_spatial_frequencies(
            self.gpts,
            self.sampling,
            device=device,
            dtype=torch.float64,
        )
        return self.wavelength.to(device) * k, phi

    def _source_wave(self, source_coeffs: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        alpha, phi = self._angular_grid(device)
        transfer = evaluate_ctf_from_coefficients(
            alpha,
            phi,
            wavelength=self.wavelength.to(device),
            coefficients=source_coeffs,
            semiangle_cutoff_mrad=self.source_semiangle_cutoff_mrad,
            soft_edge_mrad=self.source_soft_edge_mrad,
        )
        norm = torch.sqrt(torch.sum(torch.abs(transfer).square(), dim=(-2, -1), keepdim=True))
        reciprocal = transfer / norm.clamp_min(torch.finfo(torch.float64).eps)
        wave = torch.fft.ifft2(reciprocal)
        if self.center_illumination:
            centered_position_A = torch.tensor(
                [self.gpts[0] * self.sampling[0] / 2.0, self.gpts[1] * self.sampling[1] / 2.0],
                dtype=torch.float64,
                device=wave.device,
            )
            wave = fft_shift_wave(wave, centered_position_A, self.sampling)
        return wave

    def _image_from_exit_wave(self, exit_wave: torch.Tensor, image_coeffs: dict[str, torch.Tensor]) -> torch.Tensor:
        alpha, phi = self._angular_grid(exit_wave.device)
        transfer = evaluate_ctf_from_coefficients(
            alpha,
            phi,
            wavelength=self.wavelength.to(exit_wave.device),
            coefficients=image_coeffs,
            semiangle_cutoff_mrad=self.image_semiangle_cutoff_mrad,
            soft_edge_mrad=self.image_soft_edge_mrad,
        )
        reciprocal = torch.fft.fft2(exit_wave)
        image_wave = torch.fft.ifft2(reciprocal * transfer)
        return image_wave.abs().square()

    def _diffraction_from_exit_wave(self, exit_wave: torch.Tensor) -> torch.Tensor:
        return PixelatedDetector(reciprocal_space=True)(exit_wave)

    def _forward_single(self, vector: torch.Tensor) -> dict[str, torch.Tensor]:
        decoded = self.controls.decode(vector)
        potential_slices = self.potential_slices()
        source_wave = self._source_wave(decoded["source"], potential_slices.device)
        exit_wave = self.multislice(source_wave, potential_slices)
        image = self._image_from_exit_wave(exit_wave, decoded["image"])
        diffraction = self._diffraction_from_exit_wave(exit_wave)
        return {
            "source_wave": source_wave,
            "image": image,
            "diffraction": diffraction,
            "exit_wave": exit_wave,
        }

    def forward_from_named_values(
        self,
        updates: Mapping[str, torch.Tensor | float],
        *,
        base_vector: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self(self.vector_from_named_values(updates, base_vector=base_vector))

    def forward_from_stage_values(
        self,
        stage_values: Mapping[str, torch.Tensor | Sequence[float]],
        *,
        base_vector: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self(self.vector_from_stage_values(stage_values, base_vector=base_vector))

    def score_candidate_batch(
        self,
        control_batch: torch.Tensor,
        *,
        target_image: torch.Tensor,
        target_diffraction: torch.Tensor | None = None,
        target_exit_wave: torch.Tensor | None = None,
        target_source_wave: torch.Tensor | None = None,
        use_contrast: bool = True,
        normalize_diffraction: bool = True,
        align_exit_wave_phase: bool = True,
        align_source_wave_phase: bool = True,
        image_weight: float = 1.0,
        diffraction_weight: float = 0.0,
        exit_wave_weight: float = 0.0,
        source_wave_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        if image_weight < 0.0 or diffraction_weight < 0.0 or exit_wave_weight < 0.0 or source_wave_weight < 0.0:
            raise ValueError("image_weight, diffraction_weight, exit_wave_weight, and source_wave_weight must be nonnegative")
        if image_weight == 0.0 and diffraction_weight == 0.0 and exit_wave_weight == 0.0 and source_wave_weight == 0.0:
            raise ValueError("At least one of image_weight, diffraction_weight, exit_wave_weight, or source_wave_weight must be positive")
        outputs = self(control_batch)
        predicted = outputs["image"]
        target = torch.as_tensor(
            target_image,
            dtype=predicted.dtype,
            device=predicted.device,
        )
        if use_contrast:
            reference_mean = self._broadcast_reference_mean(predicted, target)
            predicted_metric = self.image_contrast(predicted, reference_mean=reference_mean)
            target_metric = self.image_contrast(target, reference_mean=target.mean(dim=(-2, -1), keepdim=True))
        else:
            predicted_metric = predicted
            target_metric = target

        target_metric = self._broadcast_metric_target(
            predicted_metric,
            target_metric,
            metric_name="image",
        )
        image_residual = predicted_metric - target_metric
        image_scores = image_residual.square().mean(dim=(-2, -1))

        diffraction_scores = torch.zeros_like(image_scores)
        predicted_diffraction_metric = None
        target_diffraction_metric = None
        if target_diffraction is not None or diffraction_weight > 0.0:
            if target_diffraction is None:
                raise ValueError("target_diffraction must be provided when diffraction_weight > 0")
            predicted_diffraction = outputs["diffraction"]
            target_diffraction_tensor = torch.as_tensor(
                target_diffraction,
                dtype=predicted_diffraction.dtype,
                device=predicted_diffraction.device,
            )
            if normalize_diffraction:
                predicted_diffraction_metric = self.normalized_diffraction(predicted_diffraction)
                target_diffraction_metric = self.normalized_diffraction(target_diffraction_tensor)
            else:
                predicted_diffraction_metric = predicted_diffraction
                target_diffraction_metric = target_diffraction_tensor
            target_diffraction_metric = self._broadcast_metric_target(
                predicted_diffraction_metric,
                target_diffraction_metric,
                metric_name="diffraction",
            )
            diffraction_residual = predicted_diffraction_metric - target_diffraction_metric
            diffraction_scores = diffraction_residual.square().mean(dim=(-2, -1))

        exit_wave_scores = torch.zeros_like(image_scores)
        predicted_exit_wave_metric = None
        target_exit_wave_metric = None
        if target_exit_wave is not None or exit_wave_weight > 0.0:
            if target_exit_wave is None:
                raise ValueError("target_exit_wave must be provided when exit_wave_weight > 0")
            predicted_exit_wave = outputs["exit_wave"]
            target_exit_wave_tensor = torch.as_tensor(
                target_exit_wave,
                dtype=predicted_exit_wave.dtype,
                device=predicted_exit_wave.device,
            )
            target_exit_wave_metric = self._broadcast_exit_wave_target(
                predicted_exit_wave,
                target_exit_wave_tensor,
            )
            predicted_exit_wave_metric = (
                self.align_exit_wave_phase(predicted_exit_wave, target_exit_wave_metric)
                if align_exit_wave_phase
                else predicted_exit_wave
            )
            exit_wave_residual = predicted_exit_wave_metric - target_exit_wave_metric
            exit_wave_scores = exit_wave_residual.abs().square().mean(dim=(-2, -1))

        source_wave_scores = torch.zeros_like(image_scores)
        predicted_source_wave_metric = None
        target_source_wave_metric = None
        if target_source_wave is not None or source_wave_weight > 0.0:
            if target_source_wave is None:
                raise ValueError("target_source_wave must be provided when source_wave_weight > 0")
            predicted_source_wave = outputs["source_wave"]
            target_source_wave_tensor = torch.as_tensor(
                target_source_wave,
                dtype=predicted_source_wave.dtype,
                device=predicted_source_wave.device,
            )
            target_source_wave_metric = self._broadcast_source_wave_target(
                predicted_source_wave,
                target_source_wave_tensor,
            )
            predicted_source_wave_metric = (
                self.align_source_wave_phase(predicted_source_wave, target_source_wave_metric)
                if align_source_wave_phase
                else predicted_source_wave
            )
            source_wave_residual = predicted_source_wave_metric - target_source_wave_metric
            source_wave_scores = source_wave_residual.abs().square().mean(dim=(-2, -1))

        scores = (
            image_weight * image_scores
            + diffraction_weight * diffraction_scores
            + exit_wave_weight * exit_wave_scores
            + source_wave_weight * source_wave_scores
        )
        return {
            "scores": scores,
            "predicted": predicted_metric,
            "target": target_metric,
            "image_scores": image_scores,
            "predicted_diffraction": predicted_diffraction_metric,
            "target_diffraction": target_diffraction_metric,
            "diffraction_scores": diffraction_scores,
            "predicted_exit_wave": predicted_exit_wave_metric,
            "target_exit_wave": target_exit_wave_metric,
            "exit_wave_scores": exit_wave_scores,
            "predicted_source_wave": predicted_source_wave_metric,
            "target_source_wave": target_source_wave_metric,
            "source_wave_scores": source_wave_scores,
            "outputs": outputs,
        }

    def loss_against_targets(
        self,
        outputs: dict[str, torch.Tensor],
        *,
        target_image: torch.Tensor,
        target_diffraction: torch.Tensor | None = None,
        target_exit_wave: torch.Tensor | None = None,
        target_source_wave: torch.Tensor | None = None,
        use_contrast: bool = True,
        normalize_diffraction: bool = True,
        align_exit_wave_phase: bool = True,
        align_source_wave_phase: bool = True,
        image_weight: float = 1.0,
        diffraction_weight: float = 0.0,
        exit_wave_weight: float = 0.0,
        source_wave_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        if image_weight < 0.0 or diffraction_weight < 0.0 or exit_wave_weight < 0.0 or source_wave_weight < 0.0:
            raise ValueError("image_weight, diffraction_weight, exit_wave_weight, and source_wave_weight must be nonnegative")
        if image_weight == 0.0 and diffraction_weight == 0.0 and exit_wave_weight == 0.0 and source_wave_weight == 0.0:
            raise ValueError("At least one of image_weight, diffraction_weight, exit_wave_weight, or source_wave_weight must be positive")

        predicted = outputs["image"]
        target = torch.as_tensor(
            target_image,
            dtype=predicted.dtype,
            device=predicted.device,
        )
        if use_contrast:
            reference_mean = self._broadcast_reference_mean(predicted, target)
            predicted_metric = self.image_contrast(predicted, reference_mean=reference_mean)
            target_metric = self.image_contrast(target, reference_mean=target.mean(dim=(-2, -1), keepdim=True))
        else:
            predicted_metric = predicted
            target_metric = target
        target_metric = self._broadcast_metric_target(
            predicted_metric,
            target_metric,
            metric_name="image",
        )
        image_residual = predicted_metric - target_metric
        image_loss = image_residual.square().mean(dim=(-2, -1))

        diffraction_loss = torch.zeros_like(image_loss)
        predicted_diffraction_metric = None
        target_diffraction_metric = None
        if target_diffraction is not None or diffraction_weight > 0.0:
            if target_diffraction is None:
                raise ValueError("target_diffraction must be provided when diffraction_weight > 0")
            predicted_diffraction = outputs["diffraction"]
            target_diffraction_tensor = torch.as_tensor(
                target_diffraction,
                dtype=predicted_diffraction.dtype,
                device=predicted_diffraction.device,
            )
            if normalize_diffraction:
                predicted_diffraction_metric = self.normalized_diffraction(predicted_diffraction)
                target_diffraction_metric = self.normalized_diffraction(target_diffraction_tensor)
            else:
                predicted_diffraction_metric = predicted_diffraction
                target_diffraction_metric = target_diffraction_tensor
            target_diffraction_metric = self._broadcast_metric_target(
                predicted_diffraction_metric,
                target_diffraction_metric,
                metric_name="diffraction",
            )
            diffraction_residual = predicted_diffraction_metric - target_diffraction_metric
            diffraction_loss = diffraction_residual.square().mean(dim=(-2, -1))
        exit_wave_loss = torch.zeros_like(image_loss)
        predicted_exit_wave_metric = None
        target_exit_wave_metric = None
        if target_exit_wave is not None or exit_wave_weight > 0.0:
            if target_exit_wave is None:
                raise ValueError("target_exit_wave must be provided when exit_wave_weight > 0")
            predicted_exit_wave = outputs["exit_wave"]
            target_exit_wave_tensor = torch.as_tensor(
                target_exit_wave,
                dtype=predicted_exit_wave.dtype,
                device=predicted_exit_wave.device,
            )
            target_exit_wave_metric = self._broadcast_exit_wave_target(
                predicted_exit_wave,
                target_exit_wave_tensor,
            )
            predicted_exit_wave_metric = (
                self.align_exit_wave_phase(predicted_exit_wave, target_exit_wave_metric)
                if align_exit_wave_phase
                else predicted_exit_wave
            )
            exit_wave_residual = predicted_exit_wave_metric - target_exit_wave_metric
            exit_wave_loss = exit_wave_residual.abs().square().mean(dim=(-2, -1))
        source_wave_loss = torch.zeros_like(image_loss)
        predicted_source_wave_metric = None
        target_source_wave_metric = None
        if target_source_wave is not None or source_wave_weight > 0.0:
            if target_source_wave is None:
                raise ValueError("target_source_wave must be provided when source_wave_weight > 0")
            predicted_source_wave = outputs["source_wave"]
            target_source_wave_tensor = torch.as_tensor(
                target_source_wave,
                dtype=predicted_source_wave.dtype,
                device=predicted_source_wave.device,
            )
            target_source_wave_metric = self._broadcast_source_wave_target(
                predicted_source_wave,
                target_source_wave_tensor,
            )
            predicted_source_wave_metric = (
                self.align_source_wave_phase(predicted_source_wave, target_source_wave_metric)
                if align_source_wave_phase
                else predicted_source_wave
            )
            source_wave_residual = predicted_source_wave_metric - target_source_wave_metric
            source_wave_loss = source_wave_residual.abs().square().mean(dim=(-2, -1))
        loss = (
            image_weight * image_loss
            + diffraction_weight * diffraction_loss
            + exit_wave_weight * exit_wave_loss
            + source_wave_weight * source_wave_loss
        )
        return {
            "loss": loss,
            "image_loss": image_loss,
            "diffraction_loss": diffraction_loss,
            "exit_wave_loss": exit_wave_loss,
            "source_wave_loss": source_wave_loss,
            "predicted_image": predicted_metric,
            "target_image": target_metric,
            "predicted_diffraction": predicted_diffraction_metric,
            "target_diffraction": target_diffraction_metric,
            "predicted_exit_wave": predicted_exit_wave_metric,
            "target_exit_wave": target_exit_wave_metric,
            "predicted_source_wave": predicted_source_wave_metric,
            "target_source_wave": target_source_wave_metric,
            "outputs": outputs,
        }

    def optimize_to_match(
        self,
        *,
        target_image: torch.Tensor,
        target_diffraction: torch.Tensor | None = None,
        target_exit_wave: torch.Tensor | None = None,
        target_source_wave: torch.Tensor | None = None,
        steps: int,
        lr: float,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Mapping[str, object] | None = None,
        use_contrast: bool = True,
        normalize_diffraction: bool = True,
        align_exit_wave_phase: bool = True,
        align_source_wave_phase: bool = True,
        image_weight: float = 1.0,
        diffraction_weight: float = 0.0,
        exit_wave_weight: float = 0.0,
        source_wave_weight: float = 0.0,
        callback=None,
    ) -> dict[str, object]:
        if steps <= 0:
            raise ValueError(f"steps must be positive, got {steps}")
        optimizer_kwargs = {} if optimizer_kwargs is None else dict(optimizer_kwargs)
        optimizer = optimizer_cls(self.parameters(), lr=lr, **optimizer_kwargs)
        initial_outputs = {
            name: value.detach().clone()
            for name, value in self().items()
        }
        history: list[float] = []
        image_history: list[float] = []
        diffraction_history: list[float] = []
        exit_wave_history: list[float] = []
        source_wave_history: list[float] = []

        for step in range(steps):
            optimizer.zero_grad()
            outputs = self()
            losses = self.loss_against_targets(
                outputs,
                target_image=target_image,
                target_diffraction=target_diffraction,
                target_exit_wave=target_exit_wave,
                target_source_wave=target_source_wave,
                use_contrast=use_contrast,
                normalize_diffraction=normalize_diffraction,
                align_exit_wave_phase=align_exit_wave_phase,
                align_source_wave_phase=align_source_wave_phase,
                image_weight=image_weight,
                diffraction_weight=diffraction_weight,
                exit_wave_weight=exit_wave_weight,
                source_wave_weight=source_wave_weight,
            )
            loss = losses["loss"]
            image_loss = losses["image_loss"]
            diffraction_loss = losses["diffraction_loss"]
            exit_wave_loss = losses["exit_wave_loss"]
            source_wave_loss = losses["source_wave_loss"]
            loss.backward()
            optimizer.step()

            history.append(float(loss.detach()))
            image_history.append(float(image_loss.detach()))
            diffraction_history.append(float(diffraction_loss.detach()))
            exit_wave_history.append(float(exit_wave_loss.detach()))
            source_wave_history.append(float(source_wave_loss.detach()))

            if callback is not None:
                callback(
                    step,
                    {
                        "loss": loss.detach(),
                        "image_loss": image_loss.detach(),
                        "diffraction_loss": diffraction_loss.detach(),
                        "exit_wave_loss": exit_wave_loss.detach(),
                        "source_wave_loss": source_wave_loss.detach(),
                        "predicted_image": losses["predicted_image"].detach(),
                        "target_image": losses["target_image"].detach(),
                        "predicted_diffraction": None
                        if losses["predicted_diffraction"] is None
                        else losses["predicted_diffraction"].detach(),
                        "target_diffraction": None
                        if losses["target_diffraction"] is None
                        else losses["target_diffraction"].detach(),
                        "predicted_exit_wave": None
                        if losses["predicted_exit_wave"] is None
                        else losses["predicted_exit_wave"].detach(),
                        "target_exit_wave": None
                        if losses["target_exit_wave"] is None
                        else losses["target_exit_wave"].detach(),
                        "predicted_source_wave": None
                        if losses["predicted_source_wave"] is None
                        else losses["predicted_source_wave"].detach(),
                        "target_source_wave": None
                        if losses["target_source_wave"] is None
                        else losses["target_source_wave"].detach(),
                        "outputs": {
                            name: value.detach()
                            for name, value in outputs.items()
                        },
                        "control_vector": self.control_vector().detach(),
                    },
                )

        final_outputs = {
            name: value.detach().clone()
            for name, value in self().items()
        }
        return {
            "history": history,
            "image_history": image_history,
            "diffraction_history": diffraction_history,
            "exit_wave_history": exit_wave_history,
            "source_wave_history": source_wave_history,
            "initial_outputs": initial_outputs,
            "final_outputs": final_outputs,
            "final_vector": self.control_vector().detach().clone(),
        }

    def coarse_to_fine_fit_from_named_parameter_grid(
        self,
        parameter_values: Mapping[str, torch.Tensor | Sequence[float]],
        *,
        target_image: torch.Tensor,
        target_diffraction: torch.Tensor | None = None,
        target_exit_wave: torch.Tensor | None = None,
        target_source_wave: torch.Tensor | None = None,
        k: int,
        steps: int,
        lr: float,
        use_contrast: bool = True,
        normalize_diffraction: bool = True,
        align_exit_wave_phase: bool = True,
        align_source_wave_phase: bool = True,
        image_weight: float = 1.0,
        diffraction_weight: float = 0.0,
        exit_wave_weight: float = 0.0,
        source_wave_weight: float = 0.0,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Mapping[str, object] | None = None,
        callback=None,
    ) -> dict[str, object]:
        screened = self.select_top_from_named_parameter_grid(
            parameter_values,
            target_image=target_image,
            target_diffraction=target_diffraction,
            target_exit_wave=target_exit_wave,
            target_source_wave=target_source_wave,
            k=k,
            use_contrast=use_contrast,
            normalize_diffraction=normalize_diffraction,
            align_exit_wave_phase=align_exit_wave_phase,
            align_source_wave_phase=align_source_wave_phase,
            image_weight=image_weight,
            diffraction_weight=diffraction_weight,
            exit_wave_weight=exit_wave_weight,
            source_wave_weight=source_wave_weight,
        )
        screened_initial_vector = screened["top_vectors"][0].detach()

        if isinstance(self.controls, ActiveFixedStackTEMControls):
            fit_controls = ActiveFixedStackTEMControls(
                base_vector=screened_initial_vector,
                active_names=self.controls.active_names,
                bounds={
                    name: (
                        float(self.controls.active_lower[i].detach().cpu()),
                        float(self.controls.active_upper[i].detach().cpu()),
                    )
                    for i, name in enumerate(self.controls.active_names)
                    if bool(self.controls.active_bounded[i].item())
                },
                dtype=self.controls.base_vector.dtype,
                device=self.controls.base_vector.device,
            )
        else:
            fit_controls = FixedStackTEMControls(
                initial_vector=screened_initial_vector,
                dtype=self.controls.control_vector.dtype,
                device=self.controls.control_vector.device,
            )

        fit_simulator = FixedStackTEMSimulator(
            controls=fit_controls,
            potential=self.potential,
            energy_eV=self.energy_eV,
            gpts=self.gpts,
            sampling=self.sampling,
            slice_thickness_A=self.slice_thickness_A,
            num_slices=self.num_slices,
            source_semiangle_cutoff_mrad=self.source_semiangle_cutoff_mrad,
            image_semiangle_cutoff_mrad=self.image_semiangle_cutoff_mrad,
            source_soft_edge_mrad=self.source_soft_edge_mrad,
            image_soft_edge_mrad=self.image_soft_edge_mrad,
            center_illumination=self.center_illumination,
        )
        optimization = fit_simulator.optimize_to_match(
            target_image=target_image,
            target_diffraction=target_diffraction,
            target_exit_wave=target_exit_wave,
            target_source_wave=target_source_wave,
            steps=steps,
            lr=lr,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            use_contrast=use_contrast,
            normalize_diffraction=normalize_diffraction,
            align_exit_wave_phase=align_exit_wave_phase,
            align_source_wave_phase=align_source_wave_phase,
            image_weight=image_weight,
            diffraction_weight=diffraction_weight,
            exit_wave_weight=exit_wave_weight,
            source_wave_weight=source_wave_weight,
            callback=callback,
        )
        optimization["screened"] = screened
        optimization["screened_initial_vector"] = screened_initial_vector.detach().clone()
        optimization["fit_simulator"] = fit_simulator
        return optimization

    def score_candidate_batch_from_named_values(
        self,
        updates: Mapping[str, torch.Tensor | float],
        *,
        target_image: torch.Tensor,
        target_diffraction: torch.Tensor | None = None,
        target_exit_wave: torch.Tensor | None = None,
        target_source_wave: torch.Tensor | None = None,
        base_vector: torch.Tensor | None = None,
        use_contrast: bool = True,
        normalize_diffraction: bool = True,
        align_exit_wave_phase: bool = True,
        align_source_wave_phase: bool = True,
        image_weight: float = 1.0,
        diffraction_weight: float = 0.0,
        exit_wave_weight: float = 0.0,
        source_wave_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        control_batch = self.vector_from_named_values(updates, base_vector=base_vector)
        return self.score_candidate_batch(
            control_batch,
            target_image=target_image,
            target_diffraction=target_diffraction,
            target_exit_wave=target_exit_wave,
            target_source_wave=target_source_wave,
            use_contrast=use_contrast,
            normalize_diffraction=normalize_diffraction,
            align_exit_wave_phase=align_exit_wave_phase,
            align_source_wave_phase=align_source_wave_phase,
            image_weight=image_weight,
            diffraction_weight=diffraction_weight,
            exit_wave_weight=exit_wave_weight,
            source_wave_weight=source_wave_weight,
        )

    def score_named_parameter_grid(
        self,
        parameter_values: Mapping[str, torch.Tensor | Sequence[float]],
        *,
        target_image: torch.Tensor,
        target_diffraction: torch.Tensor | None = None,
        target_exit_wave: torch.Tensor | None = None,
        target_source_wave: torch.Tensor | None = None,
        base_vector: torch.Tensor | None = None,
        use_contrast: bool = True,
        normalize_diffraction: bool = True,
        align_exit_wave_phase: bool = True,
        align_source_wave_phase: bool = True,
        image_weight: float = 1.0,
        diffraction_weight: float = 0.0,
        exit_wave_weight: float = 0.0,
        source_wave_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        control_batch = self.controls.named_parameter_grid(
            parameter_values,
            base_vector=self.controls.vector() if base_vector is None else base_vector,
            dtype=self.controls.control_vector.dtype,
            device=self.controls.control_vector.device,
        )
        return self.score_candidate_batch(
            control_batch,
            target_image=target_image,
            target_diffraction=target_diffraction,
            target_exit_wave=target_exit_wave,
            target_source_wave=target_source_wave,
            use_contrast=use_contrast,
            normalize_diffraction=normalize_diffraction,
            align_exit_wave_phase=align_exit_wave_phase,
            align_source_wave_phase=align_source_wave_phase,
            image_weight=image_weight,
            diffraction_weight=diffraction_weight,
            exit_wave_weight=exit_wave_weight,
            source_wave_weight=source_wave_weight,
        )

    def score_candidate_batch_from_stage_values(
        self,
        stage_values: Mapping[str, torch.Tensor | Sequence[float]],
        *,
        target_image: torch.Tensor,
        target_diffraction: torch.Tensor | None = None,
        target_exit_wave: torch.Tensor | None = None,
        target_source_wave: torch.Tensor | None = None,
        base_vector: torch.Tensor | None = None,
        use_contrast: bool = True,
        normalize_diffraction: bool = True,
        align_exit_wave_phase: bool = True,
        align_source_wave_phase: bool = True,
        image_weight: float = 1.0,
        diffraction_weight: float = 0.0,
        exit_wave_weight: float = 0.0,
        source_wave_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        control_batch = self.vector_from_stage_values(stage_values, base_vector=base_vector)
        return self.score_candidate_batch(
            control_batch,
            target_image=target_image,
            target_diffraction=target_diffraction,
            target_exit_wave=target_exit_wave,
            target_source_wave=target_source_wave,
            use_contrast=use_contrast,
            normalize_diffraction=normalize_diffraction,
            align_exit_wave_phase=align_exit_wave_phase,
            align_source_wave_phase=align_source_wave_phase,
            image_weight=image_weight,
            diffraction_weight=diffraction_weight,
            exit_wave_weight=exit_wave_weight,
            source_wave_weight=source_wave_weight,
        )

    def select_best_candidate(
        self,
        control_batch: torch.Tensor,
        *,
        target_image: torch.Tensor,
        target_diffraction: torch.Tensor | None = None,
        target_exit_wave: torch.Tensor | None = None,
        target_source_wave: torch.Tensor | None = None,
        use_contrast: bool = True,
        normalize_diffraction: bool = True,
        align_exit_wave_phase: bool = True,
        align_source_wave_phase: bool = True,
        image_weight: float = 1.0,
        diffraction_weight: float = 0.0,
        exit_wave_weight: float = 0.0,
        source_wave_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        scored = self.score_candidate_batch(
            control_batch,
            target_image=target_image,
            target_diffraction=target_diffraction,
            target_exit_wave=target_exit_wave,
            target_source_wave=target_source_wave,
            use_contrast=use_contrast,
            normalize_diffraction=normalize_diffraction,
            align_exit_wave_phase=align_exit_wave_phase,
            align_source_wave_phase=align_source_wave_phase,
            image_weight=image_weight,
            diffraction_weight=diffraction_weight,
            exit_wave_weight=exit_wave_weight,
            source_wave_weight=source_wave_weight,
        )
        scores = scored["scores"]
        best_flat = torch.argmin(scores.reshape(-1))
        batch_shape = scores.shape
        best_index = torch.unravel_index(best_flat, batch_shape)
        best_vector = control_batch[best_index]
        best_outputs = {
            name: value[best_index]
            for name, value in scored["outputs"].items()
        }
        return {
            "best_score": scores[best_index],
            "best_index": torch.tensor(best_index, dtype=torch.long, device=scores.device),
            "best_vector": best_vector,
            "best_outputs": best_outputs,
            "scores": scores,
        }

    def select_top_candidates(
        self,
        control_batch: torch.Tensor,
        *,
        target_image: torch.Tensor,
        k: int,
        use_contrast: bool = True,
        normalize_diffraction: bool = True,
        align_exit_wave_phase: bool = True,
        align_source_wave_phase: bool = True,
        image_weight: float = 1.0,
        diffraction_weight: float = 0.0,
        target_diffraction: torch.Tensor | None = None,
        target_exit_wave: torch.Tensor | None = None,
        exit_wave_weight: float = 0.0,
        target_source_wave: torch.Tensor | None = None,
        source_wave_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        scored = self.score_candidate_batch(
            control_batch,
            target_image=target_image,
            target_diffraction=target_diffraction,
            target_exit_wave=target_exit_wave,
            target_source_wave=target_source_wave,
            use_contrast=use_contrast,
            normalize_diffraction=normalize_diffraction,
            align_exit_wave_phase=align_exit_wave_phase,
            align_source_wave_phase=align_source_wave_phase,
            image_weight=image_weight,
            diffraction_weight=diffraction_weight,
            exit_wave_weight=exit_wave_weight,
            source_wave_weight=source_wave_weight,
        )
        scores = scored["scores"]
        flat_scores = scores.reshape(-1)
        k = min(k, flat_scores.numel())
        top_scores, top_flat_indices = torch.topk(flat_scores, k=k, largest=False)
        batch_shape = scores.shape
        multi_indices = []
        top_vectors = []
        top_outputs = {
            name: []
            for name in scored["outputs"]
        }
        for flat_index in top_flat_indices.tolist():
            unravelled = torch.unravel_index(
                torch.tensor(flat_index, device=scores.device),
                batch_shape,
            )
            multi_index = tuple(int(i) for i in unravelled)
            multi_indices.append(multi_index)
            top_vectors.append(control_batch[multi_index])
            for name, value in scored["outputs"].items():
                top_outputs[name].append(value[multi_index])

        if len(batch_shape) == 0:
            top_indices = torch.empty((k, 0), dtype=torch.long, device=scores.device)
        else:
            top_indices = torch.tensor(multi_indices, dtype=torch.long, device=scores.device)
        stacked_outputs = {
            name: torch.stack(values, dim=0)
            for name, values in top_outputs.items()
        }
        return {
            "top_scores": top_scores,
            "top_flat_indices": top_flat_indices,
            "top_indices": top_indices,
            "top_vectors": torch.stack(top_vectors, dim=0),
            "top_outputs": stacked_outputs,
            "scores": scores,
        }

    def select_best_candidate_from_named_values(
        self,
        updates: Mapping[str, torch.Tensor | float],
        *,
        target_image: torch.Tensor,
        target_diffraction: torch.Tensor | None = None,
        target_exit_wave: torch.Tensor | None = None,
        target_source_wave: torch.Tensor | None = None,
        base_vector: torch.Tensor | None = None,
        use_contrast: bool = True,
        normalize_diffraction: bool = True,
        align_exit_wave_phase: bool = True,
        align_source_wave_phase: bool = True,
        image_weight: float = 1.0,
        diffraction_weight: float = 0.0,
        exit_wave_weight: float = 0.0,
        source_wave_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        control_batch = self.vector_from_named_values(updates, base_vector=base_vector)
        return self.select_best_candidate(
            control_batch,
            target_image=target_image,
            target_diffraction=target_diffraction,
            target_exit_wave=target_exit_wave,
            target_source_wave=target_source_wave,
            use_contrast=use_contrast,
            normalize_diffraction=normalize_diffraction,
            align_exit_wave_phase=align_exit_wave_phase,
            align_source_wave_phase=align_source_wave_phase,
            image_weight=image_weight,
            diffraction_weight=diffraction_weight,
            exit_wave_weight=exit_wave_weight,
            source_wave_weight=source_wave_weight,
        )

    def select_top_candidates_from_named_values(
        self,
        updates: Mapping[str, torch.Tensor | float],
        *,
        target_image: torch.Tensor,
        k: int,
        target_diffraction: torch.Tensor | None = None,
        target_exit_wave: torch.Tensor | None = None,
        target_source_wave: torch.Tensor | None = None,
        base_vector: torch.Tensor | None = None,
        use_contrast: bool = True,
        normalize_diffraction: bool = True,
        align_exit_wave_phase: bool = True,
        align_source_wave_phase: bool = True,
        image_weight: float = 1.0,
        diffraction_weight: float = 0.0,
        exit_wave_weight: float = 0.0,
        source_wave_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        control_batch = self.vector_from_named_values(updates, base_vector=base_vector)
        return self.select_top_candidates(
            control_batch,
            target_image=target_image,
            k=k,
            use_contrast=use_contrast,
            normalize_diffraction=normalize_diffraction,
            align_exit_wave_phase=align_exit_wave_phase,
            align_source_wave_phase=align_source_wave_phase,
            image_weight=image_weight,
            diffraction_weight=diffraction_weight,
            target_diffraction=target_diffraction,
            target_exit_wave=target_exit_wave,
            exit_wave_weight=exit_wave_weight,
            target_source_wave=target_source_wave,
            source_wave_weight=source_wave_weight,
        )

    def select_best_from_named_parameter_grid(
        self,
        parameter_values: Mapping[str, torch.Tensor | Sequence[float]],
        *,
        target_image: torch.Tensor,
        target_diffraction: torch.Tensor | None = None,
        target_exit_wave: torch.Tensor | None = None,
        target_source_wave: torch.Tensor | None = None,
        base_vector: torch.Tensor | None = None,
        use_contrast: bool = True,
        normalize_diffraction: bool = True,
        align_exit_wave_phase: bool = True,
        align_source_wave_phase: bool = True,
        image_weight: float = 1.0,
        diffraction_weight: float = 0.0,
        exit_wave_weight: float = 0.0,
        source_wave_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        control_batch = self.controls.named_parameter_grid(
            parameter_values,
            base_vector=self.controls.vector() if base_vector is None else base_vector,
            dtype=self.controls.control_vector.dtype,
            device=self.controls.control_vector.device,
        )
        return self.select_best_candidate(
            control_batch,
            target_image=target_image,
            target_diffraction=target_diffraction,
            target_exit_wave=target_exit_wave,
            target_source_wave=target_source_wave,
            use_contrast=use_contrast,
            normalize_diffraction=normalize_diffraction,
            align_exit_wave_phase=align_exit_wave_phase,
            align_source_wave_phase=align_source_wave_phase,
            image_weight=image_weight,
            diffraction_weight=diffraction_weight,
            exit_wave_weight=exit_wave_weight,
            source_wave_weight=source_wave_weight,
        )

    def select_top_from_named_parameter_grid(
        self,
        parameter_values: Mapping[str, torch.Tensor | Sequence[float]],
        *,
        target_image: torch.Tensor,
        k: int,
        target_diffraction: torch.Tensor | None = None,
        target_exit_wave: torch.Tensor | None = None,
        target_source_wave: torch.Tensor | None = None,
        base_vector: torch.Tensor | None = None,
        use_contrast: bool = True,
        normalize_diffraction: bool = True,
        align_exit_wave_phase: bool = True,
        align_source_wave_phase: bool = True,
        image_weight: float = 1.0,
        diffraction_weight: float = 0.0,
        exit_wave_weight: float = 0.0,
        source_wave_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        control_batch = self.controls.named_parameter_grid(
            parameter_values,
            base_vector=self.controls.vector() if base_vector is None else base_vector,
            dtype=self.controls.control_vector.dtype,
            device=self.controls.control_vector.device,
        )
        return self.select_top_candidates(
            control_batch,
            target_image=target_image,
            k=k,
            use_contrast=use_contrast,
            normalize_diffraction=normalize_diffraction,
            align_exit_wave_phase=align_exit_wave_phase,
            align_source_wave_phase=align_source_wave_phase,
            image_weight=image_weight,
            diffraction_weight=diffraction_weight,
            target_diffraction=target_diffraction,
            target_exit_wave=target_exit_wave,
            exit_wave_weight=exit_wave_weight,
            target_source_wave=target_source_wave,
            source_wave_weight=source_wave_weight,
        )

    def select_best_candidate_from_stage_values(
        self,
        stage_values: Mapping[str, torch.Tensor | Sequence[float]],
        *,
        target_image: torch.Tensor,
        target_diffraction: torch.Tensor | None = None,
        target_exit_wave: torch.Tensor | None = None,
        target_source_wave: torch.Tensor | None = None,
        base_vector: torch.Tensor | None = None,
        use_contrast: bool = True,
        normalize_diffraction: bool = True,
        align_exit_wave_phase: bool = True,
        align_source_wave_phase: bool = True,
        image_weight: float = 1.0,
        diffraction_weight: float = 0.0,
        exit_wave_weight: float = 0.0,
        source_wave_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        control_batch = self.vector_from_stage_values(stage_values, base_vector=base_vector)
        return self.select_best_candidate(
            control_batch,
            target_image=target_image,
            target_diffraction=target_diffraction,
            target_exit_wave=target_exit_wave,
            target_source_wave=target_source_wave,
            use_contrast=use_contrast,
            normalize_diffraction=normalize_diffraction,
            align_exit_wave_phase=align_exit_wave_phase,
            align_source_wave_phase=align_source_wave_phase,
            image_weight=image_weight,
            diffraction_weight=diffraction_weight,
            exit_wave_weight=exit_wave_weight,
            source_wave_weight=source_wave_weight,
        )

    def select_top_candidates_from_stage_values(
        self,
        stage_values: Mapping[str, torch.Tensor | Sequence[float]],
        *,
        target_image: torch.Tensor,
        k: int,
        target_diffraction: torch.Tensor | None = None,
        target_exit_wave: torch.Tensor | None = None,
        target_source_wave: torch.Tensor | None = None,
        base_vector: torch.Tensor | None = None,
        use_contrast: bool = True,
        normalize_diffraction: bool = True,
        align_exit_wave_phase: bool = True,
        align_source_wave_phase: bool = True,
        image_weight: float = 1.0,
        diffraction_weight: float = 0.0,
        exit_wave_weight: float = 0.0,
        source_wave_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        control_batch = self.vector_from_stage_values(stage_values, base_vector=base_vector)
        return self.select_top_candidates(
            control_batch,
            target_image=target_image,
            k=k,
            use_contrast=use_contrast,
            normalize_diffraction=normalize_diffraction,
            align_exit_wave_phase=align_exit_wave_phase,
            align_source_wave_phase=align_source_wave_phase,
            image_weight=image_weight,
            diffraction_weight=diffraction_weight,
            target_diffraction=target_diffraction,
            target_exit_wave=target_exit_wave,
            exit_wave_weight=exit_wave_weight,
            target_source_wave=target_source_wave,
            source_wave_weight=source_wave_weight,
        )

    def forward(self, control_vector: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        vector = self.controls.vector() if control_vector is None else torch.as_tensor(
            control_vector,
            dtype=self.controls.control_vector.dtype,
            device=self.controls.control_vector.device,
        )
        if vector.ndim == 1:
            return self._forward_single(vector)
        batch_shape = vector.shape[:-1]
        flat_vectors = vector.reshape(-1, vector.shape[-1])
        outputs = [self._forward_single(flat_vectors[i]) for i in range(flat_vectors.shape[0])]
        return {
            name: torch.stack([output[name] for output in outputs], dim=0).reshape(
                batch_shape + outputs[0][name].shape
            )
            for name in outputs[0]
        }
