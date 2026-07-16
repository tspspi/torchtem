from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from torchtem.multislice import FresnelPropagator


def wrapped_patch_indices(
    position_px: torch.Tensor,
    patch_shape: tuple[int, int],
    object_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    y = (torch.arange(patch_shape[0], device=position_px.device) + position_px[0].long()) % object_shape[0]
    x = (torch.arange(patch_shape[1], device=position_px.device) + position_px[1].long()) % object_shape[1]
    return y, x


def extract_patch(
    obj: torch.Tensor,
    position_px: torch.Tensor,
    patch_shape: tuple[int, int],
) -> torch.Tensor:
    y, x = wrapped_patch_indices(position_px, patch_shape, tuple(obj.shape[-2:]))
    return obj.index_select(-2, y).index_select(-1, x)


def assign_patch_add(
    obj: torch.Tensor,
    position_px: torch.Tensor,
    patch_update: torch.Tensor,
) -> torch.Tensor:
    out = obj.clone()
    y, x = wrapped_patch_indices(position_px, tuple(patch_update.shape[-2:]), tuple(obj.shape[-2:]))
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    out[yy, xx] = out[yy, xx] + patch_update
    return out


class PtychographyForwardModel(nn.Module):
    """Periodic single-slice ptychography forward model."""

    def __init__(self, positions_px: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("positions_px", positions_px.to(torch.int64))

    def exit_waves(self, obj: torch.Tensor, probe: torch.Tensor) -> torch.Tensor:
        patch_shape = tuple(probe.shape[-2:])
        return torch.stack(
            [extract_patch(obj, pos, patch_shape) * probe for pos in self.positions_px],
            dim=0,
        )

    def diffraction_amplitudes(self, obj: torch.Tensor, probe: torch.Tensor) -> torch.Tensor:
        exit_waves = self.exit_waves(obj, probe)
        far_field = torch.fft.fft2(exit_waves, norm="ortho")
        return far_field.abs()

    def diffraction_intensities(self, obj: torch.Tensor, probe: torch.Tensor) -> torch.Tensor:
        amplitudes = self.diffraction_amplitudes(obj, probe)
        return amplitudes.square()


class MultislicePtychographyForwardModel(nn.Module):
    """Periodic multislice ptychography forward model over complex slice transmissions."""

    def __init__(
        self,
        positions_px: torch.Tensor,
        *,
        energy_eV: float,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        slice_thickness_A: float,
    ) -> None:
        super().__init__()
        self.register_buffer("positions_px", positions_px.to(torch.int64))
        self.propagator = FresnelPropagator(
            energy_eV=energy_eV,
            gpts=gpts,
            sampling=sampling,
            thickness_A=slice_thickness_A,
        )
        self.inverse_propagator = FresnelPropagator(
            energy_eV=energy_eV,
            gpts=gpts,
            sampling=sampling,
            thickness_A=-slice_thickness_A,
        )

    def overlap_projection(
        self, objects: torch.Tensor, probe: torch.Tensor, position_px: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        patch_shape = tuple(probe.shape[-2:])
        num_slices = int(objects.shape[0])
        probe_states = []
        exit_waves = []
        current_probe = probe.to(torch.complex128)
        for s in range(num_slices):
            object_patch = extract_patch(objects[s], position_px, patch_shape)
            probe_states.append(current_probe)
            exit_wave = object_patch * current_probe
            exit_waves.append(exit_wave)
            if s + 1 < num_slices:
                current_probe = self.propagator(exit_wave)
        return torch.stack(probe_states, dim=0), torch.stack(exit_waves, dim=0)

    def diffraction_amplitudes(self, objects: torch.Tensor, probe: torch.Tensor) -> torch.Tensor:
        amplitudes = []
        for pos in self.positions_px:
            _, exit_waves = self.overlap_projection(objects, probe, pos)
            far_field = torch.fft.fft2(exit_waves[-1], norm="ortho")
            amplitudes.append(far_field.abs())
        return torch.stack(amplitudes, dim=0)

    def diffraction_intensities(self, objects: torch.Tensor, probe: torch.Tensor) -> torch.Tensor:
        amplitudes = self.diffraction_amplitudes(objects, probe)
        return amplitudes.square()


def amplitude_projection(exit_wave: torch.Tensor, measured_amplitude: torch.Tensor) -> torch.Tensor:
    far_field = torch.fft.fft2(exit_wave, norm="ortho")
    phase = far_field / (far_field.abs() + 1e-30)
    modified_far_field = measured_amplitude.to(far_field.dtype) * phase
    return torch.fft.ifft2(modified_far_field, norm="ortho")


@dataclass
class RPIEParameters:
    alpha: float = 0.5
    beta: float = 0.5
    object_step_size: float = 1.0
    probe_step_size: float = 1.0


class RPIEReconstructor(nn.Module):
    """Simplified r-PIE reconstruction loop for a single probe/object state."""

    def __init__(self, positions_px: torch.Tensor, parameters: RPIEParameters | None = None) -> None:
        super().__init__()
        self.forward_model = PtychographyForwardModel(positions_px)
        self.parameters_cfg = parameters or RPIEParameters()

    def step(
        self,
        obj: torch.Tensor,
        probe: torch.Tensor,
        measured_amplitudes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        losses = []
        patch_shape = tuple(probe.shape[-2:])
        for i, pos in enumerate(self.forward_model.positions_px):
            object_patch = extract_patch(obj, pos, patch_shape)
            exit_wave = object_patch * probe
            modified_exit_wave = amplitude_projection(exit_wave, measured_amplitudes[i])
            exit_wave_diff = modified_exit_wave - exit_wave

            probe_conj = torch.conj(probe)
            probe_abs_squared = probe.abs().square()
            object_conj = torch.conj(object_patch)
            object_abs_squared = object_patch.abs().square()

            alpha = self.parameters_cfg.alpha
            beta = self.parameters_cfg.beta
            object_update = (
                self.parameters_cfg.object_step_size
                * probe_conj
                * exit_wave_diff
                / ((1 - alpha) * probe_abs_squared + alpha * probe_abs_squared.max() + 1e-30)
            )
            probe_update = (
                self.parameters_cfg.probe_step_size
                * object_conj
                * exit_wave_diff
                / ((1 - beta) * object_abs_squared + beta * object_abs_squared.max() + 1e-30)
            )

            obj = assign_patch_add(obj, pos, object_update)
            probe = probe + probe_update

            far_field = torch.fft.fft2(exit_wave, norm="ortho")
            losses.append(torch.mean((far_field.abs() - measured_amplitudes[i]) ** 2))

        return obj, probe, torch.stack(losses).mean()

    def reconstruct(
        self,
        measured_intensities: torch.Tensor,
        object_init: torch.Tensor,
        probe_init: torch.Tensor,
        iterations: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        obj = object_init.clone()
        probe = probe_init.clone()
        measured_amplitudes = torch.sqrt(torch.clamp(measured_intensities, min=0.0))
        losses: list[torch.Tensor] = []
        for _ in range(iterations):
            obj, probe, loss = self.step(obj, probe, measured_amplitudes)
            losses.append(loss)
        return obj, probe, losses


class MultislicePtychographyReconstructor(nn.Module):
    """Minimal multislice PIE loop over complex slice transmission functions."""

    def __init__(
        self,
        positions_px: torch.Tensor,
        *,
        energy_eV: float,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        slice_thickness_A: float,
        parameters: RPIEParameters | None = None,
        fix_probe: bool = False,
    ) -> None:
        super().__init__()
        self.forward_model = MultislicePtychographyForwardModel(
            positions_px,
            energy_eV=energy_eV,
            gpts=gpts,
            sampling=sampling,
            slice_thickness_A=slice_thickness_A,
        )
        self.parameters_cfg = parameters or RPIEParameters()
        self.fix_probe = fix_probe

    def step(
        self,
        objects: torch.Tensor,
        probe: torch.Tensor,
        measured_amplitudes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        losses = []
        patch_shape = tuple(probe.shape[-2:])
        for i, pos in enumerate(self.forward_model.positions_px):
            probe_states, exit_waves = self.forward_model.overlap_projection(objects, probe, pos)
            modified_exit_waves = torch.zeros_like(exit_waves)
            modified_exit_waves[-1] = amplitude_projection(exit_waves[-1], measured_amplitudes[i])

            for s in reversed(range(objects.shape[0])):
                object_patch = extract_patch(objects[s], pos, patch_shape)
                exit_wave = exit_waves[s]
                modified_exit_wave = modified_exit_waves[s]
                exit_wave_diff = modified_exit_wave - exit_wave

                probe_state = probe_states[s]
                probe_conj = torch.conj(probe_state)
                probe_abs_squared = probe_state.abs().square()
                object_conj = torch.conj(object_patch)
                object_abs_squared = object_patch.abs().square()

                alpha = self.parameters_cfg.alpha
                beta = self.parameters_cfg.beta
                object_update = (
                    self.parameters_cfg.object_step_size
                    * probe_conj
                    * exit_wave_diff
                    / ((1 - alpha) * probe_abs_squared + alpha * probe_abs_squared.max() + 1e-30)
                )
                objects[s] = assign_patch_add(objects[s], pos, object_update)

                if not self.fix_probe or s > 0:
                    probe_update = (
                        self.parameters_cfg.probe_step_size
                        * object_conj
                        * exit_wave_diff
                        / ((1 - beta) * object_abs_squared + beta * object_abs_squared.max() + 1e-30)
                    )
                    probe_states[s] = probe_states[s] + probe_update

                if s > 0:
                    modified_exit_waves[s - 1] = self.forward_model.inverse_propagator(
                        probe_states[s]
                    )

            probe = probe_states[0]
            far_field = torch.fft.fft2(exit_waves[-1], norm="ortho")
            losses.append(torch.mean((far_field.abs() - measured_amplitudes[i]) ** 2))

        return objects, probe, torch.stack(losses).mean()

    def reconstruct(
        self,
        measured_intensities: torch.Tensor,
        object_init: torch.Tensor,
        probe_init: torch.Tensor,
        iterations: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        objects = object_init.clone()
        probe = probe_init.clone()
        measured_amplitudes = torch.sqrt(torch.clamp(measured_intensities, min=0.0))
        losses: list[torch.Tensor] = []
        for _ in range(iterations):
            objects, probe, loss = self.step(objects, probe, measured_amplitudes)
            losses.append(loss)
        return objects, probe, losses
