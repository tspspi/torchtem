from __future__ import annotations

import runpy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

from tspi.torchtem.complex_support import abs2
from tspi.torchtem.fft_support import fft2, fft2_convolve
from tspi.torchtem.physics import ensure_tuple2, real_space_mesh


def _electron_configurations_path() -> Path:
    tspi_dir = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[2]
    candidate = tspi_dir / "snapshot" / "src" / "abtem" / "core" / "electron_configurations.py"
    if candidate.exists():
        return candidate
    return repo_root / "abtem" / "core" / "electron_configurations.py"


@lru_cache(maxsize=1)
def _electron_configuration_namespace() -> dict:
    return runpy.run_path(str(_electron_configurations_path()))


@lru_cache(maxsize=1)
def electron_configurations() -> dict[str, str]:
    return dict(_electron_configuration_namespace()["electron_configurations"])


@lru_cache(maxsize=1)
def azimuthal_number() -> dict[str, int]:
    return dict(_electron_configuration_namespace()["azimuthal_number"])


@lru_cache(maxsize=1)
def azimuthal_letter() -> dict[int, str]:
    return dict(_electron_configuration_namespace()["azimuthal_letter"])


def config_str_to_config_tuples(config_str: str) -> list[tuple[int, int, int]]:
    tuples = []
    for subshell_string in config_str.split(" "):
        tuples.append(
            (
                int(subshell_string[0]),
                azimuthal_number()[subshell_string[1]],
                int(subshell_string[2:]),
            )
        )
    return tuples


def config_tuples_to_config_str(config_tuples: list[tuple[int, int, int]]) -> str:
    strings = []
    for n, ell, occ in config_tuples:
        strings.append(str(n) + azimuthal_letter()[ell] + str(occ))
    return " ".join(strings)


def remove_electron_from_config_str(config_str: str, n: int, ell: int) -> str:
    tuples = []
    for shell in config_str_to_config_tuples(config_str):
        if shell[:2] == (n, ell):
            tuples.append(shell[:2] + (shell[2] - 1,))
        else:
            tuples.append(shell)
    return config_tuples_to_config_str(tuples)


def check_valid_quantum_numbers(symbol: str, n: int, ell: int) -> None:
    configuration = config_str_to_config_tuples(electron_configurations()[symbol])
    if not any(shell[:2] == (n, ell) for shell in configuration):
        raise RuntimeError(f"Quantum numbers (n, ell)=({n}, {ell}) not valid for element {symbol}")


@dataclass(frozen=True)
class AtomicWaveQuantumNumbers:
    n: int | None
    l: int
    ml: int


@dataclass
class SubshellTransitions:
    symbol: str
    n: int
    l: int
    order: int = 1
    epsilon_eV: float = 1.0

    def __post_init__(self) -> None:
        check_valid_quantum_numbers(self.symbol, self.n, self.l)

    @property
    def bound_configuration(self) -> str:
        return electron_configurations()[self.symbol]

    @property
    def excited_configuration(self) -> str:
        return remove_electron_from_config_str(self.bound_configuration, self.n, self.l)

    @property
    def lprimes(self) -> range:
        min_new_l = max(self.l - self.order, 0)
        return range(min_new_l, self.l + self.order + 1)

    def get_transition_quantum_numbers(
        self,
    ) -> list[tuple[AtomicWaveQuantumNumbers, AtomicWaveQuantumNumbers]]:
        bound_states = [
            AtomicWaveQuantumNumbers(self.n, self.l, ml) for ml in range(-self.l, self.l + 1)
        ]
        excited_states = [
            AtomicWaveQuantumNumbers(None, lprime, mlprime)
            for lprime in self.lprimes
            for mlprime in range(-lprime, lprime + 1)
        ]
        return [(bound, excited) for bound in bound_states for excited in excited_states]


class CoreLossTransitionPotential(nn.Module):
    """Differentiable surrogate for subshell-resolved core-loss transition potentials."""

    def __init__(
        self,
        *,
        gpts: tuple[int, int],
        sampling: tuple[float, float] | float,
        positions_A: torch.Tensor,
        transitions: list[tuple[AtomicWaveQuantumNumbers, AtomicWaveQuantumNumbers]],
        amplitudes: torch.Tensor | None = None,
        widths_A: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float64,
        device=None,
    ) -> None:
        super().__init__()
        self.gpts = gpts
        self.sampling = ensure_tuple2(sampling)
        self.dtype = dtype
        self.device_hint = device
        self.positions_A = nn.Parameter(positions_A.to(device=device, dtype=dtype))
        self.transitions = tuple(transitions)
        num_transitions = len(self.transitions)
        if amplitudes is None:
            amplitudes = torch.ones((num_transitions,), dtype=dtype)
        if widths_A is None:
            widths_A = torch.full((num_transitions,), 0.6, dtype=dtype)
        self.amplitudes = nn.Parameter(amplitudes.to(device=device, dtype=dtype))
        self.log_widths_A = nn.Parameter(widths_A.to(device=device, dtype=dtype).log())

        delta_l = torch.tensor(
            [excited.l - bound.l for bound, excited in self.transitions], dtype=dtype
        )
        delta_ml = torch.tensor(
            [excited.ml - bound.ml for bound, excited in self.transitions], dtype=dtype
        )
        self.register_buffer("delta_l", delta_l.to(device=device, dtype=dtype))
        self.register_buffer("delta_ml", delta_ml.to(device=device, dtype=dtype))

    @property
    def widths_A(self) -> torch.Tensor:
        return self.log_widths_A.exp()

    def kernel(self) -> torch.Tensor:
        yy, xx = real_space_mesh(
            self.gpts, self.sampling, device=self.positions_A.device, dtype=self.dtype
        )
        dy = yy.unsqueeze(0).unsqueeze(0) - self.positions_A[:, 0].view(1, -1, 1, 1)
        dx = xx.unsqueeze(0).unsqueeze(0) - self.positions_A[:, 1].view(1, -1, 1, 1)
        r = torch.sqrt(dx.square() + dy.square() + 1e-8)
        phi = torch.atan2(dy, dx)

        order = self.delta_l.abs().view(-1, 1, 1, 1)
        sigma2 = self.widths_A.square().view(-1, 1, 1, 1)
        radial = r.pow(order) * torch.exp(-r.square() / (2.0 * sigma2))
        angular = torch.exp(1.0j * self.delta_ml.view(-1, 1, 1, 1).to(torch.complex128) * phi.to(torch.complex128))
        amplitude = self.amplitudes.view(-1, 1, 1, 1).to(torch.complex128)
        return torch.sum(amplitude * radial.to(torch.complex128) * angular, dim=1)

    def integrated_intensities(self) -> torch.Tensor:
        return self.local_potential(space="real").sum(dim=(-2, -1)) * (
            self.sampling[0] * self.sampling[1]
        )

    def local_potential(self, space: str = "real") -> torch.Tensor:
        kernel = self.kernel()
        if space == "real":
            return abs2(kernel)
        if space == "reciprocal":
            return abs2(fft2(kernel))
        raise ValueError("space must be 'real' or 'reciprocal'")

    def filter_by_intensity(self, threshold: float) -> "CoreLossTransitionPotential":
        intensities = self.integrated_intensities()
        order = torch.argsort(intensities, descending=True)
        ordered_intensities = intensities[order]
        cumulative = torch.cumsum(ordered_intensities / ordered_intensities.sum(), dim=0)
        n = int(torch.searchsorted(cumulative, torch.as_tensor(threshold, dtype=cumulative.dtype, device=cumulative.device)).item()) + 1
        included = order[:n]

        if included.numel() == 0:
            raise RuntimeError("No transitions remain after filtering")

        transitions = [self.transitions[int(i)] for i in included.tolist()]
        return self.__class__(
            gpts=self.gpts,
            sampling=self.sampling,
            positions_A=self.positions_A.detach().clone(),
            transitions=transitions,
            amplitudes=self.amplitudes.detach()[included].clone(),
            widths_A=self.widths_A.detach()[included].clone(),
            dtype=self.dtype,
            device=self.positions_A.device,
        )

    def absolute_threshold(self, wave: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
        if threshold >= 1.0:
            return torch.as_tensor(0.0, dtype=self.dtype, device=self.positions_A.device)

        local_potential = self.local_potential(space="real").sum(dim=0)
        array = abs2(wave.to(torch.complex128))
        overlap = fft2_convolve(
            local_potential.to(torch.complex128),
            fft2(array.to(torch.complex128)),
        ).real
        overlap = torch.sort(overlap.reshape(-1), descending=True).values
        cumulative = torch.cumsum(overlap, dim=0) / overlap.sum()
        index = torch.searchsorted(
            cumulative,
            torch.as_tensor(threshold, dtype=cumulative.dtype, device=cumulative.device),
            right=False,
        ) - 1
        index = torch.clamp(index, min=0)
        return overlap[index]

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        return self.kernel() * wave.to(torch.complex128)
