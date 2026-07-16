from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch

from torchtem.atomic_parametrizations import _load_lobato_parameters
from torchtem.constants import kappa
from torchtem.core_loss import config_str_to_config_tuples, electron_configurations
from torchtem.parametrizations import LobatoParametrization


def _lobato_scattering_factor(k2: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    p = parameters
    return (
        p[0, 0] * (2.0 + p[1, 0] * k2) / (1.0 + p[1, 0] * k2) ** 2
        + p[0, 1] * (2.0 + p[1, 1] * k2) / (1.0 + p[1, 1] * k2) ** 2
        + p[0, 2] * (2.0 + p[1, 2] * k2) / (1.0 + p[1, 2] * k2) ** 2
        + p[0, 3] * (2.0 + p[1, 3] * k2) / (1.0 + p[1, 3] * k2) ** 2
        + p[0, 4] * (2.0 + p[1, 4] * k2) / (1.0 + p[1, 4] * k2) ** 2
    )


@dataclass
class LobatoFitResult:
    parameters: torch.Tensor
    loss_history: list[float]
    iterations: int


class LobatoScatteringFactorFitter(torch.nn.Module):
    """Fit the Lobato scattering-factor form to target k-space data."""

    def __init__(self, initial_parameters: torch.Tensor) -> None:
        super().__init__()
        initial_parameters = initial_parameters.to(torch.float64)
        self.a = torch.nn.Parameter(initial_parameters[0].clone())
        b0 = torch.clamp(initial_parameters[1].abs(), min=1e-12)
        self.raw_b = torch.nn.Parameter(_inverse_softplus(b0))

    @property
    def parameters_constrained(self) -> torch.Tensor:
        b = torch.nn.functional.softplus(self.raw_b) + 1e-12
        return torch.stack((self.a, b), dim=0)

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        k = k.to(torch.float64)
        return _lobato_scattering_factor(k.square(), self.parameters_constrained)

    def fit(
        self,
        k: torch.Tensor,
        target: torch.Tensor,
        *,
        regularization: float = 0.0,
        reference_parameters: torch.Tensor | None = None,
        lr: float = 1.0,
        steps: int = 200,
    ) -> LobatoFitResult:
        k = k.to(torch.float64)
        target = target.to(torch.float64)
        if reference_parameters is None:
            reference_parameters = self.parameters_constrained.detach()
        reference_parameters = reference_parameters.to(torch.float64)

        optimizer = torch.optim.LBFGS(
            self.parameters(),
            lr=lr,
            max_iter=int(steps),
            line_search_fn="strong_wolfe",
        )
        loss_history: list[float] = []

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            prediction = self(k)
            data_loss = torch.mean((target - prediction) ** 2)
            if regularization:
                reg = torch.mean((self.parameters_constrained - reference_parameters) ** 2)
                loss = data_loss + float(regularization) * reg
            else:
                loss = data_loss
            loss.backward()
            loss_history.append(float(loss.detach().item()))
            return loss

        optimizer.step(closure)
        return LobatoFitResult(
            parameters=self.parameters_constrained.detach(),
            loss_history=loss_history,
            iterations=len(loss_history),
        )


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        x > 20.0,
        x,
        torch.log(torch.expm1(x)),
    )


def _normalize_symbol(symbol: str | int) -> str:
    symbols = tuple(electron_configurations().keys())
    if isinstance(symbol, int):
        return symbols[int(symbol) - 1]
    return symbol


def _atomic_number(symbol: str | int) -> int:
    normalized = _normalize_symbol(symbol)
    symbols = tuple(electron_configurations().keys())
    return symbols.index(normalized) + 1


def _storage_key(symbol: str | int, charge: float = 0.0) -> tuple[str, int]:
    symbol = _normalize_symbol(symbol)
    rounded_charge = int(torch.sign(torch.tensor(charge)).item() * torch.ceil(torch.tensor(abs(charge))).item())
    return symbol, rounded_charge


def _lobato_scaled_projected_parameters_from_parameters(parameters: torch.Tensor) -> torch.Tensor:
    a = torch.pi**2 * parameters[0] / torch.pow(parameters[1], 1.5) / kappa
    b = 2.0 * torch.pi / torch.sqrt(parameters[1])
    return torch.stack((a, b), dim=0)


def _lobato_charge_density(r: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    bohr_A = 0.529177210903
    p = parameters.to(torch.float64)
    r = r.to(torch.float64)
    coefficients = 2.0 * torch.pi**4 * bohr_A * p[0] / torch.pow(p[1], 2.5)
    decay = torch.exp(-2.0 * torch.pi * r[None] / torch.sqrt(p[1])[:, None])
    return (coefficients[:, None] * decay).sum(dim=0)


def _lobato_potential(r: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    p = parameters.to(torch.float64)
    r = r.to(torch.float64)
    return (
        p[0, 0] * (2.0 / (p[1, 0] * r) + 1.0) * torch.exp(-p[1, 0] * r)
        + p[0, 1] * (2.0 / (p[1, 1] * r) + 1.0) * torch.exp(-p[1, 1] * r)
        + p[0, 2] * (2.0 / (p[1, 2] * r) + 1.0) * torch.exp(-p[1, 2] * r)
        + p[0, 3] * (2.0 / (p[1, 3] * r) + 1.0) * torch.exp(-p[1, 3] * r)
        + p[0, 4] * (2.0 / (p[1, 4] * r) + 1.0) * torch.exp(-p[1, 4] * r)
    )


def _lobato_potential_derivative(r: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    p = parameters.to(torch.float64)
    r = r.to(torch.float64)
    return -(
        p[0, 0] * (2.0 / (p[1, 0] * r.square()) + 2.0 / r + p[1, 0]) * torch.exp(-p[1, 0] * r)
        + p[0, 1] * (2.0 / (p[1, 1] * r.square()) + 2.0 / r + p[1, 1]) * torch.exp(-p[1, 1] * r)
        + p[0, 2] * (2.0 / (p[1, 2] * r.square()) + 2.0 / r + p[1, 2]) * torch.exp(-p[1, 2] * r)
        + p[0, 3] * (2.0 / (p[1, 3] * r.square()) + 2.0 / r + p[1, 3]) * torch.exp(-p[1, 3] * r)
        + p[0, 4] * (2.0 / (p[1, 4] * r.square()) + 2.0 / r + p[1, 4]) * torch.exp(-p[1, 4] * r)
    )


class GPAWParametrization:
    """Dependency-free analogue of the original GPAW-to-Lobato fitting path."""

    def __init__(self) -> None:
        self._parameters = {(k, 0): v.clone() for k, v in _load_lobato_parameters().items()}

    @property
    def parameters(self) -> dict[str, torch.Tensor]:
        neutral_parameters = {}
        charged_parameters = {}
        for (symbol, charge), parameters in self._parameters.items():
            if charge == 0:
                neutral_parameters[symbol] = parameters
            else:
                charged_parameters[f"{symbol}:{charge:+d}"] = parameters
        return neutral_parameters | charged_parameters

    def added_electrons(self, symbol: str | int, charge: float = 0.0) -> list[tuple[int, int, int]]:
        if not charge:
            return []

        symbol = _normalize_symbol(symbol)
        rounded_charge = int(torch.sign(torch.tensor(charge)).item() * torch.ceil(torch.tensor(abs(charge))).item())

        symbols = tuple(electron_configurations().keys())
        number = symbols.index(symbol) + 1
        config = config_str_to_config_tuples(electron_configurations()[symbols[number - 1]])
        ionic_config = config_str_to_config_tuples(
            electron_configurations()[symbols[number - rounded_charge - 1]]
        )

        config = defaultdict(lambda: 0, {shell[:2]: shell[2] for shell in config})
        ionic_config = defaultdict(lambda: 0, {shell[:2]: shell[2] for shell in ionic_config})

        electrons: list[tuple[int, int, int]] = []
        for key in set(config.keys()).union(set(ionic_config.keys())):
            difference = config[key] - ionic_config[key]
            for _ in range(abs(difference)):
                electrons.append(key + (-int(torch.sign(torch.tensor(float(difference))).item()),))
        return electrons

    def fit_lobato(
        self,
        symbol: str,
        k: torch.Tensor,
        target_scattering_factor: torch.Tensor,
        *,
        charge: float = 0.0,
        guess: torch.Tensor | None = None,
        regularization: float = 0.0,
        lr: float = 5e-2,
        steps: int = 800,
    ) -> LobatoFitResult:
        key = _storage_key(symbol, charge)
        if guess is None:
            guess = self._parameters.get(key, self._parameters[_storage_key(symbol, 0)]).clone()
        fitter = LobatoScatteringFactorFitter(guess)
        result = fitter.fit(
            k,
            target_scattering_factor,
            regularization=regularization,
            reference_parameters=guess,
            lr=lr,
            steps=steps,
        )
        self._parameters[key] = result.parameters.clone()
        return result

    def potential(self, symbol: str | int, charge: float = 0.0):
        parameters = self._parameters[_storage_key(symbol, charge)]
        return lambda r, _params=parameters: _lobato_potential(r, _params)

    def projected_potential(self, symbol: str | int, charge: float = 0.0):
        parameters = self._parameters[_storage_key(symbol, charge)]
        scaled = _lobato_scaled_projected_parameters_from_parameters(parameters)
        return lambda r, _params=scaled: LobatoParametrization._functions["projected_potential"](r, _params)

    def projected_scattering_factor(self, symbol: str | int, charge: float = 0.0):
        parameters = self._parameters[_storage_key(symbol, charge)]
        scaled = _lobato_scaled_projected_parameters_from_parameters(parameters)
        return lambda k2, _params=scaled: LobatoParametrization._functions["projected_scattering_factor"](k2, _params)

    def scattering_factor(self, symbol: str | int, charge: float = 0.0):
        parameters = self._parameters[_storage_key(symbol, charge)]
        return lambda k2, _params=parameters: LobatoParametrization._functions["scattering_factor"](k2, _params)

    def charge(self, symbol: str | int, charge: float = 0.0):
        parameters = self._parameters[_storage_key(symbol, charge)]
        return lambda r, _params=parameters: _lobato_charge_density(r, _params)

    def potential_derivative(self, symbol: str | int, charge: float = 0.0):
        parameters = self._parameters[_storage_key(symbol, charge)]
        return lambda r, _params=parameters: _lobato_potential_derivative(r, _params)

    def x_ray_scattering_factor(self, symbol: str | int, charge: float = 0.0):
        atomic_number = float(_atomic_number(symbol))
        electron_scattering_factor = self.scattering_factor(symbol, charge=charge)
        bohr_A = 0.529177210903

        def _fx(k: torch.Tensor) -> torch.Tensor:
            k = k.to(torch.float64)
            fx = atomic_number - 2.0 * torch.pi**2 * bohr_A * k.square() * electron_scattering_factor(k.square())
            return torch.where(k == 0, torch.full_like(k, atomic_number), fx)

        return _fx
