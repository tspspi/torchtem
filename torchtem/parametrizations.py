from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Callable

import torch

from torchtem.atomic_parametrizations import (
    _load_kirkland_parameters,
    _load_lobato_parameters,
    _load_peng_parameters,
    ewald_gaussian_potential,
    ewald_point_charge_potential,
    kirkland_scaled_projected_parameters,
    lobato_scaled_projected_parameters,
    peng_scaled_parameters,
)
from torchtem.poisson_helpers import load_waasmaier_kirfel_parameters


class Parametrization(ABC):
    def __init__(self, parameters: dict[str, torch.Tensor]) -> None:
        self.parameters = parameters

    @abstractmethod
    def scaled_parameters(self, symbol: str, name: str):
        raise NotImplementedError

    def get_function(self, name: str, symbol: str) -> Callable:
        params = self.scaled_parameters(symbol, name)
        return lambda x, _params=params: self._functions[name](x, _params)


class LobatoParametrization(Parametrization):
    _functions = {
        "scattering_factor": lambda k2, p: (
            p[0, 0] * (2.0 + p[1, 0] * k2) / (1.0 + p[1, 0] * k2) ** 2
            + p[0, 1] * (2.0 + p[1, 1] * k2) / (1.0 + p[1, 1] * k2) ** 2
            + p[0, 2] * (2.0 + p[1, 2] * k2) / (1.0 + p[1, 2] * k2) ** 2
            + p[0, 3] * (2.0 + p[1, 3] * k2) / (1.0 + p[1, 3] * k2) ** 2
            + p[0, 4] * (2.0 + p[1, 4] * k2) / (1.0 + p[1, 4] * k2) ** 2
        ),
        "potential": lambda r, p: (
            p[0, 0] * (2.0 / (p[1, 0] * r) + 1.0) * torch.exp(-p[1, 0] * r)
            + p[0, 1] * (2.0 / (p[1, 1] * r) + 1.0) * torch.exp(-p[1, 1] * r)
            + p[0, 2] * (2.0 / (p[1, 2] * r) + 1.0) * torch.exp(-p[1, 2] * r)
            + p[0, 3] * (2.0 / (p[1, 3] * r) + 1.0) * torch.exp(-p[1, 3] * r)
            + p[0, 4] * (2.0 / (p[1, 4] * r) + 1.0) * torch.exp(-p[1, 4] * r)
        ),
        "potential_derivative": lambda r, p: -(
            p[0, 0] * (2.0 / (p[1, 0] * r.square()) + 2.0 / r + p[1, 0]) * torch.exp(-p[1, 0] * r)
            + p[0, 1] * (2.0 / (p[1, 1] * r.square()) + 2.0 / r + p[1, 1]) * torch.exp(-p[1, 1] * r)
            + p[0, 2] * (2.0 / (p[1, 2] * r.square()) + 2.0 / r + p[1, 2]) * torch.exp(-p[1, 2] * r)
            + p[0, 3] * (2.0 / (p[1, 3] * r.square()) + 2.0 / r + p[1, 3]) * torch.exp(-p[1, 3] * r)
            + p[0, 4] * (2.0 / (p[1, 4] * r.square()) + 2.0 / r + p[1, 4]) * torch.exp(-p[1, 4] * r)
        ),
        "charge": lambda r, p: (
            2.0 * torch.pi**4 * 0.529177210903 * p[0, 0] / p[1, 0] ** 2.5 * torch.exp(-2.0 * torch.pi * r / torch.sqrt(p[1, 0]))
            + 2.0 * torch.pi**4 * 0.529177210903 * p[0, 1] / p[1, 1] ** 2.5 * torch.exp(-2.0 * torch.pi * r / torch.sqrt(p[1, 1]))
            + 2.0 * torch.pi**4 * 0.529177210903 * p[0, 2] / p[1, 2] ** 2.5 * torch.exp(-2.0 * torch.pi * r / torch.sqrt(p[1, 2]))
            + 2.0 * torch.pi**4 * 0.529177210903 * p[0, 3] / p[1, 3] ** 2.5 * torch.exp(-2.0 * torch.pi * r / torch.sqrt(p[1, 3]))
            + 2.0 * torch.pi**4 * 0.529177210903 * p[0, 4] / p[1, 4] ** 2.5 * torch.exp(-2.0 * torch.pi * r / torch.sqrt(p[1, 4]))
        ),
        "x_ray_scattering_factor": lambda k, p: (
            2.0 * torch.pi**2 * 0.529177210903 * p[0, 0] / (p[1, 0] * (1.0 + p[1, 0] * k.square()) ** 2)
            + 2.0 * torch.pi**2 * 0.529177210903 * p[0, 1] / (p[1, 1] * (1.0 + p[1, 1] * k.square()) ** 2)
            + 2.0 * torch.pi**2 * 0.529177210903 * p[0, 2] / (p[1, 2] * (1.0 + p[1, 2] * k.square()) ** 2)
            + 2.0 * torch.pi**2 * 0.529177210903 * p[0, 3] / (p[1, 3] * (1.0 + p[1, 3] * k.square()) ** 2)
            + 2.0 * torch.pi**2 * 0.529177210903 * p[0, 4] / (p[1, 4] * (1.0 + p[1, 4] * k.square()) ** 2)
        ),
        "projected_potential": lambda r, p: 2.0
        * (
            2.0 * p[0][:, None] / p[1][:, None] * torch.special.modified_bessel_k0(r[None] * p[1][:, None])
            + p[0][:, None] * r[None] * torch.special.modified_bessel_k1(r[None] * p[1][:, None])
        ).sum(0),
        "projected_scattering_factor": lambda k2, p: (
            8.0
            * torch.pi
            * (
                p[0, 0] / p[1, 0] / (4 * torch.pi**2 * k2 + p[1, 0] ** 2)
                + p[0, 0] * p[1, 0] / (4 * torch.pi**2 * k2 + p[1, 0] ** 2) ** 2
                + p[0, 1] / p[1, 1] / (4 * torch.pi**2 * k2 + p[1, 1] ** 2)
                + p[0, 1] * p[1, 1] / (4 * torch.pi**2 * k2 + p[1, 1] ** 2) ** 2
                + p[0, 2] / p[1, 2] / (4 * torch.pi**2 * k2 + p[1, 2] ** 2)
                + p[0, 2] * p[1, 2] / (4 * torch.pi**2 * k2 + p[1, 2] ** 2) ** 2
                + p[0, 3] / p[1, 3] / (4 * torch.pi**2 * k2 + p[1, 3] ** 2)
                + p[0, 3] * p[1, 3] / (4 * torch.pi**2 * k2 + p[1, 3] ** 2) ** 2
                + p[0, 4] / p[1, 4] / (4 * torch.pi**2 * k2 + p[1, 4] ** 2)
                + p[0, 4] * p[1, 4] / (4 * torch.pi**2 * k2 + p[1, 4] ** 2) ** 2
            )
        ),
    }

    def __init__(self) -> None:
        super().__init__(_load_lobato_parameters())

    def scaled_parameters(self, symbol: str, name: str):
        parameters = self.parameters[symbol]
        if name in ("scattering_factor", "charge", "x_ray_scattering_factor"):
            return parameters
        if name in ("potential", "potential_derivative", "projected_potential", "projected_scattering_factor"):
            return lobato_scaled_projected_parameters(symbol)
        raise KeyError(name)


class KirklandParametrization(Parametrization):
    _functions = {
        "scattering_factor": lambda k2, p: (
            p[0, 0] / (p[1, 0] + k2)
            + p[2, 0] * torch.exp(-p[3, 0] * k2)
            + p[0, 1] / (p[1, 1] + k2)
            + p[2, 1] * torch.exp(-p[3, 1] * k2)
            + p[0, 2] / (p[1, 2] + k2)
            + p[2, 2] * torch.exp(-p[3, 2] * k2)
        ),
        "potential": lambda r, p: (
            p[0, 0] * torch.exp(-p[1, 0] * r) / r
            + p[2, 0] * torch.exp(-p[3, 0] * r.square())
            + p[0, 1] * torch.exp(-p[1, 1] * r) / r
            + p[2, 1] * torch.exp(-p[3, 1] * r.square())
            + p[0, 2] * torch.exp(-p[1, 2] * r) / r
            + p[2, 2] * torch.exp(-p[3, 2] * r.square())
        ),
        "potential_derivative": lambda r, p: (
            -p[0, 0] * (1.0 / r + p[1, 0]) * torch.exp(-p[1, 0] * r) / r
            - 2.0 * p[2, 0] * p[3, 0] * r * torch.exp(-p[3, 0] * r.square())
            - p[0, 1] * (1.0 / r + p[1, 1]) * torch.exp(-p[1, 1] * r) / r
            - 2.0 * p[2, 1] * p[3, 1] * r * torch.exp(-p[3, 1] * r.square())
            - p[0, 2] * (1.0 / r + p[1, 2]) * torch.exp(-p[1, 2] * r) / r
            - 2.0 * p[2, 2] * p[3, 2] * r * torch.exp(-p[3, 2] * r.square())
        ),
        "projected_potential": lambda r, p: (
            2.0 * p[0][:, None] * torch.special.modified_bessel_k0(p[1][:, None] * r[None])
            + torch.sqrt(torch.pi / p[3][:, None]) * p[2][:, None] * torch.exp(-p[3][:, None] * r[None] ** 2)
        ).sum(0),
        "projected_scattering_factor": lambda k2, p: (
            4 * torch.pi * p[0, 0] / (4 * torch.pi**2 * k2 + p[1, 0] ** 2)
            + torch.sqrt(torch.pi / p[3, 0]) * p[2, 0] * torch.pi / p[3, 0] * torch.exp(-(torch.pi**2) * k2 / p[3, 0])
            + 4 * torch.pi * p[0, 1] / (4 * torch.pi**2 * k2 + p[1, 1] ** 2)
            + torch.sqrt(torch.pi / p[3, 1]) * p[2, 1] * torch.pi / p[3, 1] * torch.exp(-(torch.pi**2) * k2 / p[3, 1])
            + 4 * torch.pi * p[0, 2] / (4 * torch.pi**2 * k2 + p[1, 2] ** 2)
            + torch.sqrt(torch.pi / p[3, 2]) * p[2, 2] * torch.pi / p[3, 2] * torch.exp(-(torch.pi**2) * k2 / p[3, 2])
        ),
    }

    def __init__(self) -> None:
        super().__init__(_load_kirkland_parameters())

    def scaled_parameters(self, symbol: str, name: str):
        parameters = self.parameters[symbol]
        if name == "scattering_factor":
            return parameters
        if name in ("potential", "potential_derivative", "projected_potential", "projected_scattering_factor"):
            return kirkland_scaled_projected_parameters(symbol)
        raise KeyError(name)


class PengParametrization(Parametrization):
    _functions = {
        "scattering_factor": lambda k2, p: (
            p[0, 0] * torch.exp(-p[1, 0] * k2)
            + p[0, 1] * torch.exp(-p[1, 1] * k2)
            + p[0, 2] * torch.exp(-p[1, 2] * k2)
            + p[0, 3] * torch.exp(-p[1, 3] * k2)
            + p[0, 4] * torch.exp(-p[1, 4] * k2)
        ),
        "projected_scattering_factor": lambda k2, p: (
            p[0, 0] * torch.exp(-p[1, 0] * k2)
            + p[0, 1] * torch.exp(-p[1, 1] * k2)
            + p[0, 2] * torch.exp(-p[1, 2] * k2)
            + p[0, 3] * torch.exp(-p[1, 3] * k2)
            + p[0, 4] * torch.exp(-p[1, 4] * k2)
        ),
    }

    def __init__(self, table: str = "peng_high.json") -> None:
        self.table = table
        super().__init__(_load_peng_parameters(table))

    def get_function(self, name: str, symbol: str) -> Callable:
        if name == "finite_projected_scattering_factor":
            params = peng_scaled_parameters(symbol, self.table)["finite_projected_potential"]

            def finite_projected_scattering_factor(
                r: torch.Tensor,
                a,
                b,
                _params: torch.Tensor = params,
            ) -> torch.Tensor:
                lower = torch.as_tensor(a, dtype=r.dtype, device=r.device)
                upper = torch.as_tensor(b, dtype=r.dtype, device=r.device)
                amplitudes = _params[0].to(dtype=r.dtype, device=r.device)[:, None]
                exponents = _params[1].to(dtype=r.dtype, device=r.device)[:, None]
                erf_scale = _params[2].to(dtype=r.dtype, device=r.device)[:, None]
                return (
                    torch.abs(torch.erf(erf_scale * upper) - torch.erf(erf_scale * lower))
                    * amplitudes
                    * torch.exp(-exponents * r[None].square())
                ).sum(0) / 2.0

            return finite_projected_scattering_factor
        return super().get_function(name, symbol)

    def scaled_parameters(self, symbol: str, name: str):
        if name == "scattering_factor":
            params = self.parameters[symbol].clone()
            params[1] = params[1] / 4.0
            return params
        if name in (
            "projected_potential",
            "projected_scattering_factor",
            "finite_projected_potential",
            "finite_projected_scattering_factor",
        ):
            return peng_scaled_parameters(symbol, self.table)[name]
        raise KeyError(name)


class EwaldParametrization(Parametrization):
    _functions = {
        "gaussian_potential": lambda r, p: ewald_gaussian_potential(r, p[1], p[0]),
        "gaussian_charge": lambda r, p: p[1]
        / (p[0] ** 3 * torch.sqrt(torch.tensor(2.0 * torch.pi, dtype=r.dtype, device=r.device)) ** 3)
        * torch.exp(-(r.square()) / (2.0 * p[0] ** 2)),
        "point_charge_potential": lambda r, p: ewald_point_charge_potential(r, p[1]),
        "potential": lambda r, p: ewald_point_charge_potential(r, p[1]) - ewald_gaussian_potential(r, p[1], p[0]),
    }

    def __init__(self, width: float = 1.0) -> None:
        parameters = {
            symbol: torch.tensor([width, float(Z)], dtype=torch.float64)
            for Z, symbol in enumerate(electron_symbols(), start=1)
        }
        super().__init__(parameters)

    def scaled_parameters(self, symbol: str, name: str):
        if name not in ("gaussian_potential", "gaussian_charge", "point_charge_potential", "potential"):
            raise KeyError(name)
        return self.parameters[symbol]


class WaasmaierKirfelParametrization(Parametrization):
    _functions = {
        "x_ray_scattering_factor": lambda k, p: (
            p[0] * torch.exp(-p[6] * k.square())
            + p[1] * torch.exp(-p[7] * k.square())
            + p[2] * torch.exp(-p[8] * k.square())
            + p[3] * torch.exp(-p[9] * k.square())
            + p[4] * torch.exp(-p[10] * k.square())
            + p[5]
        ),
    }

    def __init__(self) -> None:
        super().__init__(load_waasmaier_kirfel_parameters())

    def scaled_parameters(self, symbol: str, name: str):
        if name != "x_ray_scattering_factor":
            raise KeyError(name)
        return self.parameters[symbol]


@lru_cache(maxsize=1)
def electron_symbols() -> tuple[str, ...]:
    from torchtem.core_loss import electron_configurations

    return tuple(electron_configurations().keys())


def validate_parametrization(parametrization: str | Parametrization) -> Parametrization:
    if isinstance(parametrization, Parametrization):
        return parametrization
    named = {
        "lobato": LobatoParametrization,
        "kirkland": KirklandParametrization,
        "peng": PengParametrization,
        "ewald": EwaldParametrization,
        "waasmaier_kirfel": WaasmaierKirfelParametrization,
    }
    return named[parametrization]()
