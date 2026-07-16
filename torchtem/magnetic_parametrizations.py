from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


def _lyon_data_path() -> Path:
    candidates = (
        Path(__file__).resolve().parents[1]
        / "snapshot"
        / "src"
        / "abtem"
        / "magnetism"
        / "parametrizations"
        / "data"
        / "lyon.json",
        Path(__file__).resolve().parents[2]
        / "abtem"
        / "magnetism"
        / "parametrizations"
        / "data"
        / "lyon.json",
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not locate lyon.json in snapshot or source tree.")


@lru_cache(maxsize=1)
def get_magnetic_parameters() -> dict[str, list[float]]:
    with open(_lyon_data_path(), "r") as f:
        return json.load(f)


class LyonMagneticParametrization:
    def __init__(self) -> None:
        self._parameters = get_magnetic_parameters()

    @property
    def parameters(self) -> dict[str, list[float]]:
        return self._parameters


def validate_magnetic_parametrization(
    parametrization: str | LyonMagneticParametrization,
) -> LyonMagneticParametrization:
    if isinstance(parametrization, LyonMagneticParametrization):
        return parametrization
    named = {
        "lyon": LyonMagneticParametrization,
    }
    try:
        return named[parametrization]()
    except KeyError as exc:
        raise KeyError(f"Unknown magnetic parametrization: {parametrization}") from exc
