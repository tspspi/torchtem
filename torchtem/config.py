from __future__ import annotations

import os
import threading
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Union

import yaml

no_default = "__no_default__"

if "ABTEM_CONFIG" in os.environ:
    PATH = os.environ["ABTEM_CONFIG"]
else:
    PATH = os.path.join(os.path.expanduser("~"), ".config", "abtem")

config: dict[str, Any] = {}
config_lock = threading.Lock()
defaults: list[Mapping[str, Any]] = []
deprecations: dict[str, str | None] = {}


def _canonical_name(key: str, mapping: Mapping[str, Any]) -> str:
    if key in mapping:
        return key
    lowered = {candidate.lower(): candidate for candidate in mapping}
    return lowered.get(key.lower(), key)


def _deep_update(target: dict[str, Any], source: Mapping[str, Any], *, priority: str = "new") -> None:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
            _deep_update(target[key], value, priority=priority)
        elif priority == "old" and key in target:
            continue
        else:
            target[key] = dict(value) if isinstance(value, Mapping) else value


class set:
    config: dict[str, Any]
    _record: list[tuple[Literal["insert", "replace"], tuple[str, ...], Any]]

    def __init__(
        self,
        arg: Union[Mapping[str, Any], None] = None,
        config: dict[str, Any] = config,
        lock: threading.Lock = config_lock,
        **kwargs,
    ):
        with lock:
            self.config = config
            self._record = []
            if arg is not None:
                for key, value in arg.items():
                    key = check_deprecations(key)
                    self._assign(key.split("."), value, config)
            for key, value in kwargs.items():
                key = check_deprecations(key.replace("__", "."))
                self._assign(key.split("."), value, config)

    def __enter__(self):
        return self.config

    def __exit__(self, exc_type, exc, traceback):
        for op, path, value in reversed(self._record):
            d = self.config
            if op == "replace":
                for key in path[:-1]:
                    d = d.setdefault(key, {})
                d[path[-1]] = value
            else:
                for key in path[:-1]:
                    d = d.get(key, {})
                d.pop(path[-1], None)

    def _assign(
        self,
        keys: Sequence[str],
        value: Any,
        d: dict[str, Any],
        path: tuple[str, ...] = (),
        record: bool = True,
    ) -> None:
        key = _canonical_name(keys[0], d)
        path = path + (key,)
        if len(keys) == 1:
            if record:
                if key in d:
                    self._record.append(("replace", path, d[key]))
                else:
                    self._record.append(("insert", path, None))
            d[key] = value
            return
        if key not in d:
            if record:
                self._record.append(("insert", path, None))
            d[key] = {}
            record = False
        self._assign(keys[1:], value, d[key], path, record=record)


def refresh(config: dict[str, Any] = config, defaults: list[Mapping[str, Any]] = defaults) -> None:
    config.clear()
    for default_mapping in defaults:
        _deep_update(config, default_mapping, priority="old")


def get(
    key: str,
    default: Any = no_default,
    config: dict[str, Any] = config,
    override_with: Any = None,
) -> Any:
    if override_with is not None:
        return override_with
    result: Any = config
    for part in key.split("."):
        part = _canonical_name(part, result)
        try:
            result = result[part]
        except (TypeError, KeyError, IndexError):
            if default is not no_default:
                return default
            raise
    return result


def update_defaults(
    new: Mapping[str, Any], config: dict[str, Any] = config, defaults: list[Mapping[str, Any]] = defaults
) -> None:
    defaults.append(new)
    _deep_update(config, new, priority="old")


def check_deprecations(key: str, deprecations: dict[str, str | None] = deprecations) -> str:
    if key not in deprecations:
        return key
    new = deprecations[key]
    if new:
        warnings.warn(
            f'Configuration key "{key}" has been deprecated. Please use "{new}" instead'
        )
        return new
    raise ValueError(f'Configuration value "{key}" has been removed')


def _initialize() -> None:
    path = Path(__file__).resolve().parents[2] / "abtem" / "core" / "abtem.yaml"
    with open(path) as f:
        loaded_defaults = yaml.safe_load(f)
    update_defaults(loaded_defaults)


_initialize()
