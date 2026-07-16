from __future__ import annotations

from typing import Any, Optional

from tqdm.auto import tqdm

import torchtem.config as config_module


class TqdmWrapper:
    def __init__(self, *args, enabled: Optional[bool] = None, **kwargs: Any):
        if enabled is None:
            enabled = config_module.get("diagnostics.task_progress", False)
        self._pbar = None
        if tqdm is not None and enabled:
            kwargs.setdefault("delay", 0.5)
            self._pbar = tqdm(*args, **kwargs)

    @property
    def pbar(self):
        return self._pbar

    def update_if_exists(self, n: int = 1) -> None:
        if self.pbar is not None:
            self.pbar.update(n)

    def close_if_exists(self) -> None:
        if self.pbar is not None:
            self.pbar.close()
