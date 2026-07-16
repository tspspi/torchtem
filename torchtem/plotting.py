from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch


def _to_cpu_tensor(output: torch.Tensor | float | complex) -> torch.Tensor:
    tensor = torch.as_tensor(output)
    if tensor.is_complex():
        tensor = tensor.abs()
    return tensor.detach().cpu()


def _flatten_named_outputs(
    outputs: torch.Tensor | Mapping[str, torch.Tensor],
    *,
    detector_names: Sequence[str] | None = None,
) -> list[tuple[str, torch.Tensor]]:
    if isinstance(outputs, Mapping):
        panels: list[tuple[str, torch.Tensor]] = []
        for name, value in outputs.items():
            tensor = _to_cpu_tensor(value)
            if tensor.ndim <= 2:
                panels.append((str(name), tensor))
            else:
                panels.extend((f"{name}[{i}]", tensor[i]) for i in range(tensor.shape[0]))
        return panels

    tensor = _to_cpu_tensor(outputs)
    if tensor.ndim <= 2:
        name = detector_names[0] if detector_names else "detector"
        return [(name, tensor)]

    names = detector_names or [f"detector[{i}]" for i in range(tensor.shape[0])]
    return [(str(names[i]), tensor[i]) for i in range(tensor.shape[0])]


def plot_detector_outputs(
    outputs: torch.Tensor | Mapping[str, torch.Tensor],
    *,
    detector_names: Sequence[str] | None = None,
    cmap: str = "magma",
    figsize: tuple[float, float] | None = None,
):
    """Plot detector outputs from a simulation run as matplotlib graphics.

    Scalars are drawn as single-pixel heatmaps, vectors as line plots, and 2D arrays as
    images. Leading detector dimensions can be provided either as a mapping of named
    outputs or as the first axis of a tensor.
    """
    import matplotlib.pyplot as plt

    panels = _flatten_named_outputs(outputs, detector_names=detector_names)
    n_panels = len(panels)
    ncols = min(3, n_panels)
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize or (4.0 * ncols, 3.5 * nrows),
        squeeze=False,
    )

    for ax, (name, value) in zip(axes.flat, panels):
        if value.ndim == 0:
            image = value.reshape(1, 1).to(torch.float64).numpy()
            ax.imshow(image, cmap=cmap, aspect="auto")
            ax.text(0.0, 0.0, f"{float(value):.4g}", ha="center", va="center", color="white")
        elif value.ndim == 1:
            ax.plot(value.numpy())
            ax.set_xlabel("channel")
            ax.set_ylabel("signal")
        else:
            ax.imshow(value.numpy(), cmap=cmap, aspect="auto")
        ax.set_title(name)

    for ax in axes.flat[n_panels:]:
        ax.axis("off")

    fig.tight_layout()
    return fig, axes


def _series_panel_value(value: torch.Tensor, selector_index: int = 0) -> torch.Tensor:
    panel = _to_cpu_tensor(value)
    while panel.ndim > 2:
        panel = panel[selector_index]
    return panel


def plot_series_outputs(
    outputs: torch.Tensor | Mapping[str, torch.Tensor],
    *,
    parameter_name: str,
    parameter_values: Sequence[float] | torch.Tensor,
    detector_names: Sequence[str] | None = None,
    selector_index: int = 0,
    cmap: str = "magma",
    figsize: tuple[float, float] | None = None,
):
    """Plot a parameter sweep as a rows-by-detectors grid.

    The first axis of each output is interpreted as the sweep dimension. If an output
    still has more than two dimensions after removing the series axis, successive
    leading axes are indexed with ``selector_index`` until a 1D or 2D panel remains.
    """
    import matplotlib.pyplot as plt

    raw_parameter_values = _to_cpu_tensor(parameter_values)
    if raw_parameter_values.ndim <= 1:
        parameter_values = raw_parameter_values.reshape(-1)
        row_labels = [f"{float(parameter_values[row]):.4g}" for row in range(int(parameter_values.shape[0]))]
    else:
        parameter_values = raw_parameter_values.reshape(raw_parameter_values.shape[0], -1)
        row_labels = [f"index {row}" for row in range(int(parameter_values.shape[0]))]

    if isinstance(outputs, Mapping):
        names = tuple(str(name) for name in outputs)
        detector_arrays = [outputs[name] for name in names]
    else:
        names = tuple(detector_names) if detector_names is not None else ("detector",)
        detector_arrays = [outputs]

    nrows = int(parameter_values.shape[0])
    ncols = len(detector_arrays)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize or (4.2 * ncols, 3.2 * nrows),
        squeeze=False,
    )

    for col, (name, array) in enumerate(zip(names, detector_arrays)):
        for row in range(nrows):
            ax = axes[row, col]
            panel = _series_panel_value(array[row], selector_index=selector_index)
            if panel.ndim == 0:
                image = panel.reshape(1, 1).to(torch.float64).numpy()
                ax.imshow(image, cmap=cmap, aspect="auto")
                ax.text(0.0, 0.0, f"{float(panel):.4g}", ha="center", va="center", color="white")
            elif panel.ndim == 1:
                ax.plot(panel.numpy())
                ax.set_xlabel("channel")
                ax.set_ylabel("signal")
            else:
                ax.imshow(panel.numpy(), cmap=cmap, aspect="auto")
            if row == 0:
                ax.set_title(name)
            ax.set_ylabel(f"{parameter_name} = {row_labels[row]}")

    fig.tight_layout()
    return fig, axes
