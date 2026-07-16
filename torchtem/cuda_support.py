from __future__ import annotations

import torch


def sum_run_length_encoded(
    array: torch.Tensor, result: torch.Tensor, separators: torch.Tensor
) -> torch.Tensor:
    if array.ndim != 2 or result.ndim != 2:
        raise ValueError("array and result must be rank-2 tensors")
    n_bins = separators.shape[0] - 1
    if result.shape != (array.shape[0], n_bins):
        raise ValueError("result shape must be (n_batch, n_bins)")

    output = result.clone()
    for x in range(n_bins):
        start = int(separators[x].item())
        stop = int(separators[x + 1].item())
        if stop > start:
            output[:, x] = array[:, start:stop].sum(dim=1)
        else:
            output[:, x] = 0.0
    return output


def interpolate_radial(
    array: torch.Tensor,
    positions: torch.Tensor,
    disk_indices: torch.Tensor,
    disk_counts: torch.Tensor,
    sampling: tuple[float, float],
    radial_gpts: torch.Tensor,
    radial_funcs: torch.Tensor,
    radial_deriv: torch.Tensor,
    dt: float,
    r0: float,
    chunk_offset: int = 0,
) -> torch.Tensor:
    rows, cols = array.shape[-2:]
    out = array.clone()
    sampling_0 = float(sampling[0])
    sampling_1 = float(sampling[1])

    for i in range(positions.shape[0]):
        px = positions[i, 0]
        py = positions[i, 1]
        count = int(disk_counts[i].item())
        for j in range(disk_indices.shape[0]):
            if chunk_offset + j >= count:
                continue
            k = int(torch.round(px / sampling_0).item()) + int(disk_indices[j, 0].item())
            m = int(torch.round(py / sampling_1).item()) + int(disk_indices[j, 1].item())
            if k < 0 or k >= rows or m < 0 or m >= cols:
                continue
            dx = k * sampling_0 - px
            dy = m * sampling_1 - py
            r = torch.sqrt(dx * dx + dy * dy)
            idx = int(torch.floor(torch.log(r / r0 + 1e-12) / dt).item())
            if idx < 0:
                val = radial_funcs[i, 0]
            elif idx < radial_gpts.shape[0] - 1:
                slope = radial_deriv[i, idx]
                val = radial_funcs[i, idx] + (r - radial_gpts[idx]) * slope
            else:
                continue
            out[k, m] = out[k, m] + val
    return out
