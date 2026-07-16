from __future__ import annotations

import math

import torch

PADE_B13 = torch.tensor(
    [
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
        1.0,
    ],
    dtype=torch.float64,
)
THETA_13 = 5.37


def expm(a: torch.Tensor) -> torch.Tensor:
    """Matrix exponential via a fixed [13/13] Pade approximant with scaling/squaring."""
    if a.numel() == 0:
        return torch.zeros_like(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {tuple(a.shape)}")

    n = a.shape[0]
    eye = torch.eye(n, device=a.device, dtype=a.dtype)
    mu = torch.diagonal(a).sum() / n
    A = a - eye * mu

    nrmA = torch.linalg.matrix_norm(A, ord=1).item()
    if nrmA > THETA_13:
        s = int(math.ceil(math.log2(float(nrmA) / THETA_13))) + 1
    else:
        s = 1

    A = A / (2**s)
    A2 = A @ A
    A4 = A2 @ A2
    A6 = A2 @ A4
    b = PADE_B13.to(device=a.device, dtype=a.real.dtype)
    u1, u2, v1, v2 = _expm_inner(eye, A, A2, A4, A6, b)
    u = A @ (A6 @ u1 + u2)
    v = A6 @ v1 + v2

    x = torch.linalg.solve(-u + v, u + v)
    for _ in range(s):
        x = x @ x

    return x * torch.exp(mu)


def _expm_inner(
    eye: torch.Tensor,
    A: torch.Tensor,
    A2: torch.Tensor,
    A4: torch.Tensor,
    A6: torch.Tensor,
    b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    u1 = b[13] * A6 + b[11] * A4 + b[9] * A2
    u2 = b[7] * A6 + b[5] * A4 + b[3] * A2 + b[1] * eye
    v1 = b[12] * A6 + b[10] * A4 + b[8] * A2
    v2 = b[6] * A6 + b[4] * A4 + b[2] * A2 + b[0] * eye
    return u1, u2, v1, v2
