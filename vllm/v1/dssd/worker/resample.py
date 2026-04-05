# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch


def compute_residual_distribution(
    p: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    residual = torch.clamp(p - q, min=0)
    total = residual.sum()
    if total.item() <= 0:
        raise ValueError("residual distribution has no positive mass")
    return residual / total
