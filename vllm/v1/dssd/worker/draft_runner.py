# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DraftRoundResult:
    draft_token_ids: list[int]
    q_values: list[float]
    q_dists_handle: str
