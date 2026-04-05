# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DSSDVerifierModelResult:
    accepted_count: int
    seq_probs: list[list[float]]
    bonus_probs: list[float]


def build_verifier_result(
    draft_token_ids: list[int],
    q_values: list[float],
    target_probs: list[list[float]],
) -> DSSDVerifierModelResult:
    del draft_token_ids, q_values
    return DSSDVerifierModelResult(
        accepted_count=0,
        seq_probs=target_probs[:-1],
        bonus_probs=target_probs[-1],
    )
