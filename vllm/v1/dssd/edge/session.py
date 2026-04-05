# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DSSDEdgeRoundCache:
    seq_no: int
    gamma: int
    draft_token_ids: list[int]
    q_values: list[float]
    q_dists_handle: str
    q_distributions: list[list[float]] | None = None


@dataclass
class DSSDEdgeSessionState:
    request_id: str
    local_session_id: str
    verifier_binding_id: str
    verifier_session_id: str
    prompt_token_ids: list[int]
    seq_no: int = 0
    committed_token_ids: list[int] = field(default_factory=list)
    pending_prefix_delta_token_ids: list[int] = field(default_factory=list)
