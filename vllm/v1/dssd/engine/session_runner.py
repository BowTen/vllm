# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

from vllm.v1.dssd.edge.session import DSSDEdgeSessionState
from vllm.v1.dssd.engine.batch_planner import VerifierRoundBatcher
from vllm.v1.dssd.engine.session_store import DSSDSessionStore
from vllm.v1.dssd.protocol import VerifyRoundRequest
from vllm.v1.dssd.worker.draft_runner import DraftRoundResult


class DSSDSessionRunner:

    def __init__(self) -> None:
        self.edge_sessions = DSSDSessionStore()
        self.verifier_sessions = DSSDSessionStore()
        self.verifier_batcher = VerifierRoundBatcher()

    def create_edge_session(self, session_state: DSSDEdgeSessionState) -> None:
        self.edge_sessions.put(session_state.local_session_id, session_state)

    def dssd_draft_round(self, request: Any) -> DraftRoundResult:
        del request
        return DraftRoundResult(
            draft_token_ids=[],
            q_values=[],
            q_dists_handle="",
        )

    def dssd_verify_round(self, request: VerifyRoundRequest) -> dict[str, object]:
        return {"method": "dssd_verify_round", "request": request}
