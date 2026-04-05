# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

from vllm.v1.dssd.edge.session import DSSDEdgeSessionState
from vllm.v1.dssd.engine.batch_planner import VerifierRoundBatcher
from vllm.v1.dssd.engine.session_store import DSSDSessionStore
from vllm.v1.dssd.protocol import (
    DraftRoundRequest,
    VerifierForwardResult,
    VerifyRoundRequest,
)
from vllm.v1.dssd.worker.draft_runner import DraftRoundResult
from vllm.v1.dssd.worker.verifier_runner import build_verifier_result


class DSSDSessionRunner:

    def __init__(self, model_executor: Any | None = None) -> None:
        self.edge_sessions = DSSDSessionStore()
        self.verifier_sessions = DSSDSessionStore()
        self.verifier_batcher = VerifierRoundBatcher()
        self.model_executor = model_executor

    def create_edge_session(self, session_state: DSSDEdgeSessionState) -> None:
        self.edge_sessions.put(session_state.local_session_id, session_state)

    def dssd_draft_round(self, request: DraftRoundRequest) -> DraftRoundResult:
        if not isinstance(request, DraftRoundRequest):
            raise TypeError("dssd_draft_round expects DraftRoundRequest")
        if self.model_executor is not None:
            result = self.model_executor.collective_rpc(
                "dssd_draft_round",
                args=(request,),
            )
            return result[0]
        return DraftRoundResult(
            draft_token_ids=[1] * request.gamma,
            q_values=[0.75] * request.gamma,
            q_dists_handle=f"{request.local_session_id}:{request.seq_no}",
            q_distributions=[[0.25, 0.75]] * request.gamma,
        )

    def dssd_verify_round(
        self, request: VerifyRoundRequest
    ) -> VerifierForwardResult:
        if self.model_executor is not None:
            result = self.model_executor.collective_rpc(
                "dssd_verify_round",
                args=(request,),
            )
            return result[0]
        vocab_size = max(request.draft_token_ids, default=0) + 1
        vocab_size = max(vocab_size, 1)
        target_probs = []
        for token_id in request.draft_token_ids:
            probs = [0.0] * vocab_size
            probs[token_id] = 1.0
            target_probs.append(probs)
        bonus_probs = [0.0] * vocab_size
        bonus_probs[0] = 1.0
        target_probs.append(bonus_probs)
        return build_verifier_result(
            request=request,
            target_probs=target_probs,
            finish_reason="placeholder-forward",
        )
