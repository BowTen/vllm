# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vllm.v1.dssd.edge.session import DSSDEdgeSessionState
from vllm.v1.dssd.engine.batch_planner import VerifierRoundBatcher
from vllm.v1.dssd.engine.session_store import DSSDSessionStore
from vllm.v1.dssd.protocol import (
    CloseSessionRequest,
    DSSDVerifierExecutionRequest,
    DraftRoundRequest,
    VerifierCommitRequest,
    VerifierSessionInitRequest,
    VerifierForwardResult,
    VerifyRoundRequest,
)
from vllm.v1.dssd.worker.draft_runner import DraftRoundResult
from vllm.v1.dssd.worker.verifier_runner import build_verifier_result


@dataclass
class DSSDEngineVerifierSessionState:
    verifier_session_id: str
    binding_id: str
    sampling_params_digest: str
    prompt_token_ids: list[int]
    committed_token_ids: list[int] = field(default_factory=list)


class DSSDSessionRunner:

    def __init__(self, model_executor: Any | None = None) -> None:
        self.edge_sessions = DSSDSessionStore()
        self.verifier_sessions = DSSDSessionStore()
        self.verifier_batcher = VerifierRoundBatcher()
        self.model_executor = model_executor

    def create_edge_session(self, session_state: DSSDEdgeSessionState) -> None:
        self.edge_sessions.put(session_state.local_session_id, session_state)

    def create_verifier_session(self, request: VerifierSessionInitRequest) -> bool:
        if not isinstance(request, VerifierSessionInitRequest):
            raise TypeError(
                "create_verifier_session expects VerifierSessionInitRequest"
            )
        state = DSSDEngineVerifierSessionState(
            verifier_session_id=request.verifier_session_id,
            binding_id=request.binding_id,
            sampling_params_digest=request.sampling_params_digest,
            prompt_token_ids=list(request.prompt_token_ids),
            committed_token_ids=list(request.prompt_token_ids),
        )
        self.verifier_sessions.put(request.verifier_session_id, state)
        return True

    def close_verifier_session(self, request: CloseSessionRequest) -> bool:
        if not isinstance(request, CloseSessionRequest):
            raise TypeError("close_verifier_session expects CloseSessionRequest")
        state = self.verifier_sessions.get(request.verifier_session_id)
        if state is None:
            return False
        self.verifier_sessions.delete(request.verifier_session_id)
        return True

    def commit_verifier_tokens(self, request: VerifierCommitRequest) -> bool:
        if not isinstance(request, VerifierCommitRequest):
            raise TypeError("commit_verifier_tokens expects VerifierCommitRequest")
        state = self.verifier_sessions.get(request.verifier_session_id)
        if state is None:
            return False
        if request.token_ids:
            state.committed_token_ids.extend(request.token_ids)
        return True

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
        execution_request = self._build_verifier_execution_request(request)
        if self.model_executor is not None:
            result = self.model_executor.collective_rpc(
                "dssd_verify_round",
                args=(execution_request,),
            )
            response = result[0]
        else:
            vocab_size = max(execution_request.draft_token_ids, default=0) + 1
            vocab_size = max(vocab_size, 1)
            target_probs = []
            for token_id in execution_request.draft_token_ids:
                probs = [0.0] * vocab_size
                probs[token_id] = 1.0
                target_probs.append(probs)
            bonus_probs = [0.0] * vocab_size
            bonus_probs[0] = 1.0
            target_probs.append(bonus_probs)
            response = build_verifier_result(
                request=execution_request,
                target_probs=target_probs,
                finish_reason="placeholder-forward",
            )

        session_state = self.verifier_sessions.get(request.verifier_session_id)
        if session_state is not None and request.prefix_delta_token_ids:
            session_state.committed_token_ids.extend(request.prefix_delta_token_ids)
        return response

    def _build_verifier_execution_request(
        self, request: VerifyRoundRequest
    ) -> DSSDVerifierExecutionRequest:
        session_state = self.verifier_sessions.get(request.verifier_session_id)
        committed_token_ids = []
        if session_state is not None:
            committed_token_ids.extend(session_state.committed_token_ids)
        committed_token_ids.extend(request.prefix_delta_token_ids)
        return DSSDVerifierExecutionRequest(
            binding_id=request.binding_id,
            verifier_session_id=request.verifier_session_id,
            seq_no=request.seq_no,
            committed_token_ids=committed_token_ids,
            draft_token_ids=list(request.draft_token_ids),
            q_values=list(request.q_values),
            prefix_delta_token_ids=list(request.prefix_delta_token_ids),
        )
