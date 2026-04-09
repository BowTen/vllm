from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import torch

from .types import (
    VerifierRoundRequest,
    VerifierRoundResult,
    VerifierRoundState,
    VerifierSession,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner


class VerifierStateBridge:
    def __init__(self) -> None:
        self._round_states: dict[str, VerifierRoundState] = defaultdict(
            VerifierRoundState
        )

    def finish_prefill_without_commit(self, session: VerifierSession) -> None:
        session.token_ids = list(session.prompt_token_ids)
        session.num_computed_tokens = session.prompt_len
        session.total_len = session.prompt_len
        self.clear_round_state(session)

    def prepare_round(
        self,
        session: VerifierSession,
        request: VerifierRoundRequest,
        model_runner: "GPUModelRunner",
        gamma: int,
    ) -> None:
        self._validate_req_id(session.req_id, request.req_id)
        request.validate(gamma=gamma)

        req_idx = model_runner.req_states.req_id_to_index[session.req_id]
        model_runner.req_states.last_sampled_tokens[
            req_idx, 0
        ] = request.committed_token_id

        draft_tokens = model_runner.req_states.draft_tokens[req_idx]
        draft_tokens.zero_()
        if request.draft_token_ids:
            draft_tokens[: len(request.draft_token_ids)] = torch.tensor(
                request.draft_token_ids,
                dtype=draft_tokens.dtype,
                device=draft_tokens.device,
            )

        round_state = self._round_state(session)
        round_state.committed_token_id = request.committed_token_id
        round_state.committed_token_committed = False
        round_state.draft_token_ids = list(request.draft_token_ids)
        round_state.draft_q_values = list(request.draft_q_values)
        round_state.last_result = None

    def set_round_q_values(
        self, session: VerifierSession, q_values: list[float]
    ) -> None:
        self._round_state(session).draft_q_values = list(q_values)

    def commit_committed_token_before_postprocess(
        self,
        session: VerifierSession,
        committed_token_id: int,
        model_runner: "GPUModelRunner",
    ) -> None:
        round_state = self._round_state(session)
        if round_state.committed_token_id != committed_token_id:
            raise ValueError(
                "prepared committed token must match the token being committed"
            )
        req_idx = model_runner.req_states.req_id_to_index[session.req_id]
        model_runner.commit_input_token(req_idx, committed_token_id)
        session.token_ids.append(committed_token_id)
        session.total_len += 1
        round_state.committed_token_committed = True

    def set_round_result(
        self,
        session: VerifierSession,
        result: VerifierRoundResult,
    ) -> None:
        self._validate_req_id(session.req_id, result.req_id)
        round_state = self._round_state(session)
        if not round_state.committed_token_committed:
            raise ValueError("committed token must be committed before postprocess")
        if not 0 <= result.accepted_len <= len(round_state.draft_token_ids):
            raise ValueError("accepted_len must be within the current draft range")
        session.token_ids.extend(round_state.draft_token_ids[: result.accepted_len])
        session.num_computed_tokens += 1 + result.accepted_len
        session.total_len += result.accepted_len
        round_state.committed_token_committed = False

    def clear_round_state(self, session: VerifierSession) -> None:
        self._round_state(session).reset()

    def remove_round_state(self, session: VerifierSession) -> None:
        self._round_states.pop(session.req_id, None)

    def _round_state(self, session: VerifierSession) -> VerifierRoundState:
        return self._round_states[session.req_id]

    @staticmethod
    def _validate_req_id(session_req_id: str, round_req_id: str) -> None:
        if session_req_id != round_req_id:
            raise ValueError("session and round req_id must match")
