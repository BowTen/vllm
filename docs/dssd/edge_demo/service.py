from __future__ import annotations

from typing import Protocol

import torch

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams

from .engine import EdgeDecodeEngine
from .types import (
    EdgeOpenSessionResult,
    EdgeSession,
    EdgeVerifyRequest,
    EdgeVerifyResponse,
)


class VerifierTransport(Protocol):
    """最小 verifier transport 抽象。"""

    def open_session(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        lora_request: LoRARequest | None = None,
    ) -> EdgeOpenSessionResult: ...

    def verify(self, request: EdgeVerifyRequest) -> EdgeVerifyResponse: ...

    def close_session(self, req_id: str) -> None: ...


class DSSDEdgeService:
    """最外层 edge 服务入口。"""

    def __init__(
        self,
        decode_engine: EdgeDecodeEngine,
        verifier: VerifierTransport,
        eos_token_id: int,
        gamma: int,
    ) -> None:
        self.decode_engine = decode_engine
        self.verifier = verifier
        self.eos_token_id = eos_token_id
        self.gamma = gamma

    def open_session(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        lora_request: LoRARequest | None = None,
    ) -> EdgeOpenSessionResult:
        remote_result = self.verifier.open_session(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        session = self.decode_engine.open_session(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        bootstrap_token_id = self.decode_engine.prefill(
            session,
            bootstrap_token_id=remote_result.bootstrap_token_id,
        )
        return EdgeOpenSessionResult(
            req_id=req_id,
            bootstrap_token_id=bootstrap_token_id,
            session=session,
        )

    def generate(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        lora_request: LoRARequest | None = None,
    ) -> list[int]:
        remote_result = self.verifier.open_session(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        session = self.decode_engine.open_session(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        committed_token_id = self.decode_engine.prefill(
            session,
            bootstrap_token_id=remote_result.bootstrap_token_id,
        )

        if self._has_eos_in_recent_committed_tokens(session, committed_count=1):
            self._rollback_tokens_after_recent_eos(session, committed_count=1)
            self.close_session(session.req_id)
            return session.committed_output_ids()

        while True:
            round_state = self.decode_engine.draft(
                session,
                committed_token_id,
                self.gamma,
            )
            request = EdgeVerifyRequest(
                req_id=req_id,
                committed_token_id=committed_token_id,
                draft_token_ids=list(round_state.draft_token_ids),
                draft_q_values=list(round_state.draft_q_values),
            )
            response = self.verifier.verify(request)
            committed_token_id, committed_count = self._commit_verify_result(
                session,
                response,
            )
            if self._has_eos_in_recent_committed_tokens(session, committed_count):
                self._rollback_tokens_after_recent_eos(
                    session,
                    committed_count,
                )
                break

        self.close_session(session.req_id)
        return session.committed_output_ids()

    def close_session(self, req_id: str) -> None:
        session = self.decode_engine.sessions[req_id]
        self.decode_engine.close_session(session)
        self.verifier.close_session(req_id)

    def _commit_verify_result(
        self,
        session: EdgeSession,
        response: EdgeVerifyResponse,
    ) -> tuple[int, int]:
        draft_len = len(session.round_state.draft_token_ids)
        rejected_count = draft_len - response.accepted_len
        if rejected_count > 0:
            self.decode_engine.rollback(session, rejected_count)
            next_token_id = self._resample_rejected_token(session, response)
        else:
            if response.next_token_id is None:
                raise RuntimeError("全接受时 verifier 必须返回 next_token_id")
            next_token_id = response.next_token_id
        committed_count = response.accepted_len + 1
        return (
            self.decode_engine.commit_external_token(session, next_token_id),
            committed_count,
        )

    def _resample_rejected_token(
        self,
        session: EdgeSession,
        response: EdgeVerifyResponse,
    ) -> int:
        rejected_index = response.accepted_len
        if response.rejected_target_logits is None:
            raise RuntimeError("拒绝时 verifier 必须返回 rejected_target_logits")

        q_logits = session.round_state.q_dist_at(rejected_index)
        p_logits = response.rejected_target_logits
        q_probs = torch.softmax(q_logits, dim=-1)
        p_probs = torch.softmax(p_logits, dim=-1)
        residual = torch.clamp(p_probs - q_probs, min=0.0)
        norm = residual.sum()
        if float(norm.item()) <= 0.0:
            residual = p_probs
            norm = residual.sum()
        sampled = torch.multinomial(residual / norm, num_samples=1)
        return int(sampled.item())

    def _has_eos_in_recent_committed_tokens(
        self,
        session: EdgeSession,
        committed_count: int,
    ) -> bool:
        recent = session.token_ids[-committed_count:]
        return any(self._is_eos(token_id) for token_id in recent)

    def _rollback_tokens_after_recent_eos(
        self,
        session: EdgeSession,
        committed_count: int,
    ) -> None:
        recent = session.token_ids[-committed_count:]
        for index, token_id in enumerate(recent):
            if not self._is_eos(token_id):
                continue
            trailing_count = committed_count - index - 1
            if trailing_count > 0:
                self.decode_engine.rollback(session, trailing_count)
            return

    def _is_eos(self, token_id: int) -> bool:
        return int(token_id) == self.eos_token_id
