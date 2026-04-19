from __future__ import annotations

import torch

from vllm.dssd.edge.types import EdgeOpenSessionResult
from vllm.dssd.protocol import VerifyRoundRequest


class DSSDEdgeService:
    def __init__(
        self,
        *,
        decode_engine,
        verifier,
        eos_token_id: int,
        gamma: int,
    ) -> None:
        self.decode_engine = decode_engine
        self.verifier = verifier
        self.eos_token_id = int(eos_token_id)
        self.gamma = int(gamma)

    def open_session(
        self,
        *,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params,
        lora_request=None,
    ) -> EdgeOpenSessionResult:
        remote_result = self.verifier.open_session(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        try:
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
        except Exception:
            self.verifier.close_session(req_id)
            raise
        return EdgeOpenSessionResult(
            req_id=req_id,
            bootstrap_token_id=bootstrap_token_id,
        )

    def generate(
        self,
        *,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params,
        lora_request=None,
    ) -> list[int]:
        max_tokens = sampling_params.max_tokens or 0
        if max_tokens <= 0:
            return []

        opened = self.open_session(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        session = self.decode_engine.sessions[req_id]
        try:
            committed_token_id = opened.bootstrap_token_id

            if self._output_len(session) >= max_tokens:
                overflow = self._output_len(session) - max_tokens
                if overflow > 0:
                    self.decode_engine.rollback(session, overflow)
                return session.committed_output_ids()

            if self._has_eos_in_recent_committed_tokens(session, committed_count=1):
                self._rollback_tokens_after_recent_eos(session, committed_count=1)
                return session.committed_output_ids()

            while True:
                round_state = self.decode_engine.draft(
                    session,
                    committed_token_id,
                    self.gamma,
                )
                response = self.verifier.verify_round(
                    VerifyRoundRequest(
                        req_id=req_id,
                        committed_token_id=committed_token_id,
                        draft_token_ids=list(round_state.draft_token_ids),
                        draft_q_values=list(round_state.draft_q_values),
                    )
                )
                committed_token_id, committed_count = self._commit_verify_result(
                    session,
                    response,
                )
                if self._output_len(session) >= max_tokens:
                    overflow = self._output_len(session) - max_tokens
                    if overflow > 0:
                        self.decode_engine.rollback(session, overflow)
                    break
                if self._has_eos_in_recent_committed_tokens(session, committed_count):
                    self._rollback_tokens_after_recent_eos(session, committed_count)
                    break

            return session.committed_output_ids()
        finally:
            if req_id in self.decode_engine.sessions:
                self.close_session(req_id)

    def close_session(self, req_id: str) -> None:
        session = self.decode_engine.sessions[req_id]
        self.decode_engine.close_session(session)
        self.verifier.close_session(req_id)

    def _commit_verify_result(
        self,
        session,
        response,
    ) -> tuple[int, int]:
        draft_len = len(session.round_state.draft_token_ids)
        rejected_count = draft_len - response.accepted_len
        if rejected_count > 0:
            self.decode_engine.rollback(session, rejected_count)
            next_token_id = self._resample_rejected_token(session, response)
        else:
            if response.bonus_token_id is None:
                raise RuntimeError("all-accepted verifier response requires bonus_token_id")
            next_token_id = response.bonus_token_id
        committed_count = response.accepted_len + 1
        return (
            self.decode_engine.commit_external_token(session, next_token_id),
            committed_count,
        )

    def _resample_rejected_token(self, session, response) -> int:
        rejected_index = response.accepted_len
        if response.rejected_target_logits is None:
            raise RuntimeError("rejected verifier response requires logits")

        q_logits = session.round_state.q_dist_at(rejected_index)
        p_logits = response.rejected_target_logits.to(
            device=q_logits.device,
            dtype=q_logits.dtype,
        )
        q_probs = torch.softmax(q_logits, dim=-1)
        p_probs = torch.softmax(p_logits, dim=-1)
        residual = torch.clamp(p_probs - q_probs, min=0.0)
        norm = residual.sum()
        if float(norm.item()) <= 0.0:
            residual = p_probs
            norm = residual.sum()
        sampled = torch.multinomial(residual / norm, num_samples=1)
        return int(sampled.item())

    def _has_eos_in_recent_committed_tokens(self, session, committed_count: int) -> bool:
        recent = session.token_ids[-committed_count:]
        return any(self._is_eos(token_id) for token_id in recent)

    def _rollback_tokens_after_recent_eos(self, session, committed_count: int) -> None:
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

    @staticmethod
    def _output_len(session) -> int:
        return len(session.committed_output_ids())
