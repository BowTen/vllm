from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from vllm.dssd.edge.types import EdgeOpenSessionResult
from vllm.dssd.protocol import VerifyRoundRequest

_GREEDY_TEMPERATURE_EPS = 1e-5
_RandomSampleFn = Callable[[torch.Tensor, dict[int, torch.Generator]], torch.Tensor]
random_sample: _RandomSampleFn | None
random_sample = None


def _random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    global random_sample
    if random_sample is None:
        from vllm.v1.sample.ops.topk_topp_sampler import random_sample as sample

        random_sample = sample
    return random_sample(probs, generators)


def _trace_top_entries(
    probs: torch.Tensor,
    top_k: int,
) -> list[dict[str, float | int]]:
    limit = min(int(top_k), int(probs.numel()))
    if limit <= 0:
        return []
    values, indices = torch.topk(probs.detach().to(torch.float32), k=limit)
    return [
        {"token_id": int(token_id.item()), "value": float(value.item())}
        for value, token_id in zip(values.cpu(), indices.cpu())
    ]


def _append_trace_record(record: dict) -> None:
    trace_path = os.environ.get("DSSD_CORRECTNESS_TRACE")
    if not trace_path:
        return
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _trace_enabled() -> bool:
    return bool(os.environ.get("DSSD_CORRECTNESS_TRACE"))


@dataclass
class EdgeGenerationStats:
    total_rounds: int = 0
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    all_accept_rounds: int = 0

    @property
    def draft_acceptance_rate(self) -> float:
        if self.total_draft_tokens <= 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    @property
    def all_accept_round_rate(self) -> float:
        if self.total_rounds <= 0:
            return 0.0
        return self.all_accept_rounds / self.total_rounds

    @property
    def avg_accepted_len_per_round(self) -> float:
        if self.total_rounds <= 0:
            return 0.0
        return self.total_accepted_tokens / self.total_rounds

    def record_round(self, *, draft_len: int, accepted_len: int) -> None:
        self.total_rounds += 1
        self.total_draft_tokens += int(draft_len)
        self.total_accepted_tokens += int(accepted_len)
        if accepted_len == draft_len:
            self.all_accept_rounds += 1


class DSSDEdgeService:
    def __init__(
        self,
        *,
        decode_engine,
        verifier,
        tokenizer=None,
        eos_token_id: int,
        gamma: int,
    ) -> None:
        self.decode_engine = decode_engine
        self.verifier = verifier
        self.tokenizer = tokenizer
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
        output_ids, _ = self.generate_with_stats(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        return output_ids

    def generate_with_stats(
        self,
        *,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params,
        lora_request=None,
    ) -> tuple[list[int], EdgeGenerationStats]:
        max_tokens = sampling_params.max_tokens or 0
        stats = EdgeGenerationStats()
        if max_tokens <= 0:
            return [], stats

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
                session._dssd_round_index = (
                    getattr(session, "_dssd_round_index", 0) + 1
                )
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
                stats.record_round(
                    draft_len=len(round_state.draft_token_ids),
                    accepted_len=response.accepted_len,
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

            return session.committed_output_ids(), stats
        finally:
            if req_id in self.decode_engine.sessions:
                self.close_session(req_id)

    def complete(
        self,
        *,
        req_id: str,
        prompt: str,
        sampling_params,
        lora_request=None,
    ) -> str:
        if self.tokenizer is None:
            raise RuntimeError("DSSDEdgeService.complete requires a tokenizer")
        prompt_token_ids = list(self.tokenizer(prompt).input_ids)
        output_ids = self.generate(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        return self.tokenizer.decode(
            list(output_ids),
            skip_special_tokens=True,
        )

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
        rejected_token_id = getattr(response, "rejected_token_id", None)

        if self._is_greedy_session(session):
            if rejected_token_id is not None:
                return int(rejected_token_id)
            if response.rejected_target_logits is None:
                raise RuntimeError(
                    "greedy rejected verifier response requires token_id or logits"
                )
            return int(torch.argmax(response.rejected_target_logits).item())

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
        recovery_probs = residual / norm
        trace_recovery_probs = (
            recovery_probs.detach().clone() if _trace_enabled() else None
        )
        probs = recovery_probs.view(1, -1)
        generator = getattr(session, "_dssd_sampling_generator", None)
        generators = {0: generator} if generator is not None else {}
        sampled = _random_sample(probs, generators)
        sampled_token_id = int(sampled.item())
        if trace_recovery_probs is not None:
            rejected_draft_token_id = int(
                session.round_state.draft_token_ids[rejected_index]
            )
            _append_trace_record(
                {
                    "source": "dssd_edge_resample",
                    "req_id": session.req_id,
                    "round_index": getattr(session, "_dssd_round_index", None),
                    "accepted_len": response.accepted_len,
                    "rejected_index": rejected_index,
                    "confirmed_prefix": list(session.token_ids),
                    "draft_token_ids": list(session.round_state.draft_token_ids),
                    "draft_q_values": list(session.round_state.draft_q_values),
                    "rejected_draft_token_id": rejected_draft_token_id,
                    "p_value": float(p_probs[rejected_draft_token_id].item()),
                    "q_value": float(q_probs[rejected_draft_token_id].item()),
                    "residual_norm": float(norm.item()),
                    "recovery_token_id": sampled_token_id,
                    "p_top": _trace_top_entries(p_probs, 20),
                    "q_top": _trace_top_entries(q_probs, 20),
                    "residual_top": _trace_top_entries(
                        trace_recovery_probs, 20
                    ),
                    "sample_score_top": _trace_top_entries(probs[0], 20),
                }
            )
        return sampled_token_id

    @staticmethod
    def _is_greedy_session(session) -> bool:
        sampling_params = getattr(session, "sampling_params", None)
        temperature = getattr(sampling_params, "temperature", None)
        return temperature is not None and float(temperature) < _GREEDY_TEMPERATURE_EPS

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
