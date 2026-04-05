# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import torch

from vllm.distributed import get_pp_group
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.dssd.protocol import (
    DSSDVerifierExecutionRequest,
    VerifierForwardResult,
    VerifyRoundRequest,
)


@dataclass(frozen=True)
class DSSDVerifierReplayRequestView:
    request_id: str
    prompt_token_ids: list[int]
    spec_token_ids: list[int]
    num_scheduled_tokens: int
    block_ids: tuple[list[int], ...]


def extract_forward_probs(
    logits: torch.Tensor,
    metadata,
) -> tuple[list[list[float]], list[float]]:
    target_probs = torch.softmax(
        logits[metadata.target_logits_indices].to(torch.float32),
        dim=-1,
    )
    bonus_probs = torch.softmax(
        logits[metadata.bonus_logits_indices].to(torch.float32),
        dim=-1,
    )
    seq_probs = target_probs.tolist()
    bonus_list = bonus_probs[0].tolist()
    return seq_probs, bonus_list


def build_verifier_result(
    request: VerifyRoundRequest | DSSDVerifierExecutionRequest,
    target_probs: list[list[float]],
    *,
    finished: bool = False,
    finish_reason: str | None = None,
) -> VerifierForwardResult:
    expected_num_probs = len(request.draft_token_ids) + 1
    if len(target_probs) != expected_num_probs:
        raise ValueError("target_probs must include one bonus distribution")
    return VerifierForwardResult(
        verifier_session_id=request.verifier_session_id,
        seq_no=request.seq_no,
        seq_probs=target_probs[:-1],
        bonus_probs=target_probs[-1],
        finished=finished,
        finish_reason=finish_reason,
    )


def build_verifier_result_from_logits(
    *,
    request: VerifyRoundRequest | DSSDVerifierExecutionRequest,
    logits: torch.Tensor,
    metadata,
    finished: bool = False,
    finish_reason: str | None = None,
) -> VerifierForwardResult:
    seq_probs, bonus_probs = extract_forward_probs(logits, metadata)
    return build_verifier_result(
        request=request,
        target_probs=[*seq_probs, bonus_probs],
        finished=finished,
        finish_reason=finish_reason,
    )


def build_verifier_replay_request_view(
    request: DSSDVerifierExecutionRequest,
    *,
    block_sizes: tuple[int, ...],
) -> DSSDVerifierReplayRequestView:
    if not request.committed_token_ids:
        raise ValueError("DSSD verifier replay requires committed_token_ids")
    if not request.draft_token_ids:
        raise ValueError("DSSD verifier replay requires draft_token_ids")
    if len(request.q_values) != len(request.draft_token_ids):
        raise ValueError("q_values must align with draft_token_ids")
    if not block_sizes:
        raise ValueError("block_sizes must not be empty")

    total_num_tokens = len(request.committed_token_ids) + len(request.draft_token_ids)
    block_ids = tuple(
        list(range(ceil(total_num_tokens / block_size)))
        for block_size in block_sizes
    )
    return DSSDVerifierReplayRequestView(
        request_id=f"dssd-verify:{request.verifier_session_id}:{request.seq_no}",
        prompt_token_ids=list(request.committed_token_ids),
        spec_token_ids=list(request.draft_token_ids),
        num_scheduled_tokens=total_num_tokens,
        block_ids=block_ids,
    )


def build_verifier_replay_scheduler_output(
    request: DSSDVerifierExecutionRequest,
    *,
    block_sizes: tuple[int, ...],
) -> tuple[DSSDVerifierReplayRequestView, SchedulerOutput]:
    view = build_verifier_replay_request_view(request, block_sizes=block_sizes)
    prompt_token_ids = list(view.prompt_token_ids)
    spec_token_ids = list(view.spec_token_ids)
    block_ids = tuple(list(block_id_list) for block_id_list in view.block_ids)
    new_req = NewRequestData(
        req_id=view.request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
        pooling_params=None,
        block_ids=block_ids,
        num_computed_tokens=0,
        lora_request=None,
    )
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[new_req],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={view.request_id: view.num_scheduled_tokens},
        total_num_scheduled_tokens=view.num_scheduled_tokens,
        scheduled_spec_decode_tokens={view.request_id: spec_token_ids},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    return view, scheduler_output


def _get_verifier_replay_block_sizes(model_runner) -> tuple[int, ...]:
    return tuple(
        group.kv_cache_spec.block_size
        for group in model_runner.kv_cache_config.kv_cache_groups
    )


def _cleanup_verifier_replay_request(model_runner, request_id: str) -> None:
    model_runner.requests.pop(request_id, None)
    model_runner.num_prompt_logprobs.pop(request_id, None)
    model_runner.late_interaction_runner.on_requests_finished({request_id})
    removed = model_runner.input_batch.remove_request(request_id)
    if removed is not None:
        model_runner.input_batch.condense()


def _is_output_pp_rank() -> bool:
    try:
        return get_pp_group().is_last_rank
    except AssertionError:
        return True


def run_verifier_replay_forward(
    model_runner,
    request: DSSDVerifierExecutionRequest,
) -> VerifierForwardResult | None:
    if not isinstance(request, DSSDVerifierExecutionRequest):
        raise TypeError("dssd_verify_round expects DSSDVerifierExecutionRequest")
    if model_runner.use_async_scheduling:
        raise NotImplementedError(
            "DSSD verifier replay does not support async scheduling yet"
        )
    if model_runner.is_pooling_model:
        raise NotImplementedError(
            "DSSD verifier replay does not support pooling models"
        )
    if model_runner.supports_mm_inputs:
        raise NotImplementedError(
            "DSSD verifier replay does not support multimodal models"
        )

    view, scheduler_output = build_verifier_replay_scheduler_output(
        request,
        block_sizes=_get_verifier_replay_block_sizes(model_runner),
    )
    try:
        result = model_runner.execute_model(scheduler_output)
        if not _is_output_pp_rank():
            return None
        if result is not None:
            raise RuntimeError(
                "DSSD verifier replay expected execute_model to cache logits state"
            )

        state = model_runner.execute_model_state
        if (
            state is None
            or state.logits is None
            or state.spec_decode_metadata is None
        ):
            raise RuntimeError(
                "DSSD verifier replay expected cached logits and "
                "spec decode metadata"
            )
        return build_verifier_result_from_logits(
            request=request,
            logits=state.logits,
            metadata=state.spec_decode_metadata,
            finish_reason="gpu-replay-forward",
        )
    finally:
        model_runner.execute_model_state = None
        model_runner._draft_token_ids = None
        model_runner._draft_token_req_ids = None
        model_runner.input_batch.prev_sampled_token_ids = None
        _cleanup_verifier_replay_request(model_runner, view.request_id)
