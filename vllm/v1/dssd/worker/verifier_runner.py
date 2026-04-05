# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import torch

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
