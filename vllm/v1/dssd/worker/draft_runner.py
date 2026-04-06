# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from copy import copy
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
from vllm.v1.dssd.protocol import DraftRoundRequest
from vllm.v1.sample.sampler import Sampler


@dataclass
class DraftRoundResult:
    draft_token_ids: list[int]
    q_values: list[float]
    q_dists_handle: str
    q_distributions: list[list[float]] | None = None


@dataclass(frozen=True)
class DSSDDraftReplayRequestView:
    request_id: str
    prompt_token_ids: list[int]
    num_scheduled_tokens: int
    block_ids: tuple[list[int], ...]


def _get_draft_replay_sampling_params(
    request: DraftRoundRequest,
    *,
    draft_token_ids: list[int],
) -> SamplingParams:
    if request.sampling_params is None:
        return SamplingParams(temperature=0.0, max_tokens=1)

    clone = getattr(request.sampling_params, "clone", None)
    if callable(clone):
        sampling_params = clone()
    else:
        sampling_params = SamplingParams(
            temperature=request.sampling_params.temperature,
            top_p=request.sampling_params.top_p,
            top_k=request.sampling_params.top_k,
            min_p=request.sampling_params.min_p,
            seed=request.sampling_params.seed,
            min_tokens=getattr(request.sampling_params, "min_tokens", 0),
            max_tokens=request.sampling_params.max_tokens,
        )
    sampling_params.max_tokens = 1
    original_min_tokens = getattr(request.sampling_params, "min_tokens", 0)
    consumed_min_tokens = len(request.committed_token_ids) + len(draft_token_ids)
    if hasattr(sampling_params, "min_tokens"):
        sampling_params.min_tokens = max(original_min_tokens - consumed_min_tokens, 0)
        sampling_params.min_tokens = 1 if sampling_params.min_tokens > 0 else 0
    return sampling_params


def build_draft_result_from_distribution(
    *,
    q_distribution: torch.Tensor,
    sampled_token_id: int,
    q_dists_handle: str,
) -> DraftRoundResult:
    if q_distribution.ndim != 1:
        raise ValueError("draft q_distribution must be one-dimensional")
    if sampled_token_id < 0 or sampled_token_id >= q_distribution.shape[-1]:
        raise ValueError("sampled_token_id is out of bounds for q_distribution")

    distribution = q_distribution.tolist()
    return DraftRoundResult(
        draft_token_ids=[sampled_token_id],
        q_values=[float(distribution[sampled_token_id])],
        q_dists_handle=q_dists_handle,
        q_distributions=[distribution],
    )


def _get_draft_replay_processed_sampler(model_runner) -> Sampler:
    sampler = getattr(model_runner, "_dssd_processed_sampler", None)
    if sampler is None:
        sampler = Sampler(logprobs_mode="processed_logprobs")
        model_runner._dssd_processed_sampler = sampler
    return sampler


def _build_draft_replay_sampling_metadata(
    *,
    request: DraftRoundRequest,
    draft_token_ids: list[int],
    model_runner,
    logits: torch.Tensor,
) -> object:
    input_batch = getattr(model_runner, "input_batch", None)
    if input_batch is None:
        raise ValueError("draft replay requires a real sampler and input batch")

    sampling_metadata = getattr(input_batch, "sampling_metadata", None)
    if sampling_metadata is None:
        raise ValueError("draft replay requires sampling metadata")

    replay_sampling_metadata = copy(sampling_metadata)
    replay_sampling_metadata.max_num_logprobs = -1
    replay_sampling_metadata.prompt_token_ids = torch.tensor(
        [request.prompt_token_ids],
        dtype=torch.int64,
        device=logits.device,
    )
    replay_sampling_metadata.output_token_ids = [
        [*request.committed_token_ids, *draft_token_ids]
    ]
    return replay_sampling_metadata


def _build_draft_step_result(
    *,
    model_runner,
    request: DraftRoundRequest,
    draft_token_ids: list[int],
    logits: torch.Tensor,
    q_dists_handle: str,
) -> tuple[int | None, DraftRoundResult | None]:
    sampler = _get_draft_replay_processed_sampler(model_runner)
    sampling_metadata = _build_draft_replay_sampling_metadata(
        model_runner=model_runner,
        request=request,
        draft_token_ids=draft_token_ids,
        logits=logits,
    )
    sampler_output = sampler(
        logits=logits,
        sampling_metadata=sampling_metadata,
    )
    sampled_ids = getattr(sampler_output, "sampled_token_ids", None)
    if sampled_ids is None or len(sampled_ids) == 0 or len(sampled_ids[0]) == 0:
        return None, None

    sampled_token_id = int(sampled_ids[0][0])
    logprobs_tensors = getattr(sampler_output, "logprobs_tensors", None)
    if logprobs_tensors is None or getattr(logprobs_tensors, "logprobs", None) is None:
        raise RuntimeError(
            "DSSD draft replay expected processed logprobs from sampler"
        )

    if sampling_metadata.all_greedy:
        q_distribution = torch.zeros_like(logprobs_tensors.logprobs[-1])
        q_distribution[sampled_token_id] = 1.0
    else:
        q_distribution = logprobs_tensors.logprobs[-1].exp()

    return sampled_token_id, build_draft_result_from_distribution(
        q_distribution=q_distribution,
        sampled_token_id=sampled_token_id,
        q_dists_handle=q_dists_handle,
    )


def build_draft_replay_request_view(
    request: DraftRoundRequest,
    *,
    draft_token_ids: list[int],
    step_index: int,
    block_sizes: tuple[int, ...],
) -> DSSDDraftReplayRequestView:
    if step_index < 0:
        raise ValueError("step_index must be non-negative")
    if not block_sizes:
        raise ValueError("block_sizes must not be empty")

    prompt_token_ids = [
        *request.prompt_token_ids,
        *request.committed_token_ids,
        *draft_token_ids,
    ]
    total_num_tokens = len(prompt_token_ids)
    block_ids = tuple(
        list(range(ceil(total_num_tokens / block_size)))
        for block_size in block_sizes
    )
    return DSSDDraftReplayRequestView(
        request_id=(
            f"dssd-draft:{request.local_session_id}:{request.seq_no}:{step_index}"
        ),
        prompt_token_ids=prompt_token_ids,
        num_scheduled_tokens=total_num_tokens,
        block_ids=block_ids,
    )


def build_draft_replay_scheduler_output(
    request: DraftRoundRequest,
    *,
    draft_token_ids: list[int],
    step_index: int,
    block_sizes: tuple[int, ...],
) -> tuple[DSSDDraftReplayRequestView, SchedulerOutput]:
    view = build_draft_replay_request_view(
        request,
        draft_token_ids=draft_token_ids,
        step_index=step_index,
        block_sizes=block_sizes,
    )
    new_req = NewRequestData(
        req_id=view.request_id,
        prompt_token_ids=list(view.prompt_token_ids),
        mm_features=[],
        sampling_params=_get_draft_replay_sampling_params(
            request,
            draft_token_ids=draft_token_ids,
        ),
        pooling_params=None,
        block_ids=tuple(list(block_id_list) for block_id_list in view.block_ids),
        num_computed_tokens=0,
        lora_request=None,
    )
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[new_req],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={view.request_id: view.num_scheduled_tokens},
        total_num_scheduled_tokens=view.num_scheduled_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    return view, scheduler_output


def _get_draft_replay_block_sizes(model_runner) -> tuple[int, ...]:
    return tuple(
        group.kv_cache_spec.block_size
        for group in model_runner.kv_cache_config.kv_cache_groups
    )


def _cleanup_draft_replay_request(model_runner, request_id: str) -> None:
    model_runner.requests.pop(request_id, None)
    model_runner.num_prompt_logprobs.pop(request_id, None)
    model_runner.late_interaction_runner.on_requests_finished({request_id})
    removed = model_runner.input_batch.remove_request(request_id)
    if removed is not None:
        model_runner.input_batch.condense()


def _get_pp_group_or_none():
    try:
        return get_pp_group()
    except AssertionError:
        return None


def _is_output_pp_rank() -> bool:
    pp = _get_pp_group_or_none()
    if pp is None:
        return True
    return pp.is_last_rank


def _broadcast_sampled_token(sampled_token_id: int | None) -> int | None:
    pp = _get_pp_group_or_none()
    if pp is None or pp.world_size == 1:
        return sampled_token_id

    if pp.is_last_rank:
        value = -1 if sampled_token_id is None else sampled_token_id
        token_tensor = torch.tensor([value], dtype=torch.int64, device=pp.device)
    else:
        token_tensor = torch.empty(1, dtype=torch.int64, device=pp.device)
    torch.distributed.broadcast(
        token_tensor,
        src=pp.last_rank,
        group=pp.device_group,
    )
    token_id = int(token_tensor.item())
    return None if token_id < 0 else token_id


def run_draft_replay_forward(
    model_runner,
    request: DraftRoundRequest,
) -> DraftRoundResult | None:
    if not isinstance(request, DraftRoundRequest):
        raise TypeError("dssd_draft_round expects DraftRoundRequest")
    if model_runner.use_async_scheduling:
        raise NotImplementedError(
            "DSSD draft replay does not support async scheduling yet"
        )
    if model_runner.is_pooling_model:
        raise NotImplementedError(
            "DSSD draft replay does not support pooling models"
        )
    if model_runner.supports_mm_inputs:
        raise NotImplementedError(
            "DSSD draft replay does not support multimodal models"
        )

    block_sizes = _get_draft_replay_block_sizes(model_runner)
    q_dists_handle = f"{request.local_session_id}:{request.seq_no}"
    draft_token_ids: list[int] = []
    q_values: list[float] = []
    q_distributions: list[list[float]] = []

    for step_index in range(request.gamma):
        view, scheduler_output = build_draft_replay_scheduler_output(
            request,
            draft_token_ids=draft_token_ids,
            step_index=step_index,
            block_sizes=block_sizes,
        )
        try:
            result = model_runner.execute_model(scheduler_output)
            if _is_output_pp_rank():
                if result is not None:
                    raise RuntimeError(
                        "DSSD draft replay expected execute_model to cache logits "
                        "state on the output PP rank"
                    )
                state = model_runner.execute_model_state
                if state is None or state.logits is None:
                    raise RuntimeError(
                        "DSSD draft replay expected cached logits on the output "
                        "PP rank"
                    )

                sampled_token_id, step_result = _build_draft_step_result(
                    model_runner=model_runner,
                    request=request,
                    draft_token_ids=draft_token_ids,
                    logits=state.logits,
                    q_dists_handle=q_dists_handle,
                )
            else:
                sampled_token_id = None
                step_result = None

            sampled_token_id = _broadcast_sampled_token(sampled_token_id)
            if sampled_token_id is None:
                break

            draft_token_ids.append(sampled_token_id)
            if step_result is not None and step_result.q_distributions is not None:
                q_values.extend(step_result.q_values)
                q_distributions.extend(step_result.q_distributions)
        finally:
            model_runner.execute_model_state = None
            model_runner.kv_connector_output = None
            model_runner._draft_token_ids = None
            model_runner._draft_token_req_ids = None
            model_runner.input_batch.prev_sampled_token_ids = None
            _cleanup_draft_replay_request(model_runner, view.request_id)

    if not _is_output_pp_rank():
        return None
    return DraftRoundResult(
        draft_token_ids=draft_token_ids,
        q_values=q_values,
        q_dists_handle=q_dists_handle,
        q_distributions=q_distributions,
    )
