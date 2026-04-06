# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from types import SimpleNamespace

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
    verifier_session_id: str
    seq_no: int
    prompt_token_ids: list[int]
    spec_token_ids: list[int]
    scheduled_spec_decode_tokens: list[int]
    num_scheduled_tokens: int
    block_ids: tuple[list[int], ...]


@dataclass(frozen=True)
class DSSDVerifierReplayState:
    request_id: str
    verifier_session_id: str
    seq_no: int
    committed_token_ids: tuple[int, ...]
    draft_token_ids: tuple[int, ...]
    q_values: tuple[float, ...]
    scheduled_spec_decode_tokens: tuple[int, ...]
    num_scheduled_tokens: int
    block_ids: tuple[tuple[int, ...], ...]
    request_view: DSSDVerifierReplayRequestView
    scheduler_output: SchedulerOutput


@dataclass
class DSSDVerifierReplaySessionState:
    request_id: str
    verifier_session_id: str
    seq_no: int | None = None
    committed_token_ids: tuple[int, ...] = ()


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
    request_id: str | None = None,
) -> DSSDVerifierReplayRequestView:
    if not request.draft_token_ids:
        raise ValueError("DSSD verifier replay requires draft_token_ids")
    if len(request.q_values) != len(request.draft_token_ids):
        raise ValueError("q_values must align with draft_token_ids")
    if not block_sizes:
        raise ValueError("block_sizes must not be empty")

    total_num_tokens = len(request.committed_token_ids) + len(request.draft_token_ids)
    block_ids = _build_block_ids(total_num_tokens, block_sizes)
    return DSSDVerifierReplayRequestView(
        request_id=request_id
        or f"dssd-verify:{request.verifier_session_id}",
        verifier_session_id=request.verifier_session_id,
        seq_no=request.seq_no,
        prompt_token_ids=list(request.committed_token_ids),
        spec_token_ids=list(request.draft_token_ids),
        scheduled_spec_decode_tokens=list(request.draft_token_ids),
        num_scheduled_tokens=total_num_tokens,
        block_ids=block_ids,
    )


def _build_verifier_replay_scheduler_output(
    view: DSSDVerifierReplayRequestView,
) -> SchedulerOutput:
    prompt_token_ids = list(view.prompt_token_ids)
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
        scheduled_spec_decode_tokens={
            view.request_id: list(view.scheduled_spec_decode_tokens)
        },
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    return scheduler_output


def _build_block_ids(
    total_num_tokens: int,
    block_sizes: tuple[int, ...],
) -> tuple[list[int], ...]:
    return tuple(
        list(range(ceil(total_num_tokens / block_size)))
        for block_size in block_sizes
    )


def _build_verifier_cached_replay_scheduler_output(
    request: DSSDVerifierExecutionRequest,
    *,
    request_id: str,
    cached_block_ids: tuple[list[int], ...],
    num_output_tokens: int,
    block_sizes: tuple[int, ...],
) -> SchedulerOutput:
    total_num_tokens = len(request.committed_token_ids) + len(request.draft_token_ids)
    target_block_ids = _build_block_ids(total_num_tokens, block_sizes)
    new_block_ids = tuple(
        target_ids[len(current_ids):]
        for current_ids, target_ids in zip(cached_block_ids, target_block_ids)
    )
    cached_req = CachedRequestData(
        req_ids=[request_id],
        resumed_req_ids=set(),
        new_token_ids=[],
        all_token_ids={},
        new_block_ids=[new_block_ids if any(new_block_ids) else None],
        num_computed_tokens=[len(request.committed_token_ids) - 1],
        num_output_tokens=[num_output_tokens],
    )
    num_scheduled_tokens = 1 + len(request.draft_token_ids)
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_req,
        num_scheduled_tokens={request_id: num_scheduled_tokens},
        total_num_scheduled_tokens=num_scheduled_tokens,
        scheduled_spec_decode_tokens={request_id: list(request.draft_token_ids)},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def build_verifier_replay_state(
    request: DSSDVerifierExecutionRequest,
    *,
    block_sizes: tuple[int, ...],
    request_id: str | None = None,
) -> DSSDVerifierReplayState:
    view = build_verifier_replay_request_view(
        request,
        block_sizes=block_sizes,
        request_id=request_id,
    )
    scheduler_output = _build_verifier_replay_scheduler_output(view)
    return DSSDVerifierReplayState(
        request_id=view.request_id,
        verifier_session_id=request.verifier_session_id,
        seq_no=request.seq_no,
        committed_token_ids=tuple(request.committed_token_ids),
        draft_token_ids=tuple(request.draft_token_ids),
        q_values=tuple(request.q_values),
        scheduled_spec_decode_tokens=tuple(view.scheduled_spec_decode_tokens),
        num_scheduled_tokens=view.num_scheduled_tokens,
        block_ids=tuple(tuple(block_id_list) for block_id_list in view.block_ids),
        request_view=view,
        scheduler_output=scheduler_output,
    )


def build_verifier_replay_scheduler_output(
    request: DSSDVerifierExecutionRequest,
    *,
    block_sizes: tuple[int, ...],
    request_id: str | None = None,
) -> tuple[DSSDVerifierReplayRequestView, SchedulerOutput]:
    state = build_verifier_replay_state(
        request,
        block_sizes=block_sizes,
        request_id=request_id,
    )
    return state.request_view, state.scheduler_output


def _get_verifier_replay_block_sizes(model_runner) -> tuple[int, ...]:
    return tuple(
        group.kv_cache_spec.block_size
        for group in model_runner.kv_cache_config.kv_cache_groups
    )


def _get_or_create_verifier_replay_session_state(
    model_runner,
    verifier_session_id: str,
) -> DSSDVerifierReplaySessionState:
    replay_sessions = getattr(model_runner, "_dssd_verifier_replay_sessions", None)
    if replay_sessions is None:
        replay_sessions = {}
        model_runner._dssd_verifier_replay_sessions = replay_sessions

    session_state = replay_sessions.get(verifier_session_id)
    if session_state is None:
        session_state = DSSDVerifierReplaySessionState(
            request_id=f"dssd-verify:{verifier_session_id}",
            verifier_session_id=verifier_session_id,
        )
        replay_sessions[verifier_session_id] = session_state
    return session_state


def _cleanup_verifier_replay_request(model_runner, request_id: str) -> None:
    _cleanup_verifier_replay_batch_state(model_runner, request_id)
    model_runner.requests.pop(request_id, None)
    model_runner.num_prompt_logprobs.pop(request_id, None)
    model_runner.late_interaction_runner.on_requests_finished({request_id})


def _cleanup_verifier_replay_batch_state(model_runner, request_id: str) -> None:
    removed = model_runner.input_batch.remove_request(request_id)
    if removed is not None:
        model_runner.input_batch.condense()


def _reset_verifier_replay_session_state(
    model_runner,
    verifier_session_id: str,
) -> None:
    replay_sessions = getattr(model_runner, "_dssd_verifier_replay_sessions", None)
    request_id = f"dssd-verify:{verifier_session_id}"
    if replay_sessions is not None:
        session_state = replay_sessions.pop(verifier_session_id, None)
        if session_state is not None:
            request_id = session_state.request_id
    _cleanup_verifier_replay_request(model_runner, request_id)


def _bootstrap_verifier_replay_session_state(
    model_runner,
    request: DSSDVerifierExecutionRequest,
    session_state: DSSDVerifierReplaySessionState,
) -> None:
    request_id = session_state.request_id
    req_state = model_runner.requests.get(request_id)
    if req_state is None:
        raise RuntimeError(
            "missing cached verifier replay request for non-initial round"
        )

    prompt_token_ids = req_state.prompt_token_ids
    if prompt_token_ids is None:
        raise RuntimeError(
            "cached verifier replay request must retain prompt_token_ids"
        )

    prompt_len = len(prompt_token_ids)
    current_committed_prefix = tuple(prompt_token_ids) + tuple(
        req_state.output_token_ids
    )
    if prompt_len > len(request.committed_token_ids) or (
            tuple(request.committed_token_ids[:prompt_len])
            != tuple(prompt_token_ids)):
        raise RuntimeError(
            "committed prefix diverged from cached verifier replay request"
        )
    if len(current_committed_prefix) > len(request.committed_token_ids):
        raise RuntimeError(
            "committed prefix must extend previous verifier replay state"
        )

    session_state.seq_no = max(request.seq_no - 1, 0)
    session_state.committed_token_ids = current_committed_prefix


def _is_prefix_extension(
    committed_prefix: tuple[int, ...],
    candidate_prefix: list[int],
) -> bool:
    prefix_len = len(committed_prefix)
    return tuple(candidate_prefix[:prefix_len]) == committed_prefix


def _sync_cached_verifier_request_state(
    model_runner,
    *,
    request_id: str,
    committed_token_ids: list[int],
    block_sizes: tuple[int, ...],
) -> tuple[tuple[list[int], ...], int]:
    req_state = model_runner.requests.get(request_id)
    if req_state is None:
        raise RuntimeError(
            "missing cached verifier replay request for non-initial round"
        )

    prompt_token_ids = req_state.prompt_token_ids
    if prompt_token_ids is None:
        raise RuntimeError(
            "cached verifier replay request must retain prompt_token_ids"
        )
    prompt_len = len(prompt_token_ids)
    if prompt_len > len(committed_token_ids) or (
            tuple(committed_token_ids[:prompt_len]) != tuple(prompt_token_ids)):
        raise RuntimeError(
            "committed prefix diverged from cached verifier replay request"
        )

    output_token_ids = committed_token_ids[prompt_len:]
    req_state.output_token_ids.clear()
    req_state.output_token_ids.extend(output_token_ids)
    req_state.num_computed_tokens = max(len(committed_token_ids) - 1, 0)
    req_state.block_ids = _build_block_ids(len(committed_token_ids), block_sizes)
    _sync_cached_verifier_batch_state(
        model_runner,
        request_id=request_id,
        prompt_len=prompt_len,
        output_token_ids=output_token_ids,
        num_computed_tokens=req_state.num_computed_tokens,
    )
    return req_state.block_ids, len(output_token_ids)


def _sync_cached_verifier_batch_state(
    model_runner,
    *,
    request_id: str,
    prompt_len: int,
    output_token_ids: list[int],
    num_computed_tokens: int,
) -> None:
    input_batch = model_runner.input_batch
    req_id_to_index = getattr(input_batch, "req_id_to_index", None)
    if not isinstance(req_id_to_index, dict):
        return

    req_index = req_id_to_index.get(request_id)
    if req_index is None:
        return

    num_tokens = prompt_len + len(output_token_ids)
    if hasattr(input_batch, "num_computed_tokens_cpu"):
        input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
    if hasattr(input_batch, "num_tokens_no_spec"):
        input_batch.num_tokens_no_spec[req_index] = num_tokens
    if hasattr(input_batch, "req_output_token_ids"):
        input_batch.req_output_token_ids[req_index] = (
            model_runner.requests[request_id].output_token_ids
        )
    block_table = getattr(input_batch, "block_table", None)
    if block_table is not None and hasattr(block_table, "add_row"):
        block_table.add_row(model_runner.requests[request_id].block_ids, req_index)
    token_ids_cpu = getattr(input_batch, "token_ids_cpu", None)
    if token_ids_cpu is not None:
        start_idx = prompt_len
        end_idx = start_idx + len(output_token_ids)
        if torch.is_tensor(token_ids_cpu):
            token_ids_cpu[req_index, start_idx:end_idx] = torch.tensor(
                output_token_ids,
                dtype=token_ids_cpu.dtype,
                device=token_ids_cpu.device,
            )
        else:
            token_ids_cpu[req_index, start_idx:end_idx] = output_token_ids


def _clear_verifier_replay_transient_state(model_runner) -> None:
    model_runner.execute_model_state = None
    model_runner.kv_connector_output = None
    model_runner._draft_token_ids = None
    model_runner._draft_token_req_ids = None
    model_runner.input_batch.prev_sampled_token_ids = None


def _is_output_pp_rank() -> bool:
    try:
        return get_pp_group().is_last_rank
    except AssertionError:
        return True


def _execute_verifier_replay_requests(
    model_runner,
    replay_requests: list[dict[str, object]],
):
    execute_batch = getattr(model_runner, "execute_verifier_replay_requests", None)
    if execute_batch is not None:
        return execute_batch(replay_requests=replay_requests)
    if len(replay_requests) != 1:
        raise RuntimeError(
            "DSSD verifier replay batch execution requires "
            "execute_verifier_replay_requests"
        )

    replay_request = replay_requests[0]
    return model_runner.execute_verifier_replay_request(
        request_id=replay_request["request_id"],
        committed_token_ids=replay_request["committed_token_ids"],
        draft_token_ids=replay_request["draft_token_ids"],
    )


def _build_verifier_results_from_batch_logits(
    requests: list[DSSDVerifierExecutionRequest],
    *,
    logits: torch.Tensor,
    metadata,
) -> list[VerifierForwardResult]:
    if len(requests) == 1 and not hasattr(metadata, "num_draft_tokens"):
        return [
            build_verifier_result_from_logits(
                request=requests[0],
                logits=logits,
                metadata=metadata,
                finish_reason="gpu-replay-forward",
            )
        ]

    num_draft_tokens = getattr(metadata, "num_draft_tokens", None)
    if num_draft_tokens is None or len(num_draft_tokens) != len(requests):
        raise RuntimeError(
            "DSSD verifier replay expected batched num_draft_tokens metadata"
        )

    target_start = 0
    results: list[VerifierForwardResult] = []
    for index, (request, expected_targets) in enumerate(
        zip(requests, num_draft_tokens)
    ):
        target_end = target_start + expected_targets
        request_metadata = SimpleNamespace(
            target_logits_indices=metadata.target_logits_indices[
                target_start:target_end
            ],
            bonus_logits_indices=metadata.bonus_logits_indices[index:index + 1],
        )
        results.append(
            build_verifier_result_from_logits(
                request=request,
                logits=logits,
                metadata=request_metadata,
                finish_reason="gpu-replay-forward",
            )
        )
        target_start = target_end

    if target_start != len(metadata.target_logits_indices):
        raise RuntimeError(
            "DSSD verifier replay target logits metadata did not align with "
            "batched requests"
        )
    return results


def close_verifier_replay_session(model_runner, request) -> bool:
    from vllm.v1.dssd.protocol import CloseSessionRequest

    if not isinstance(request, CloseSessionRequest):
        raise TypeError(
            "dssd_close_verifier_session expects CloseSessionRequest"
        )

    replay_sessions = getattr(model_runner, "_dssd_verifier_replay_sessions", None)
    request_id = f"dssd-verify:{request.verifier_session_id}"
    had_session = False
    if replay_sessions is not None:
        session_state = replay_sessions.get(request.verifier_session_id)
        if session_state is not None:
            request_id = session_state.request_id
            had_session = True

    had_request = request_id in model_runner.requests
    _reset_verifier_replay_session_state(model_runner, request.verifier_session_id)
    _clear_verifier_replay_transient_state(model_runner)
    return had_session or had_request


def run_verifier_replay_forward(
    model_runner,
    request: DSSDVerifierExecutionRequest,
) -> VerifierForwardResult | None:
    results = run_verifier_replay_forward_batch(model_runner, [request])
    return results[0]


def run_verifier_replay_forward_batch(
    model_runner,
    requests: list[DSSDVerifierExecutionRequest],
) -> list[VerifierForwardResult | None]:
    if not requests:
        return []

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

    replay_batches: list[tuple[DSSDVerifierExecutionRequest,
                               DSSDVerifierReplaySessionState]] = []
    verifier_session_ids: list[str] = []

    try:
        for request in requests:
            if not isinstance(request, DSSDVerifierExecutionRequest):
                raise TypeError(
                    "dssd_verify_round expects DSSDVerifierExecutionRequest"
                )

            replay_session_state = _get_or_create_verifier_replay_session_state(
                model_runner,
                request.verifier_session_id,
            )
            verifier_session_ids.append(request.verifier_session_id)
            request_id = replay_session_state.request_id
            has_stale_request = request_id in model_runner.requests

            if request.seq_no == 0 and (
                replay_session_state.seq_no is not None or has_stale_request
            ):
                _reset_verifier_replay_session_state(
                    model_runner, request.verifier_session_id
                )
                replay_session_state = _get_or_create_verifier_replay_session_state(
                    model_runner,
                    request.verifier_session_id,
                )

            if replay_session_state.seq_no is None:
                if request.seq_no != 0:
                    raise RuntimeError(
                        "verifier replay requires seq_no == 0 after session reset"
                    )
            else:
                if request.seq_no <= replay_session_state.seq_no:
                    raise RuntimeError(
                        "verifier replay requires strictly increasing seq_no"
                    )
                if not _is_prefix_extension(
                    replay_session_state.committed_token_ids,
                    request.committed_token_ids,
                ):
                    raise RuntimeError(
                        "committed prefix must extend previous verifier replay state"
                    )

            replay_batches.append((request, replay_session_state))

        result = _execute_verifier_replay_requests(
            model_runner,
            replay_requests=[
                {
                    "request_id": replay_session_state.request_id,
                    "committed_token_ids": request.committed_token_ids,
                    "draft_token_ids": request.draft_token_ids,
                }
                for request, replay_session_state in replay_batches
            ],
        )
        if not _is_output_pp_rank():
            for request, replay_session_state in replay_batches:
                replay_session_state.seq_no = request.seq_no
                replay_session_state.committed_token_ids = tuple(
                    request.committed_token_ids
                )
            return [None] * len(replay_batches)
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
        for request, replay_session_state in replay_batches:
            replay_session_state.seq_no = request.seq_no
            replay_session_state.committed_token_ids = tuple(
                request.committed_token_ids
            )
        return _build_verifier_results_from_batch_logits(
            [request for request, _ in replay_batches],
            logits=state.logits,
            metadata=state.spec_decode_metadata,
        )
    except Exception:
        for verifier_session_id in dict.fromkeys(verifier_session_ids):
            _reset_verifier_replay_session_state(model_runner, verifier_session_id)
        raise
    finally:
        _clear_verifier_replay_transient_state(model_runner)
