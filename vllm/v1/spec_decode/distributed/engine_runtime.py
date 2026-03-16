# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import copy
import json
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import torch

from vllm import SamplingParams
from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind, StructuredOutputsParams
from vllm.v1.engine import FinishReason
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor import UniProcExecutor
from vllm.v1.request import Request
from vllm.v1.spec_decode.distributed.errors import VerifierSessionMissingError
from vllm.v1.spec_decode.distributed.logprobs import (
    dense_probs_from_packed_logprobs,
    pack_logprobs_lists,
    pack_logprobs_tensors,
    pack_sample_logprobs,
)
from vllm.v1.spec_decode.distributed.protocol import (
    CloseSessionRequest,
    DraftProposal,
    OpenSessionRequest,
    OpenSessionResponse,
    PackedLogprobs,
    ResyncSessionRequest,
    ResyncSessionResponse,
    SamplingMetadata,
    VerificationResult,
)
from vllm.v1.spec_decode.distributed.sampling import (
    is_terminal_token,
    sample_from_logits,
)
from vllm.v1.spec_decode.distributed.structured_output import (
    StructuredOutputFactory,
    accept_structured_output_tokens,
    apply_structured_output_mask,
    reset_structured_output_session,
    rollback_structured_output_tokens,
)
from vllm.v1.spec_decode.distributed.runtime import (
    DraftProposalOutput,
    serialize_probs,
)

if TYPE_CHECKING:
    pass

logger = init_logger(__name__)


def _raw_query_sampling_metadata(sampling: SamplingMetadata) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        seed=0,
        max_tokens=1,
        min_tokens=0,
        stop_token_ids=[],
        eos_token_id=None,
        ignore_eos=True,
        logprobs=sampling.logprobs,
        prompt_logprobs=sampling.prompt_logprobs,
    )


def _logits_from_packed_logprobs(
    packed: PackedLogprobs,
    vocab_size: int,
) -> torch.Tensor:
    probs = dense_probs_from_packed_logprobs(packed, vocab_size)
    return probs.clamp_min(torch.finfo(probs.dtype).tiny).log()


def _structured_outputs_from_sampling(
    sampling: SamplingMetadata,
) -> StructuredOutputsParams | None:
    if (
        sampling.structured_output_backend is None
        or sampling.structured_output_type is None
    ):
        return None

    spec = sampling.structured_output_spec
    params: StructuredOutputsParams
    match sampling.structured_output_type:
        case "JSON":
            params = StructuredOutputsParams(json=spec or "")
        case "JSON_OBJECT":
            params = StructuredOutputsParams(json_object=True)
        case "REGEX":
            params = StructuredOutputsParams(regex=spec or "")
        case "GRAMMAR":
            params = StructuredOutputsParams(grammar=spec or "")
        case "CHOICE":
            params = StructuredOutputsParams(choice=json.loads(spec or "[]"))
        case "STRUCTURAL_TAG":
            params = StructuredOutputsParams(structural_tag=spec or "")
        case _:
            raise ValueError(
                "Unsupported structured output type for distributed "
                f"runtime: {sampling.structured_output_type}."
            )

    params.disable_any_whitespace = (
        sampling.structured_output_disable_any_whitespace
    )
    params.disable_additional_properties = (
        sampling.structured_output_disable_additional_properties
    )
    params._backend = sampling.structured_output_backend
    params._backend_was_auto = False
    return params


def sampling_metadata_to_sampling_params(
    sampling: SamplingMetadata,
    *,
    max_tokens: int,
    logprobs: int | None,
    prompt_logprobs: int | None,
) -> SamplingParams:
    params = SamplingParams(
        max_tokens=max_tokens,
        min_tokens=sampling.min_tokens,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        top_k=sampling.top_k,
        min_p=sampling.min_p,
        presence_penalty=sampling.presence_penalty,
        frequency_penalty=sampling.frequency_penalty,
        repetition_penalty=sampling.repetition_penalty,
        seed=sampling.seed,
        stop_token_ids=list(sampling.stop_token_ids),
        ignore_eos=sampling.ignore_eos,
        logprobs=logprobs,
        prompt_logprobs=prompt_logprobs,
        detokenize=False,
        skip_special_tokens=False,
        include_stop_str_in_output=True,
        output_kind=RequestOutputKind.FINAL_ONLY,
        structured_outputs=_structured_outputs_from_sampling(sampling),
        bad_words=[],
    )
    params._eos_token_id = sampling.eos_token_id
    params.logit_bias = copy.deepcopy(sampling.logit_bias)
    params.allowed_token_ids = copy.deepcopy(sampling.allowed_token_ids)
    params._bad_words_token_ids = copy.deepcopy(sampling.bad_words_token_ids)
    return params


def _engine_request(
    request_id: str,
    prompt_token_ids: list[int],
    sampling: SamplingMetadata,
    *,
    max_tokens: int,
    logprobs: int | None,
    prompt_logprobs: int | None,
) -> Request:
    query_sampling = _raw_query_sampling_metadata(sampling)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(prompt_token_ids),
        sampling_params=sampling_metadata_to_sampling_params(
            query_sampling,
            max_tokens=max_tokens,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
        ),
        pooling_params=None,
        arrival_time=time.time(),
        mm_features=None,
        resumable=False,
    )


@dataclass
class EngineChunkResult:
    token_ids: list[int]
    logprobs: list[PackedLogprobs]
    prompt_logprobs: list[PackedLogprobs]
    finish_reason: FinishReason | None
    stop_reason: int | str | None


@dataclass
class EngineQuery:
    session_id: str
    prefix_token_ids: list[int]
    sampling: SamplingMetadata
    max_tokens: int
    logprobs: int | None = None
    prompt_logprobs: int | None = None


@dataclass
class _PendingEngineQuery:
    request_id: str
    prefix_token_ids: list[int]
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[PackedLogprobs] = field(default_factory=list)
    prompt_logprobs: list[PackedLogprobs] = field(default_factory=list)
    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None


@dataclass
class RuntimeSessionRecord:
    session_id: str
    next_request_index: int = 0


@dataclass
class _DraftProposalJob:
    session_id: str
    proposal_id: int
    base_version: int
    accepted_prefix_token_ids: list[int]
    prompt_len: int
    sampling: SamplingMetadata
    generator: torch.Generator
    structured_output_session: Any | None
    future: asyncio.Future[DraftProposalOutput]


@dataclass
class _DraftProposalBatchState:
    job: _DraftProposalJob
    prefix_token_ids: list[int]
    max_steps: int
    stopped: bool
    generator_state: torch.Tensor
    draft_token_ids: list[int] = field(default_factory=list)
    draft_token_probs: list[float] = field(default_factory=list)
    structured_advances: int = 0


def _override_runtime_device(
    vllm_config: VllmConfig,
    device: str | None,
) -> VllmConfig:
    if device is None:
        return vllm_config
    device_config = copy.copy(vllm_config.device_config)
    device_obj = torch.device(device)
    device_config.device = device_obj
    device_config.device_type = device_obj.type
    return replace(vllm_config, device_config=device_config)


def _single_worker_runtime_config(vllm_config: VllmConfig) -> VllmConfig:
    parallel_config = copy.copy(vllm_config.parallel_config)
    parallel_config.tensor_parallel_size = 1
    parallel_config.pipeline_parallel_size = 1
    parallel_config.data_parallel_size = 1
    parallel_config.decode_context_parallel_size = 1
    parallel_config.prefill_context_parallel_size = 1
    parallel_config.distributed_executor_backend = "uni"
    parallel_config.rank = 0

    scheduler_config = copy.copy(vllm_config.scheduler_config)
    scheduler_config.max_num_seqs = 1
    scheduler_config.async_scheduling = False
    scheduler_config.max_num_batched_tokens = max(
        vllm_config.scheduler_config.max_num_batched_tokens,
        64,
    )
    scheduler_config.max_num_scheduled_tokens = max(
        vllm_config.scheduler_config.max_num_scheduled_tokens
        or vllm_config.scheduler_config.max_num_batched_tokens,
        64,
    )
    return replace(
        vllm_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        speculative_config=None,
    )


def build_target_runtime_config(
    model_name: str,
    *,
    device: str | None,
    dtype: str,
    trust_remote_code: bool,
    gpu_memory_utilization: float = 0.3,
) -> VllmConfig:
    engine_args = EngineArgs(
        model=model_name,
        tokenizer=model_name,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        distributed_executor_backend="uni",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        max_num_seqs=1,
        max_num_batched_tokens=64,
        disable_log_stats=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    runtime_config = engine_args.create_engine_config()
    scheduler_config = copy.copy(runtime_config.scheduler_config)
    scheduler_config.async_scheduling = False
    runtime_config = replace(runtime_config, speculative_config=None)
    runtime_config = replace(runtime_config, scheduler_config=scheduler_config)
    return _override_runtime_device(runtime_config, device)


class VllmEngineSessionRuntime:
    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        self.vocab_size = int(vllm_config.model_config.get_vocab_size())
        self._engine = EngineCore(
            vllm_config=vllm_config,
            executor_class=UniProcExecutor,
            log_stats=False,
        )
        self._sessions: dict[str, RuntimeSessionRecord] = {}
        self._lock = asyncio.Lock()

    async def open_session(
        self,
        session_id: str,
        prompt_token_ids: list[int],
        sampling: SamplingMetadata,
        *,
        prompt_logprobs: int | None = None,
    ) -> list[PackedLogprobs]:
        async with self._lock:
            return await asyncio.to_thread(
                self._open_session_sync,
                session_id,
                prompt_token_ids,
                sampling,
                prompt_logprobs,
            )

    async def run_chunk(
        self,
        session_id: str,
        prefix_token_ids: list[int],
        sampling: SamplingMetadata,
        *,
        max_tokens: int,
        logprobs: int | None = None,
        prompt_logprobs: int | None = None,
    ) -> EngineChunkResult:
        async with self._lock:
            return await asyncio.to_thread(
                self._run_chunk_sync,
                session_id,
                prefix_token_ids,
                sampling,
                max_tokens,
                logprobs,
                prompt_logprobs,
            )

    async def run_queries(
        self,
        queries: list[EngineQuery],
    ) -> list[EngineChunkResult]:
        if not queries:
            return []
        async with self._lock:
            return await asyncio.to_thread(self._run_queries_sync, queries)

    async def close_session(self, session_id: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._close_session_sync, session_id)

    def shutdown(self) -> None:
        self._sessions.clear()
        self._engine.shutdown()

    def _open_session_sync(
        self,
        session_id: str,
        prompt_token_ids: list[int],
        sampling: SamplingMetadata,
        prompt_logprobs: int | None,
    ) -> list[PackedLogprobs]:
        self._sessions.setdefault(
            session_id,
            RuntimeSessionRecord(session_id=session_id),
        )
        if prompt_logprobs is None or not prompt_token_ids:
            return []
        probe = self._run_chunk_sync(
            session_id,
            prompt_token_ids,
            sampling,
            max_tokens=1,
            logprobs=0,
            prompt_logprobs=prompt_logprobs,
        )
        return probe.prompt_logprobs

    def _close_session_sync(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _run_chunk_sync(
        self,
        session_id: str,
        prefix_token_ids: list[int],
        sampling: SamplingMetadata,
        max_tokens: int,
        logprobs: int | None,
        prompt_logprobs: int | None,
    ) -> EngineChunkResult:
        return self._run_queries_sync(
            [
                EngineQuery(
                    session_id=session_id,
                    prefix_token_ids=prefix_token_ids,
                    sampling=sampling,
                    max_tokens=max_tokens,
                    logprobs=logprobs,
                    prompt_logprobs=prompt_logprobs,
                )
            ]
        )[0]

    def _run_queries_sync(
        self,
        queries: list[EngineQuery],
    ) -> list[EngineChunkResult]:
        pending: dict[str, _PendingEngineQuery] = {}
        ordered_request_ids: list[str] = []

        for query in queries:
            if query.max_tokens <= 0:
                raise ValueError(
                    "max_tokens must be positive when running engine queries."
                )
            record = self._sessions.setdefault(
                query.session_id,
                RuntimeSessionRecord(session_id=query.session_id),
            )
            request_id = f"{query.session_id}:{record.next_request_index}"
            record.next_request_index += 1
            self._engine.add_request(
                _engine_request(
                    request_id,
                    query.prefix_token_ids,
                    query.sampling,
                    max_tokens=query.max_tokens,
                    logprobs=query.logprobs,
                    prompt_logprobs=query.prompt_logprobs,
                )
            )
            ordered_request_ids.append(request_id)
            pending[request_id] = _PendingEngineQuery(
                request_id=request_id,
                prefix_token_ids=list(query.prefix_token_ids),
            )

        remaining_request_ids = set(ordered_request_ids)
        while remaining_request_ids:
            outputs_by_client, model_executed = self._engine.step()
            self._engine.post_step(model_executed)
            outputs = outputs_by_client.get(0)
            if outputs is None:
                continue
            for output in outputs.outputs:
                query = pending.get(output.request_id)
                if query is None:
                    continue
                query.token_ids.extend(output.new_token_ids)
                if output.new_logprobs is not None:
                    query.logprobs.extend(
                        pack_logprobs_lists(
                            output.new_logprobs,
                            output.new_token_ids,
                        )
                    )
                if output.new_prompt_logprobs_tensors is not None:
                    query.prompt_logprobs = pack_logprobs_tensors(
                        output.new_prompt_logprobs_tensors,
                        query.prefix_token_ids[1:],
                    )
                query.finish_reason = output.finish_reason
                query.stop_reason = output.stop_reason
                if output.finish_reason is not None:
                    remaining_request_ids.discard(output.request_id)

        return [
            EngineChunkResult(
                token_ids=pending[request_id].token_ids,
                logprobs=pending[request_id].logprobs,
                prompt_logprobs=pending[request_id].prompt_logprobs,
                finish_reason=pending[request_id].finish_reason,
                stop_reason=pending[request_id].stop_reason,
            )
            for request_id in ordered_request_ids
        ]


class VllmEdgeDraftRunner:
    _DEFAULT_BATCH_WAIT_S = 0.001
    _DEFAULT_MAX_BATCH_SIZE = 8

    def __init__(self, vllm_config: "VllmConfig") -> None:
        from vllm.v1.spec_decode.utils import create_vllm_config_for_draft_model

        spec_config = vllm_config.speculative_config
        assert spec_config is not None
        runtime_config = create_vllm_config_for_draft_model(vllm_config)
        runtime_config = _single_worker_runtime_config(runtime_config)
        cache_config = copy.copy(runtime_config.cache_config)
        cache_config.gpu_memory_utilization = (
            spec_config.distributed_runtime_gpu_memory_utilization
        )
        runtime_config = replace(runtime_config, cache_config=cache_config)
        runtime_config = _override_runtime_device(
            runtime_config,
            spec_config.draft_device,
        )
        self._runtime = VllmEngineSessionRuntime(runtime_config)
        self.vocab_size = self._runtime.vocab_size
        self.num_speculative_tokens = spec_config.num_speculative_tokens
        self._queue: asyncio.Queue[_DraftProposalJob] = asyncio.Queue()
        self._pending_jobs: list[_DraftProposalJob] = []
        self._worker_task: asyncio.Task[None] | None = None
        self._closed = False

    def close_session(self, session_id: str) -> None:
        self._runtime._close_session_sync(session_id)

    def clear_sessions(self) -> None:
        sessions = getattr(self._runtime, "_sessions", None)
        if sessions is None:
            return
        for session_id in list(sessions.keys()):
            self._runtime._close_session_sync(session_id)

    def shutdown(self) -> None:
        self._closed = True
        if self._worker_task is not None:
            self._worker_task.cancel()
            self._worker_task = None
        self.clear_sessions()
        shutdown_runtime = getattr(self._runtime, "shutdown", None)
        if callable(shutdown_runtime):
            shutdown_runtime()

    async def propose(
        self,
        session_id: str,
        proposal_id: int,
        base_version: int,
        accepted_prefix_token_ids: list[int],
        prompt_len: int,
        sampling: SamplingMetadata,
        generator: torch.Generator,
        structured_output_session: Any | None = None,
    ) -> DraftProposalOutput:
        if self._closed:
            raise RuntimeError("VllmEdgeDraftRunner is closed.")
        future = asyncio.get_running_loop().create_future()
        await self._queue.put(
            _DraftProposalJob(
                session_id=session_id,
                proposal_id=proposal_id,
                base_version=base_version,
                accepted_prefix_token_ids=list(accepted_prefix_token_ids),
                prompt_len=prompt_len,
                sampling=sampling,
                generator=generator,
                structured_output_session=structured_output_session,
                future=future,
            )
        )
        self._ensure_worker()
        return await future

    def _ensure_worker(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(
                self._run_scheduler(),
                name="DistributedSpecDraftProposalScheduler",
            )

    async def _run_scheduler(self) -> None:
        try:
            while True:
                first_job = await self._next_job()
                jobs = await self._collect_batch(first_job)
                if not jobs:
                    continue
                try:
                    outputs = await self._process_batch(jobs)
                except Exception as exc:
                    for job in jobs:
                        if not job.future.done():
                            job.future.set_exception(exc)
                else:
                    for job, output in zip(jobs, outputs):
                        if not job.future.done():
                            job.future.set_result(output)
        except asyncio.CancelledError:
            for job in self._pending_jobs:
                if not job.future.done():
                    job.future.set_exception(
                        RuntimeError("Draft proposal scheduler was cancelled.")
                    )
            self._pending_jobs.clear()
            raise

    async def _next_job(self) -> _DraftProposalJob:
        if self._pending_jobs:
            return self._pending_jobs.pop(0)
        return await self._queue.get()

    async def _collect_batch(
        self,
        first_job: _DraftProposalJob,
    ) -> list[_DraftProposalJob]:
        batch = [first_job]
        seen_session_ids = {first_job.session_id}
        if self._DEFAULT_MAX_BATCH_SIZE == 1:
            return batch

        deadline = time.monotonic() + self._DEFAULT_BATCH_WAIT_S
        while len(batch) < self._DEFAULT_MAX_BATCH_SIZE:
            timeout_s = deadline - time.monotonic()
            if timeout_s <= 0:
                break
            try:
                job = await asyncio.wait_for(self._queue.get(), timeout=timeout_s)
            except asyncio.TimeoutError:
                break
            if job.session_id in seen_session_ids:
                self._pending_jobs.append(job)
                continue
            batch.append(job)
            seen_session_ids.add(job.session_id)

        while len(batch) < self._DEFAULT_MAX_BATCH_SIZE:
            try:
                job = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if job.session_id in seen_session_ids:
                self._pending_jobs.append(job)
                continue
            batch.append(job)
            seen_session_ids.add(job.session_id)
        return batch

    async def _process_batch(
        self,
        jobs: list[_DraftProposalJob],
    ) -> list[DraftProposalOutput]:
        remaining = None
        states: list[_DraftProposalBatchState] = []
        for job in jobs:
            current_output_len = len(job.accepted_prefix_token_ids) - job.prompt_len
            if job.sampling.max_tokens is not None:
                remaining = max(0, job.sampling.max_tokens - current_output_len)
            else:
                remaining = None
            max_steps = self.num_speculative_tokens
            if remaining is not None:
                max_steps = min(max_steps, remaining)
            states.append(
                _DraftProposalBatchState(
                    job=job,
                    prefix_token_ids=list(job.accepted_prefix_token_ids),
                    max_steps=max_steps,
                    stopped=max_steps < self.num_speculative_tokens,
                    generator_state=job.generator.get_state(),
                )
            )

        try:
            max_steps = max((state.max_steps for state in states), default=0)
            for _ in range(max_steps):
                active_states = [
                    state
                    for state in states
                    if not state.stopped and len(state.draft_token_ids) < state.max_steps
                ]
                if not active_states:
                    break
                chunks = await self._runtime.run_queries(
                    [
                        EngineQuery(
                            session_id=state.job.session_id,
                            prefix_token_ids=state.prefix_token_ids,
                            sampling=state.job.sampling,
                            max_tokens=1,
                            logprobs=-1,
                        )
                        for state in active_states
                    ]
                )
                for state, chunk in zip(active_states, chunks):
                    if len(chunk.logprobs) != 1:
                        raise ValueError(
                            "Draft runtime expected a single-token probability result."
                        )
                    logits = _logits_from_packed_logprobs(
                        chunk.logprobs[0],
                        self.vocab_size,
                    )
                    masked_logits = apply_structured_output_mask(
                        logits,
                        state.job.structured_output_session,
                    )
                    sample = sample_from_logits(
                        masked_logits,
                        state.job.sampling,
                        prompt_token_ids=state.prefix_token_ids[: state.job.prompt_len],
                        output_token_ids=state.prefix_token_ids[state.job.prompt_len :],
                        generator=state.job.generator,
                        suppress_stops=len(state.prefix_token_ids) - state.job.prompt_len
                        < state.job.sampling.min_tokens,
                    )
                    state.draft_token_ids.append(sample.token_id)
                    state.draft_token_probs.append(sample.token_prob)
                    state.prefix_token_ids.append(sample.token_id)
                    accept_structured_output_tokens(
                        state.job.structured_output_session,
                        [sample.token_id],
                    )
                    state.structured_advances += 1
                    if is_terminal_token(
                        sample.token_id,
                        state.job.sampling,
                        len(state.prefix_token_ids) - state.job.prompt_len,
                    ):
                        state.stopped = True

            return [
                self._build_proposal_output(state)
                for state in states
            ]
        except Exception:
            for state in states:
                state.job.generator.set_state(state.generator_state)
                rollback_structured_output_tokens(
                    state.job.structured_output_session,
                    state.structured_advances,
                )
                state.structured_advances = 0
            raise
        finally:
            for state in states:
                rollback_structured_output_tokens(
                    state.job.structured_output_session,
                    state.structured_advances,
                )
                state.structured_advances = 0

    def _build_proposal_output(
        self,
        state: _DraftProposalBatchState,
    ) -> DraftProposalOutput:
        from vllm.v1.spec_decode.distributed.protocol import DraftProposal

        return DraftProposalOutput(
            proposal=DraftProposal(
                session_id=state.job.session_id,
                proposal_id=state.job.proposal_id,
                base_version=state.job.base_version,
                accepted_prefix_len=len(state.job.accepted_prefix_token_ids),
                draft_token_ids=state.draft_token_ids,
                draft_token_probs=state.draft_token_probs,
                draft_stopped=state.stopped,
            ),
            stopped=state.stopped,
        )


@dataclass
class VllmCloudSession:
    session_id: str
    prompt_len: int
    accepted_prefix_token_ids: list[int]
    version: int
    sampling: SamplingMetadata
    generator: torch.Generator
    structured_output_session: Any | None = None


@dataclass
class _ProposalVerificationState:
    proposal: DraftProposal
    session: VllmCloudSession
    accepted_token_ids: list[int]
    accepted_logprobs: list[PackedLogprobs] | None
    generator_state: torch.Tensor
    committed_structured_tokens: int = 0
    final_prefix_token_ids: list[int] | None = None
    final_version: int | None = None
    result: VerificationResult | None = None

    def current_prefix_token_ids(self) -> list[int]:
        return list(self.session.accepted_prefix_token_ids) + self.accepted_token_ids


class VllmTargetVerificationRunner:
    def __init__(
        self,
        model_name: str,
        device: str | None,
        dtype: str,
        trust_remote_code: bool,
        gpu_memory_utilization: float = 0.3,
    ) -> None:
        runtime_config = build_target_runtime_config(
            model_name,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self._runtime = VllmEngineSessionRuntime(runtime_config)
        self.vocab_size = self._runtime.vocab_size
        self._sessions: dict[str, VllmCloudSession] = {}
        self._structured_output_factory = StructuredOutputFactory(
            model_name=model_name,
            trust_remote_code=trust_remote_code,
            vocab_size=self.vocab_size,
        )

    def shutdown(self) -> None:
        self._sessions.clear()
        self._structured_output_factory.close()
        self._runtime.shutdown()

    async def open_session(self, request: OpenSessionRequest) -> OpenSessionResponse:
        seed = request.sampling_metadata.seed
        if seed is None:
            seed = torch.seed()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + 1)
        prompt_logprobs = await self._runtime.open_session(
            request.session_id,
            request.prompt_token_ids,
            request.sampling_metadata,
            prompt_logprobs=request.sampling_metadata.prompt_logprobs,
        )
        self._sessions[request.session_id] = VllmCloudSession(
            session_id=request.session_id,
            prompt_len=len(request.prompt_token_ids),
            accepted_prefix_token_ids=list(request.prompt_token_ids),
            version=request.initial_version,
            sampling=request.sampling_metadata,
            generator=generator,
            structured_output_session=self._structured_output_factory.create_session(
                request.session_id,
                request.sampling_metadata,
            ),
        )
        from vllm.v1.spec_decode.distributed.protocol import OpenSessionResponse

        return OpenSessionResponse(
            session_id=request.session_id,
            session_version=request.initial_version,
            vocab_size=self.vocab_size,
            prompt_logprobs=prompt_logprobs,
        )

    async def close_session(self, request: CloseSessionRequest) -> None:
        self._sessions.pop(request.session_id, None)
        await self._runtime.close_session(request.session_id)

    async def resync_session(
        self,
        request: ResyncSessionRequest,
    ) -> ResyncSessionResponse:
        session = self._sessions.get(request.session_id)
        if session is None:
            sampling = request.sampling_metadata
            if sampling is None:
                raise VerifierSessionMissingError(
                    "Cannot recreate a missing verifier session without "
                    "sampling metadata."
                )
            seed = sampling.seed
            if seed is None:
                seed = torch.seed()
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int(seed) + 1)
            session = VllmCloudSession(
                session_id=request.session_id,
                prompt_len=request.prompt_len,
                accepted_prefix_token_ids=list(request.accepted_prefix_token_ids),
                version=request.edge_version,
                sampling=sampling,
                generator=generator,
                structured_output_session=self._structured_output_factory.create_session(
                    request.session_id,
                    sampling,
                ),
            )
            self._sessions[request.session_id] = session
        session.accepted_prefix_token_ids = list(request.accepted_prefix_token_ids)
        session.version = request.edge_version
        reset_structured_output_session(
            session.structured_output_session,
            session.accepted_prefix_token_ids[session.prompt_len :],
        )
        from vllm.v1.spec_decode.distributed.protocol import ResyncSessionResponse

        return ResyncSessionResponse(
            session_id=request.session_id,
            session_version=request.edge_version,
        )

    async def verify_proposal(
        self,
        proposal: DraftProposal,
    ) -> VerificationResult:
        return (await self.verify_proposals_batch([proposal]))[0]

    async def verify_proposals_batch(
        self,
        proposals: list[DraftProposal],
    ) -> list[VerificationResult]:
        if not proposals:
            return []

        states = [self._build_verification_state(proposal) for proposal in proposals]
        try:
            max_steps = max(len(state.proposal.draft_token_ids) for state in states)
            for step_idx in range(max_steps):
                active_states = [
                    state
                    for state in states
                    if state.result is None
                    and step_idx < len(state.proposal.draft_token_ids)
                ]
                if not active_states:
                    continue
                chunks = await self._runtime.run_queries(
                    [
                        EngineQuery(
                            session_id=state.proposal.session_id,
                            prefix_token_ids=state.current_prefix_token_ids(),
                            sampling=state.session.sampling,
                            max_tokens=1,
                            logprobs=-1,
                        )
                        for state in active_states
                    ]
                )
                for state, chunk in zip(active_states, chunks):
                    self._apply_verification_step(state, step_idx, chunk)

            bonus_states = [
                state
                for state in states
                if state.result is None
                and state.accepted_token_ids
                and not state.proposal.draft_stopped
            ]
            if bonus_states:
                chunks = await self._runtime.run_queries(
                    [
                        EngineQuery(
                            session_id=state.proposal.session_id,
                            prefix_token_ids=state.current_prefix_token_ids(),
                            sampling=state.session.sampling,
                            max_tokens=1,
                            logprobs=-1,
                        )
                        for state in bonus_states
                    ]
                )
                for state, chunk in zip(bonus_states, chunks):
                    self._apply_bonus_step(state, chunk)

            for state in states:
                if state.result is None:
                    self._finalize_accept_state(state)
                self._commit_state(state)
            return [state.result for state in states if state.result is not None]
        except Exception:
            for state in states:
                self._rollback_state(state)
            raise

    def _build_verification_state(
        self,
        proposal: DraftProposal,
    ) -> _ProposalVerificationState:
        session = self._sessions.get(proposal.session_id)
        if session is None:
            raise VerifierSessionMissingError(
                f"Unknown verifier session {proposal.session_id}."
            )
        if proposal.base_version != session.version:
            raise ValueError(
                f"Session {proposal.session_id} version mismatch: "
                f"got {proposal.base_version}, expected {session.version}."
            )
        if proposal.accepted_prefix_len != len(session.accepted_prefix_token_ids):
            raise ValueError(
                f"Session {proposal.session_id} prefix mismatch: "
                f"got {proposal.accepted_prefix_len}, "
                f"expected {len(session.accepted_prefix_token_ids)}."
            )
        return _ProposalVerificationState(
            proposal=proposal,
            session=session,
            accepted_token_ids=[],
            accepted_logprobs=(
                [] if session.sampling.logprobs is not None else None
            ),
            generator_state=session.generator.get_state(),
        )

    def _apply_verification_step(
        self,
        state: _ProposalVerificationState,
        reject_pos: int,
        chunk: EngineChunkResult,
    ) -> None:
        if len(chunk.logprobs) != 1:
            raise ValueError("Target verifier expected a single-token chunk result.")
        sample = self._sample_query_result(state, chunk)
        draft_token_id = state.proposal.draft_token_ids[reject_pos]
        if sample.token_id != draft_token_id:
            final_prefix = state.current_prefix_token_ids()
            final_version = state.proposal.base_version + len(state.accepted_token_ids)
            state.final_prefix_token_ids = final_prefix
            state.final_version = final_version
            state.result = VerificationResult(
                session_id=state.proposal.session_id,
                proposal_id=state.proposal.proposal_id,
                base_version=state.proposal.base_version,
                accepted_len=len(state.accepted_token_ids),
                accepted_token_ids=list(state.accepted_token_ids),
                reject_pos=reject_pos,
                target_probs_at_reject_pos=serialize_probs(sample.probs),
                verifier_version=final_version,
                accepted_logprobs=list(state.accepted_logprobs or []),
            )
            return

        state.accepted_token_ids.append(draft_token_id)
        accept_structured_output_tokens(
            state.session.structured_output_session,
            [draft_token_id],
        )
        state.committed_structured_tokens += 1
        if state.accepted_logprobs is not None:
            packed = pack_sample_logprobs(
                sample.probs,
                sample.token_id,
                state.session.sampling.logprobs,
            )
            assert packed is not None
            state.accepted_logprobs.append(packed)

    def _apply_bonus_step(
        self,
        state: _ProposalVerificationState,
        chunk: EngineChunkResult,
    ) -> None:
        if len(chunk.logprobs) != 1:
            raise ValueError("Target verifier expected a single-token bonus result.")
        sample = self._sample_query_result(state, chunk)
        bonus_token_id = sample.token_id
        accept_structured_output_tokens(
            state.session.structured_output_session,
            [bonus_token_id],
        )
        state.committed_structured_tokens += 1
        bonus_logprobs = None
        if state.session.sampling.logprobs is not None:
            bonus_logprobs = pack_sample_logprobs(
                sample.probs,
                bonus_token_id,
                state.session.sampling.logprobs,
            )
        final_prefix = state.current_prefix_token_ids() + [bonus_token_id]
        final_version = state.proposal.base_version + len(state.accepted_token_ids) + 1
        state.final_prefix_token_ids = final_prefix
        state.final_version = final_version
        state.result = VerificationResult(
            session_id=state.proposal.session_id,
            proposal_id=state.proposal.proposal_id,
            base_version=state.proposal.base_version,
            accepted_len=len(state.accepted_token_ids),
            accepted_token_ids=list(state.accepted_token_ids),
            verifier_version=final_version,
            bonus_token_id=bonus_token_id,
            accepted_logprobs=list(state.accepted_logprobs or []),
            bonus_logprobs=bonus_logprobs,
        )

    def _finalize_accept_state(self, state: _ProposalVerificationState) -> None:
        final_prefix = state.current_prefix_token_ids()
        final_version = state.proposal.base_version + len(state.accepted_token_ids)
        state.final_prefix_token_ids = final_prefix
        state.final_version = final_version
        state.result = VerificationResult(
            session_id=state.proposal.session_id,
            proposal_id=state.proposal.proposal_id,
            base_version=state.proposal.base_version,
            accepted_len=len(state.accepted_token_ids),
            accepted_token_ids=list(state.accepted_token_ids),
            verifier_version=final_version,
            accepted_logprobs=list(state.accepted_logprobs or []),
        )

    def _commit_state(self, state: _ProposalVerificationState) -> None:
        assert state.final_prefix_token_ids is not None
        assert state.final_version is not None
        state.session.accepted_prefix_token_ids = state.final_prefix_token_ids
        state.session.version = state.final_version
        state.committed_structured_tokens = 0

    def _rollback_state(self, state: _ProposalVerificationState) -> None:
        state.session.generator.set_state(state.generator_state)
        rollback_structured_output_tokens(
            state.session.structured_output_session,
            state.committed_structured_tokens,
        )
        state.committed_structured_tokens = 0

    def _sample_query_result(
        self,
        state: _ProposalVerificationState,
        chunk: EngineChunkResult,
    ):
        logits = _logits_from_packed_logprobs(chunk.logprobs[0], self.vocab_size)
        masked_logits = apply_structured_output_mask(
            logits,
            state.session.structured_output_session,
        )
        prefix_token_ids = state.current_prefix_token_ids()
        return sample_from_logits(
            masked_logits,
            state.session.sampling,
            prompt_token_ids=prefix_token_ids[: state.session.prompt_len],
            output_token_ids=prefix_token_ids[state.session.prompt_len :],
            generator=state.session.generator,
            suppress_stops=len(prefix_token_ids) - state.session.prompt_len
            < state.session.sampling.min_tokens,
        )
