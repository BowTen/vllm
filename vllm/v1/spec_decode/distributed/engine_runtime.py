# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import copy
import json
import time
from dataclasses import dataclass, replace
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
class RuntimeSessionRecord:
    session_id: str
    next_request_index: int = 0


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
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive when running a chunk.")
        record = self._sessions.setdefault(
            session_id,
            RuntimeSessionRecord(session_id=session_id),
        )
        request_id = f"{session_id}:{record.next_request_index}"
        record.next_request_index += 1
        self._engine.add_request(
            _engine_request(
                request_id,
                prefix_token_ids,
                sampling,
                max_tokens=max_tokens,
                logprobs=logprobs,
                prompt_logprobs=prompt_logprobs,
            )
        )

        token_ids: list[int] = []
        packed_logprobs: list[PackedLogprobs] = []
        packed_prompt_logprobs: list[PackedLogprobs] = []
        finish_reason = None
        stop_reason = None

        while finish_reason is None:
            outputs_by_client, model_executed = self._engine.step()
            self._engine.post_step(model_executed)
            outputs = outputs_by_client.get(0)
            if outputs is None:
                continue
            for output in outputs.outputs:
                if output.request_id != request_id:
                    continue
                token_ids.extend(output.new_token_ids)
                if output.new_logprobs is not None:
                    packed_logprobs.extend(
                        pack_logprobs_lists(
                            output.new_logprobs,
                            output.new_token_ids,
                        )
                    )
                if output.new_prompt_logprobs_tensors is not None:
                    packed_prompt_logprobs = pack_logprobs_tensors(
                        output.new_prompt_logprobs_tensors,
                        prefix_token_ids[1:],
                    )
                finish_reason = output.finish_reason
                stop_reason = output.stop_reason
                if finish_reason is not None:
                    break

        return EngineChunkResult(
            token_ids=token_ids,
            logprobs=packed_logprobs,
            prompt_logprobs=packed_prompt_logprobs,
            finish_reason=finish_reason,
            stop_reason=stop_reason,
        )


class VllmEdgeDraftRunner:
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

    def close_session(self, session_id: str) -> None:
        self._runtime._close_session_sync(session_id)

    def clear_sessions(self) -> None:
        for session_id in list(self._runtime._sessions.keys()):
            self._runtime._close_session_sync(session_id)

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
        remaining = None
        current_output_len = len(accepted_prefix_token_ids) - prompt_len
        if sampling.max_tokens is not None:
            remaining = max(0, sampling.max_tokens - current_output_len)
        max_steps = self.num_speculative_tokens
        if remaining is not None:
            max_steps = min(max_steps, remaining)
        if max_steps <= 0:
            from vllm.v1.spec_decode.distributed.protocol import DraftProposal

            return DraftProposalOutput(
                proposal=DraftProposal(
                    session_id=session_id,
                    proposal_id=proposal_id,
                    base_version=base_version,
                    accepted_prefix_len=len(accepted_prefix_token_ids),
                    draft_token_ids=[],
                    draft_token_probs=[],
                    draft_stopped=True,
                ),
                stopped=True,
            )
        draft_token_ids: list[int] = []
        draft_token_probs: list[float] = []
        prefix_token_ids = list(accepted_prefix_token_ids)
        structured_advances = 0
        stopped = max_steps < self.num_speculative_tokens
        try:
            for _ in range(max_steps):
                chunk = await self._runtime.run_chunk(
                    session_id,
                    prefix_token_ids,
                    sampling,
                    max_tokens=1,
                    logprobs=-1,
                )
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
                    structured_output_session,
                )
                sample = sample_from_logits(
                    masked_logits,
                    sampling,
                    prompt_token_ids=prefix_token_ids[:prompt_len],
                    output_token_ids=prefix_token_ids[prompt_len:],
                    generator=generator,
                    suppress_stops=len(prefix_token_ids) - prompt_len
                    < sampling.min_tokens,
                )
                draft_token_ids.append(sample.token_id)
                draft_token_probs.append(sample.token_prob)
                prefix_token_ids.append(sample.token_id)
                accept_structured_output_tokens(
                    structured_output_session,
                    [sample.token_id],
                )
                structured_advances += 1
                if is_terminal_token(
                    sample.token_id,
                    sampling,
                    len(prefix_token_ids) - prompt_len,
                ):
                    stopped = True
                    break
        finally:
            rollback_structured_output_tokens(
                structured_output_session,
                structured_advances,
            )

        from vllm.v1.spec_decode.distributed.protocol import DraftProposal

        return DraftProposalOutput(
            proposal=DraftProposal(
                session_id=session_id,
                proposal_id=proposal_id,
                base_version=base_version,
                accepted_prefix_len=len(accepted_prefix_token_ids),
                draft_token_ids=draft_token_ids,
                draft_token_probs=draft_token_probs,
                draft_stopped=stopped,
            ),
            stopped=stopped,
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
        from vllm.v1.spec_decode.distributed.protocol import VerificationResult

        session = self._sessions[proposal.session_id]
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

        accepted_token_ids: list[int] = []
        accepted_logprobs = [] if session.sampling.logprobs is not None else None
        prefix = list(session.accepted_prefix_token_ids)

        for reject_pos, draft_token_id in enumerate(proposal.draft_token_ids):
            chunk = await self._runtime.run_chunk(
                proposal.session_id,
                prefix + accepted_token_ids,
                session.sampling,
                max_tokens=1,
                logprobs=-1,
            )
            if len(chunk.logprobs) != 1:
                raise ValueError(
                    "Target verifier expected a single-token chunk result."
                )
            logits = _logits_from_packed_logprobs(
                chunk.logprobs[0],
                self.vocab_size,
            )
            masked_logits = apply_structured_output_mask(
                logits,
                session.structured_output_session,
            )
            sample = sample_from_logits(
                masked_logits,
                session.sampling,
                prompt_token_ids=prefix[: session.prompt_len],
                output_token_ids=(prefix + accepted_token_ids)[session.prompt_len :],
                generator=session.generator,
                suppress_stops=len(prefix) + len(accepted_token_ids)
                - session.prompt_len < session.sampling.min_tokens,
            )
            if sample.token_id != draft_token_id:
                session.accepted_prefix_token_ids = prefix + accepted_token_ids
                session.version = proposal.base_version + len(accepted_token_ids)
                return VerificationResult(
                    session_id=proposal.session_id,
                    proposal_id=proposal.proposal_id,
                    base_version=proposal.base_version,
                    accepted_len=len(accepted_token_ids),
                    accepted_token_ids=accepted_token_ids,
                    reject_pos=reject_pos,
                    target_probs_at_reject_pos=serialize_probs(sample.probs),
                    verifier_version=session.version,
                    accepted_logprobs=accepted_logprobs or [],
                )
            accepted_token_ids.append(draft_token_id)
            accept_structured_output_tokens(
                session.structured_output_session,
                [draft_token_id],
            )
            if accepted_logprobs is not None:
                packed = pack_sample_logprobs(
                    sample.probs,
                    sample.token_id,
                    session.sampling.logprobs,
                )
                assert packed is not None
                accepted_logprobs.append(packed)

        session.accepted_prefix_token_ids = prefix + accepted_token_ids
        session.version = proposal.base_version + len(accepted_token_ids)

        bonus_token_id = None
        bonus_logprobs = None
        if accepted_token_ids and not proposal.draft_stopped:
            chunk = await self._runtime.run_chunk(
                proposal.session_id,
                session.accepted_prefix_token_ids,
                session.sampling,
                max_tokens=1,
                logprobs=-1,
            )
            if len(chunk.logprobs) != 1:
                raise ValueError(
                    "Target verifier expected a single-token bonus result."
                )
            logits = _logits_from_packed_logprobs(
                chunk.logprobs[0],
                self.vocab_size,
            )
            masked_logits = apply_structured_output_mask(
                logits,
                session.structured_output_session,
            )
            sample = sample_from_logits(
                masked_logits,
                session.sampling,
                prompt_token_ids=session.accepted_prefix_token_ids[: session.prompt_len],
                output_token_ids=session.accepted_prefix_token_ids[session.prompt_len :],
                generator=session.generator,
                suppress_stops=len(session.accepted_prefix_token_ids)
                - session.prompt_len < session.sampling.min_tokens,
            )
            bonus_token_id = sample.token_id
            if session.sampling.logprobs is not None:
                bonus_logprobs = pack_sample_logprobs(
                    sample.probs,
                    bonus_token_id,
                    session.sampling.logprobs,
                )
            session.accepted_prefix_token_ids.append(bonus_token_id)
            accept_structured_output_tokens(
                session.structured_output_session,
                [bonus_token_id],
            )
            session.version += 1

        return VerificationResult(
            session_id=proposal.session_id,
            proposal_id=proposal.proposal_id,
            base_version=proposal.base_version,
            accepted_len=len(accepted_token_ids),
            accepted_token_ids=accepted_token_ids,
            verifier_version=session.version,
            bonus_token_id=bonus_token_id,
            accepted_logprobs=accepted_logprobs or [],
            bonus_logprobs=bonus_logprobs,
        )

    async def verify_proposals_batch(
        self,
        proposals: list[DraftProposal],
    ) -> list[VerificationResult]:
        return [await self.verify_proposal(proposal) for proposal in proposals]
