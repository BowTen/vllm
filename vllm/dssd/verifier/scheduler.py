from __future__ import annotations

from collections.abc import Callable

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.request import Request

from .types import VerifierRoundRequest, VerifierSession


class VerifierSchedulerAdapter:
    def __init__(
        self,
        kv_cache_manager: KVCacheManager | None = None,
        request_block_hasher: Callable | None = None,
    ) -> None:
        self.kv_cache_manager = kv_cache_manager
        self.request_block_hasher = request_block_hasher
        self._pending_new_block_ids_to_zero: list[int] | None = None

    def allocate_blocks(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        lora_request: LoRARequest | None = None,
    ) -> tuple[list[int], ...]:
        if self.kv_cache_manager is None:
            self._pending_new_block_ids_to_zero = None
            return ([],)

        request = Request(
            request_id=req_id,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=sampling_params,
            pooling_params=None,
            lora_request=lora_request,
            block_hasher=self.request_block_hasher,
        )
        blocks = self.kv_cache_manager.allocate_slots(
            request=request,
            num_new_tokens=request.num_tokens - request.num_computed_tokens,
        )
        if blocks is None:
            raise RuntimeError("prefill cannot allocate KV blocks for verifier")

        self._pending_new_block_ids_to_zero = self._drain_new_block_ids_to_zero(
            blocks
        )
        block_ids = blocks.get_block_ids(allow_none=True)
        return ([],) if block_ids is None else block_ids

    def build_open_session_step(self, session: VerifierSession) -> SchedulerOutput:
        prompt_len = len(session.prompt_token_ids)
        return SchedulerOutput(
            scheduled_new_reqs=[
                NewRequestData(
                    req_id=session.req_id,
                    prompt_token_ids=list(session.prompt_token_ids),
                    mm_features=[],
                    sampling_params=session.sampling_params,
                    pooling_params=None,
                    block_ids=session.block_ids,
                    num_computed_tokens=0,
                    lora_request=session.lora_request,
                    prefill_token_ids=list(session.prompt_token_ids),
                )
            ],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={session.req_id: prompt_len},
            total_num_scheduled_tokens=prompt_len,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            new_block_ids_to_zero=self._take_pending_new_block_ids_to_zero(),
        )

    def build_verify_step(
        self,
        session: VerifierSession,
        request: VerifierRoundRequest,
        *,
        use_spec_decode: bool = True,
    ) -> SchedulerOutput:
        self._validate_req_id(session.req_id, request.req_id)
        query_len = 1 + len(request.draft_token_ids)
        new_block_ids_to_zero: list[int] | None = None
        new_block_ids: list[tuple[list[int], ...] | None] = [None]
        if self.kv_cache_manager is not None:
            request_view = self._make_request(session)
            blocks = self.kv_cache_manager.allocate_slots(
                request=request_view,
                num_new_tokens=query_len,
            )
            if blocks is None:
                raise RuntimeError("verify round cannot allocate KV blocks")
            new_block_ids = [blocks.get_block_ids(allow_none=True)]
            new_block_ids_to_zero = self._drain_new_block_ids_to_zero(blocks)
        scheduled_spec_decode_tokens = (
            {session.req_id: list(request.draft_token_ids)}
            if use_spec_decode and request.draft_token_ids
            else {}
        )
        return SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData(
                req_ids=[session.req_id],
                resumed_req_ids=set(),
                new_token_ids=[[]],
                all_token_ids={},
                new_block_ids=new_block_ids,
                num_computed_tokens=[session.num_computed_tokens],
                num_output_tokens=[session.output_len],
            ),
            num_scheduled_tokens={session.req_id: query_len},
            total_num_scheduled_tokens=query_len,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            new_block_ids_to_zero=new_block_ids_to_zero,
        )

    def build_decode_step(self, session: VerifierSession) -> SchedulerOutput:
        query_len = 1
        new_block_ids_to_zero: list[int] | None = None
        new_block_ids: list[tuple[list[int], ...] | None] = [None]
        if self.kv_cache_manager is not None:
            request_view = self._make_request(session)
            blocks = self.kv_cache_manager.allocate_slots(
                request=request_view,
                num_new_tokens=query_len,
            )
            if blocks is None:
                raise RuntimeError("local decode cannot allocate KV blocks")
            new_block_ids = [blocks.get_block_ids(allow_none=True)]
            new_block_ids_to_zero = self._drain_new_block_ids_to_zero(blocks)
        return SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData(
                req_ids=[session.req_id],
                resumed_req_ids=set(),
                new_token_ids=[[]],
                all_token_ids={},
                new_block_ids=new_block_ids,
                num_computed_tokens=[session.num_computed_tokens],
                num_output_tokens=[session.output_len],
            ),
            num_scheduled_tokens={session.req_id: query_len},
            total_num_scheduled_tokens=query_len,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            new_block_ids_to_zero=new_block_ids_to_zero,
        )

    def build_close_step(self, req_id: str) -> SchedulerOutput:
        return SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids={req_id},
            free_encoder_mm_hashes=[],
        )

    def free_blocks(self, session: VerifierSession) -> None:
        if self.kv_cache_manager is None:
            return

        request = self._make_request(session)
        self.kv_cache_manager.free(request)

    @staticmethod
    def _validate_req_id(session_req_id: str, request_req_id: str) -> None:
        if session_req_id != request_req_id:
            raise ValueError("session and request req_id must match")

    @staticmethod
    def _new_block_ids_to_zero(blocks: KVCacheBlocks) -> list[int] | None:
        block_groups = blocks.get_unhashed_block_ids_all_groups()
        flat_ids = [block_id for group in block_groups for block_id in group]
        return flat_ids or None

    def _drain_new_block_ids_to_zero(
        self, blocks: KVCacheBlocks
    ) -> list[int] | None:
        if self.kv_cache_manager is None:
            return self._new_block_ids_to_zero(blocks)
        drained_ids = self.kv_cache_manager.take_new_block_ids()
        return drained_ids or self._new_block_ids_to_zero(blocks)

    def _make_request(self, session: VerifierSession) -> Request:
        request = Request(
            request_id=session.req_id,
            prompt_token_ids=list(session.prompt_token_ids),
            sampling_params=session.sampling_params,
            pooling_params=None,
            lora_request=session.lora_request,
            block_hasher=self.request_block_hasher,
        )
        finalized_output = session.token_ids[session.prompt_len :]
        if finalized_output:
            request.append_output_token_ids(finalized_output)
        request.num_computed_tokens = session.num_computed_tokens
        return request

    def _take_pending_new_block_ids_to_zero(self) -> list[int] | None:
        block_ids = self._pending_new_block_ids_to_zero
        self._pending_new_block_ids_to_zero = None
        return block_ids
