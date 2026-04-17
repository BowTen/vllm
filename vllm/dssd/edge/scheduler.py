from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.request import Request

from .types import EdgeSession

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass(slots=True)
class _EdgeDecodeRequestView:
    request_id: str
    num_tokens: int
    num_computed_tokens: int


class EdgeSchedulerAdapter:
    """Adapter for the first-pass single-request, decoder-only text path.

    This edge path does not handle multimodal or encoder inputs.
    """

    def __init__(
        self,
        kv_cache_manager: KVCacheManager | None = None,
        request_block_hasher: Callable[[Request], list[BlockHash]] | None = None,
    ) -> None:
        self.kv_cache_manager = kv_cache_manager
        self.request_block_hasher = request_block_hasher
        self._pending_prefill_new_block_ids_to_zero: dict[str, list[int] | None] = {}

    def allocate_blocks(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        lora_request: LoRARequest | None = None,
    ) -> tuple[list[int], ...]:
        if self.kv_cache_manager is None:
            return ([],)

        request = self._build_prefill_request(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        num_new_tokens = request.num_tokens - request.num_computed_tokens
        new_blocks = self.kv_cache_manager.allocate_slots(
            request=request,
            num_new_tokens=num_new_tokens,
        )
        if new_blocks is None:
            raise RuntimeError("prefill failed to allocate KV blocks for edge")
        self._pending_prefill_new_block_ids_to_zero[req_id] = (
            self._take_new_block_ids_to_zero()
        )
        return new_blocks.get_block_ids(allow_none=False)

    def build_prefill_step(self, session: EdgeSession) -> SchedulerOutput:
        prompt_len = len(session.prompt_token_ids)
        return SchedulerOutput(
            scheduled_new_reqs=[self.make_new_request(session)],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={session.req_id: prompt_len},
            total_num_scheduled_tokens=prompt_len,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=self._make_num_common_prefix_blocks(),
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            new_block_ids_to_zero=self._take_prefill_new_block_ids_to_zero(
                session.req_id
            ),
        )

    def build_decode_step(self, session: EdgeSession) -> SchedulerOutput:
        return SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=self.make_cached_request(
                session,
                self._allocate_decode_blocks(session),
            ),
            num_scheduled_tokens={session.req_id: 1},
            total_num_scheduled_tokens=1,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=self._make_num_common_prefix_blocks(),
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            new_block_ids_to_zero=self._take_new_block_ids_to_zero(),
        )

    def build_close_step(self, req_id: str) -> SchedulerOutput:
        return SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=self._make_num_common_prefix_blocks(),
            finished_req_ids={req_id},
            free_encoder_mm_hashes=[],
        )

    def free_blocks(self, session: EdgeSession) -> None:
        self._pending_prefill_new_block_ids_to_zero.pop(session.req_id, None)
        if self.kv_cache_manager is None:
            return
        self.kv_cache_manager.free(self._build_close_request(session))

    def make_new_request(self, session: EdgeSession) -> NewRequestData:
        return NewRequestData(
            req_id=session.req_id,
            prompt_token_ids=session.prompt_token_ids,
            prefill_token_ids=session.prompt_token_ids,
            mm_features=[],
            sampling_params=session.sampling_params,
            pooling_params=None,
            block_ids=session.block_ids,
            num_computed_tokens=0,
            lora_request=session.lora_request,
        )

    def make_cached_request(
        self,
        session: EdgeSession,
        new_block_ids: tuple[list[int], ...] | None = None,
    ) -> CachedRequestData:
        return CachedRequestData(
            req_ids=[session.req_id],
            resumed_req_ids=set(),
            new_token_ids=[[]],
            all_token_ids={},
            new_block_ids=[new_block_ids],
            num_computed_tokens=[session.num_computed_tokens],
            num_output_tokens=[session.output_len],
        )

    def _build_prefill_request(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        lora_request: LoRARequest | None = None,
    ) -> Request:
        request = Request(
            request_id=req_id,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=sampling_params,
            pooling_params=None,
            lora_request=lora_request,
            block_hasher=self.request_block_hasher,
        )
        request.num_computed_tokens = 0
        return request

    def _build_decode_request(self, session: EdgeSession) -> Request:
        request = Request(
            request_id=session.req_id,
            prompt_token_ids=list(session.prompt_token_ids),
            sampling_params=session.sampling_params,
            pooling_params=None,
            lora_request=session.lora_request,
            block_hasher=self.request_block_hasher,
        )
        output_token_ids = session.token_ids[session.prompt_len:]
        if output_token_ids:
            request.append_output_token_ids(output_token_ids)
        request.num_computed_tokens = session.num_computed_tokens
        return request

    def _build_close_request(self, session: EdgeSession) -> Request:
        return self._build_decode_request(session)

    def _allocate_decode_blocks(
        self,
        session: EdgeSession,
    ) -> tuple[list[int], ...] | None:
        if self.kv_cache_manager is None:
            return None

        request = self._make_decode_allocation_request(session)
        # Edge decode always schedules a single query token. By the time we
        # reach this path, `request.num_tokens` already reflects the committed
        # prefix, so `request.num_tokens - request.num_computed_tokens` can be
        # zero for a normal decode step. We still need one new KV slot for the
        # next token to be computed.
        num_new_tokens = 1
        new_blocks = self.kv_cache_manager.allocate_slots(
            request=request,
            num_new_tokens=num_new_tokens,
        )
        if new_blocks is None:
            raise RuntimeError("decode failed to allocate KV blocks for edge")
        return new_blocks.get_block_ids(allow_none=True)

    def _make_decode_allocation_request(
        self,
        session: EdgeSession,
    ) -> Request | _EdgeDecodeRequestView:
        kv_cache_manager = self.kv_cache_manager
        if kv_cache_manager is not None and not kv_cache_manager.enable_caching:
            return _EdgeDecodeRequestView(
                request_id=session.req_id,
                num_tokens=session.total_len,
                num_computed_tokens=session.num_computed_tokens,
            )
        return self._build_decode_request(session)

    def _take_new_block_ids_to_zero(self) -> list[int] | None:
        if self.kv_cache_manager is None:
            return None
        return self.kv_cache_manager.take_new_block_ids() or None

    def _take_prefill_new_block_ids_to_zero(self, req_id: str) -> list[int] | None:
        return self._pending_prefill_new_block_ids_to_zero.pop(req_id, None)

    def _make_num_common_prefix_blocks(self) -> list[int]:
        if self.kv_cache_manager is None:
            return [0]

        return [0] * max(len(self.kv_cache_manager.kv_cache_config.kv_cache_groups), 1)
