from __future__ import annotations

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.request import Request

from .types import VerifierRoundRequest, VerifierSession


class VerifierSchedulerAdapter:
    """把 DSSD verifier 的一轮执行翻译成 vLLM 的 SchedulerOutput。

    这里不复刻完整 Scheduler，只保留最小映射关系，方便阅读：
    - open_session 对应一次 new request prefill
    - verify_round 对应一次 cached request + scheduled_spec_decode_tokens
    - close_session 对应 finished_req_ids
    """

    def __init__(self, kv_cache_manager: KVCacheManager | None = None) -> None:
        self.kv_cache_manager = kv_cache_manager

    def allocate_blocks(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        lora_request: LoRARequest | None = None,
    ) -> tuple[list[int], ...]:
        if self.kv_cache_manager is None:
            return ([],)

        built_request = self._build_prefill_request(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        num_new_tokens = built_request.num_tokens - built_request.num_computed_tokens
        new_blocks = self.kv_cache_manager.allocate_slots(
            request=built_request,
            num_new_tokens=num_new_tokens,
        )
        if new_blocks is None:
            raise RuntimeError("prefill 无法为 verifier 申请新的 KV blocks")
        return new_blocks.get_block_ids(allow_none=True)

    def build_open_session_step(
        self,
        session: VerifierSession,
    ) -> SchedulerOutput:
        prompt_len = len(session.prompt_token_ids)
        return SchedulerOutput(
            scheduled_new_reqs=[self.make_new_request(session)],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={session.req_id: prompt_len},
            total_num_scheduled_tokens=prompt_len,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            new_block_ids_to_zero=self._take_new_block_ids_to_zero(),
        )

    def build_verify_step(
        self,
        session: VerifierSession,
        request: VerifierRoundRequest,
    ) -> SchedulerOutput:
        query_len = 1 + len(request.draft_token_ids)
        new_block_ids = self._allocate_verify_blocks(session, request)
        return SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=self.make_cached_request(session, new_block_ids),
            num_scheduled_tokens={session.req_id: query_len},
            total_num_scheduled_tokens=query_len,
            scheduled_spec_decode_tokens={session.req_id: request.draft_token_ids},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
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
            num_common_prefix_blocks=[],
            finished_req_ids={req_id},
            free_encoder_mm_hashes=[],
        )

    def free_blocks(self, session: VerifierSession) -> None:
        """按真实 vLLM close 语义释放整个 request 的 KV blocks。"""
        if self.kv_cache_manager is None:
            return
        self.kv_cache_manager.free(self._build_close_request(session))

    def make_new_request(self, session: VerifierSession) -> NewRequestData:
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
        session: VerifierSession,
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

    def _build_verify_request(
        self,
        session: VerifierSession,
        request: VerifierRoundRequest,
    ) -> Request:
        built_request = Request(
            request_id=session.req_id,
            prompt_token_ids=list(session.prompt_token_ids),
            sampling_params=session.sampling_params,
            pooling_params=None,
            lora_request=session.lora_request,
        )
        finalized_output_ids = session.token_ids[session.prompt_len :]
        if finalized_output_ids:
            built_request.append_output_token_ids(finalized_output_ids)
        built_request.append_output_token_ids(request.committed_token_id)
        built_request.spec_token_ids = list(request.draft_token_ids)
        built_request.num_computed_tokens = session.num_computed_tokens
        return built_request

    def _build_prefill_request(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        lora_request: LoRARequest | None = None,
    ) -> Request:
        built_request = Request(
            request_id=req_id,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=sampling_params,
            pooling_params=None,
            lora_request=lora_request,
        )
        built_request.num_computed_tokens = 0
        return built_request

    def _build_close_request(self, session: VerifierSession) -> Request:
        built_request = Request(
            request_id=session.req_id,
            prompt_token_ids=list(session.prompt_token_ids),
            sampling_params=session.sampling_params,
            pooling_params=None,
            lora_request=session.lora_request,
        )
        finalized_output_ids = session.token_ids[session.prompt_len :]
        if finalized_output_ids:
            built_request.append_output_token_ids(finalized_output_ids)
        built_request.num_computed_tokens = session.num_computed_tokens
        return built_request

    def _allocate_verify_blocks(
        self,
        session: VerifierSession,
        request: VerifierRoundRequest,
    ) -> tuple[list[int], ...] | None:
        if self.kv_cache_manager is None:
            return None

        built_request = self._build_verify_request(session, request)
        num_new_tokens = (
            built_request.num_tokens_with_spec - built_request.num_computed_tokens
        )
        new_blocks = self.kv_cache_manager.allocate_slots(
            request=built_request,
            num_new_tokens=num_new_tokens,
        )
        if new_blocks is None:
            raise RuntimeError("verify_round 无法为 verifier 申请新的 KV blocks")
        return new_blocks.get_block_ids(allow_none=True)

    def _take_new_block_ids_to_zero(self) -> list[int] | None:
        if self.kv_cache_manager is None:
            return None
        return self.kv_cache_manager.take_new_block_ids() or None
