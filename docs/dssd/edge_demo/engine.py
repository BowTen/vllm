from __future__ import annotations

from typing import cast

import torch

from vllm.config import VllmConfig
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker as GPUWorker

from .sampler import DSSDEdgeDraftSampler
from .scheduler import EdgeSchedulerAdapter
from .state_bridge import EdgeStateBridge
from .types import EdgeSession


class EdgeDecodeEngine:
    """真正绑定 vLLM 执行流的 edge 端执行层。"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        worker: GPUWorker,
        scheduler: EdgeSchedulerAdapter,
        state_bridge: EdgeStateBridge,
        draft_sampler: DSSDEdgeDraftSampler,
    ) -> None:
        self.vllm_config = vllm_config
        self.worker = worker
        self.scheduler = scheduler
        self.state_bridge = state_bridge
        self.draft_sampler = draft_sampler
        self.sessions: dict[str, EdgeSession] = {}

    @property
    def model_runner(self) -> GPUModelRunner:
        return self.worker.model_runner

    def init_device(self) -> None:
        self.worker.init_device()

    def load_model(self) -> None:
        self.worker.load_model()

    def init_kv_cache(self) -> None:
        raise NotImplementedError("demo 不负责构造 KVCacheConfig")

    def open_session(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        lora_request: LoRARequest | None = None,
    ) -> EdgeSession:
        session = EdgeSession(
            req_id=req_id,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=sampling_params,
            block_ids=self.scheduler.allocate_blocks(
                req_id=req_id,
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
                lora_request=lora_request,
            ),
            prompt_len=len(prompt_token_ids),
            total_len=len(prompt_token_ids),
            token_ids=list(prompt_token_ids),
            lora_request=lora_request,
        )
        self.sessions[req_id] = session
        return session

    def prefill(
        self,
        session: EdgeSession,
        bootstrap_token_id: int,
    ) -> int:
        scheduler_output = self.scheduler.build_prefill_step(session)
        self._execute(scheduler_output)
        session.num_computed_tokens = session.prompt_len
        session.total_len = session.prompt_len
        return self.state_bridge.bootstrap_first_token(
            session,
            bootstrap_token_id,
            self.model_runner,
        )

    def decode_one(
        self,
        session: EdgeSession,
        input_token_id: int,
        processed_logits_dst: torch.Tensor,
    ) -> tuple[int, float]:
        self.state_bridge.prepare_next_decode(
            session,
            input_token_id,
            self.model_runner,
        )
        scheduler_output = self.scheduler.build_decode_step(session)
        self._execute(scheduler_output)
        return self._sample_with_draft_sampler(session, processed_logits_dst)

    def draft(
        self,
        session: EdgeSession,
        first_token_id: int,
        gamma: int,
    ):
        self.state_bridge.clear_round_state(session)
        next_input_token_id = int(first_token_id)
        session.round_state.committed_token_id = int(first_token_id)
        session.round_state.prepare_logits_buffer(
            gamma=gamma,
            vocab_size=self.model_runner.vocab_size,
            device=self.model_runner.device,
            dtype=torch.float32,
        )
        for step_idx in range(gamma):
            token_id, q_value = self.decode_one(
                session,
                next_input_token_id,
                session.round_state.logits_row_view(step_idx),
            )
            session.round_state.append_step(token_id, q_value)
            next_input_token_id = token_id
        return session.round_state

    def commit_external_token(
        self,
        session: EdgeSession,
        token_id: int,
    ) -> int:
        self.state_bridge.inject_external_token(session, token_id, self.model_runner)
        self.state_bridge.clear_round_state(session)
        return int(token_id)

    def rollback(self, session: EdgeSession, rejected_count: int) -> None:
        self.state_bridge.rollback(session, rejected_count, self.model_runner)

    def close_session(self, session: EdgeSession) -> None:
        self.scheduler.free_blocks(session)
        scheduler_output = self.scheduler.build_close_step(session.req_id)
        self._execute(scheduler_output)
        self.state_bridge.clear_round_state(session)
        self.sessions.pop(session.req_id, None)

    def _execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput | None:
        output = self.worker.execute_model(scheduler_output)
        if isinstance(output, AsyncModelRunnerOutput):
            return output.get_output()
        if isinstance(output, ModelRunnerOutput):
            return output
        return None

    def _sample(self) -> ModelRunnerOutput:
        output = self.worker.sample_tokens(None)
        if isinstance(output, AsyncModelRunnerOutput):
            return output.get_output()
        if not isinstance(output, ModelRunnerOutput):
            raise RuntimeError("sample_tokens 没有返回 ModelRunnerOutput")
        return output

    def _sample_with_draft_sampler(
        self,
        session: EdgeSession,
        processed_logits_dst: torch.Tensor,
    ) -> tuple[int, float]:
        state = self.model_runner.execute_model_state
        if state is None:
            raise RuntimeError("执行 decode_one 前必须先 execute_model")
        self.model_runner.execute_model_state = None

        if not isinstance(state.hidden_states, torch.Tensor):
            raise NotImplementedError("demo 只覆盖单卡/最后一个 PP rank")

        input_batch = state.input_batch
        hidden_states = cast(torch.Tensor, state.hidden_states)
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        logits = self.model_runner.model.compute_logits(sample_hidden_states)

        token_id, q_value = self.draft_sampler.sample_step(
            logits,
            input_batch,
            processed_logits_dst,
        )
        sampled_tokens = self.draft_sampler.build_sampled_tokens(
            token_id,
            input_batch.seq_lens.device,
        )
        num_sampled = self.draft_sampler.build_num_sampled(input_batch.seq_lens.device)
        num_rejected = self.draft_sampler.build_num_rejected(
            input_batch.seq_lens.device
        )
        self.model_runner.postprocess(
            input_batch,
            sampled_tokens,
            num_sampled,
            num_rejected,
        )
        self.state_bridge.commit_token(session, token_id)
        return token_id, q_value
