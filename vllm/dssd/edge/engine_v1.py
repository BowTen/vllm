from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

from .state_bridge import validate_edge_sampling_params
from .types import EdgeRoundState, EdgeSession

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm.v1.worker.gpu_worker import Worker

    from .sampler_v1 import DSSDEdgeDraftSamplerV1
    from .scheduler import EdgeSchedulerAdapter
    from .state_bridge_v1 import EdgeStateBridgeV1


class EdgeDecodeEngineV1:
    """Edge decode orchestration for the old v1 GPU model runner path."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        worker: "Worker",
        scheduler: "EdgeSchedulerAdapter",
        state_bridge: "EdgeStateBridgeV1",
        draft_sampler: "DSSDEdgeDraftSamplerV1",
        sessions: dict[str, EdgeSession] | None = None,
    ) -> None:
        self.vllm_config = vllm_config
        self.worker = worker
        self.scheduler = scheduler
        self.state_bridge = state_bridge
        self.draft_sampler = draft_sampler
        self.sessions = {} if sessions is None else sessions

    @property
    def model_runner(self) -> "GPUModelRunner":
        return self.worker.model_runner

    @property
    def vocab_size(self) -> int:
        model_runner = self.model_runner
        if hasattr(model_runner, "vocab_size"):
            return int(model_runner.vocab_size)
        return model_runner.model_config.get_vocab_size()

    def open_session(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        lora_request: LoRARequest | None = None,
    ) -> EdgeSession:
        validate_edge_sampling_params(sampling_params)
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
        self._execute(self.scheduler.build_prefill_step(session))
        self._clear_execute_model_state()
        session.num_computed_tokens = session.prompt_len
        session.total_len = session.prompt_len
        return self.state_bridge.bootstrap_first_token(
            session,
            bootstrap_token_id,
            self.model_runner,
        )

    def generate_local(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        lora_request: LoRARequest | None = None,
    ) -> list[int]:
        session = self.open_session(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        try:
            max_tokens = sampling_params.max_tokens or 0
            if max_tokens <= 0:
                return []

            processed_logits = torch.empty(
                (1, self.vocab_size),
                dtype=torch.float32,
                device=self.model_runner.device,
            )
            next_token_id = self._bootstrap_local_first_token(
                session,
                processed_logits,
            )
            output_token_ids = [next_token_id]

            if self._should_stop_local_generation(sampling_params, next_token_id):
                return output_token_ids

            while len(output_token_ids) < max_tokens:
                next_token_id, _q_value = self.decode_one(
                    session,
                    next_token_id,
                    processed_logits,
                )
                output_token_ids.append(next_token_id)
                if self._should_stop_local_generation(
                    sampling_params,
                    next_token_id,
                ):
                    break

            return output_token_ids
        finally:
            self.close_session(session)

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
        self._execute(self.scheduler.build_decode_step(session))
        return self._sample_with_draft_sampler(session, processed_logits_dst)

    def draft(
        self,
        session: EdgeSession,
        first_token_id: int,
        gamma: int,
    ) -> EdgeRoundState:
        self.state_bridge.clear_round_state(session)
        session.round_state.committed_token_id = int(first_token_id)
        session.round_state.prepare_logits_buffer(
            gamma=gamma,
            vocab_size=self.vocab_size,
            device=self.model_runner.device,
            dtype=torch.float32,
        )

        next_input_token_id = int(first_token_id)
        for step_idx in range(gamma):
            token_id, q_value = self.decode_one(
                session,
                next_input_token_id,
                session.round_state.logits_row_view(step_idx),
            )
            session.round_state.append_step(token_id, q_value)
            next_input_token_id = token_id

        if gamma > 0:
            self._compute_pending_token(session, next_input_token_id)

        return session.round_state

    def commit_external_token(
        self,
        session: EdgeSession,
        token_id: int,
    ) -> int:
        self.state_bridge.inject_external_token(
            session,
            token_id,
            self.model_runner,
        )
        self.state_bridge.clear_round_state(session)
        return int(token_id)

    def rollback(self, session: EdgeSession, rejected_count: int) -> None:
        self.state_bridge.rollback(session, rejected_count, self.model_runner)

    def close_session(self, session: EdgeSession) -> None:
        self.scheduler.free_blocks(session)
        self._execute(self.scheduler.build_close_step(session.req_id))
        self.state_bridge.clear_round_state(session)
        self.sessions.pop(session.req_id, None)

    def _execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput | None:
        output = self.worker.execute_model(scheduler_output)
        if isinstance(output, AsyncModelRunnerOutput):
            return output.get_output()
        if isinstance(output, ModelRunnerOutput):
            return output
        return None

    def _sample_with_draft_sampler(
        self,
        session: EdgeSession,
        processed_logits_dst: torch.Tensor,
    ) -> tuple[int, float]:
        state = self._take_execute_model_state()
        if state is None:
            raise RuntimeError(
                "must call execute_model before sampling edge draft token")

        logits = getattr(state, "logits", None)
        if logits is None:
            raise RuntimeError("model did not produce logits for edge decode")

        token_id, q_value = self.draft_sampler.sample_step(
            logits,
            self.model_runner.input_batch.sampling_metadata,
            processed_logits_dst,
        )
        self.state_bridge.commit_token(session, token_id, self.model_runner)
        return token_id, q_value

    def _compute_pending_token(
        self,
        session: EdgeSession,
        input_token_id: int,
    ) -> None:
        self.state_bridge.prepare_next_decode(
            session,
            input_token_id,
            self.model_runner,
        )
        self._execute(self.scheduler.build_decode_step(session))
        self._clear_execute_model_state()
        self.state_bridge.mark_pending_token_computed(session, self.model_runner)

    def _bootstrap_local_first_token(
        self,
        session: EdgeSession,
        processed_logits_dst: torch.Tensor,
    ) -> int:
        self._execute(self.scheduler.build_prefill_step(session))
        session.num_computed_tokens = session.prompt_len
        session.total_len = session.prompt_len
        token_id, _q_value = self._sample_with_draft_sampler(
            session,
            processed_logits_dst,
        )
        return token_id

    def _should_stop_local_generation(
        self,
        sampling_params: SamplingParams,
        token_id: int,
    ) -> bool:
        if sampling_params.ignore_eos:
            return False
        eos_token_id = sampling_params._eos_token_id
        return eos_token_id is not None and int(token_id) == int(eos_token_id)

    def _take_execute_model_state(self):
        model_runner = self.model_runner
        take_state = getattr(model_runner, "take_execute_model_state", None)
        if callable(take_state):
            if getattr(model_runner, "execute_model_state", None) is None:
                return None
            return take_state()

        state = getattr(model_runner, "execute_model_state", None)
        model_runner.execute_model_state = None
        return state

    def _clear_execute_model_state(self) -> None:
        self._take_execute_model_state()
