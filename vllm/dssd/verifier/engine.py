from __future__ import annotations

from typing import Any, cast

import torch

from vllm.config import VllmConfig
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker as GPUWorker

from .sampler import DSSDVerifierSampler
from .scheduler import VerifierSchedulerAdapter
from .state_bridge import VerifierStateBridge
from .types import (
    VerifierOpenSessionResult,
    VerifierRoundRequest,
    VerifierRoundResult,
    VerifierSession,
)


class VerifierDecodeEngine:
    def __init__(
        self,
        vllm_config: VllmConfig,
        worker: GPUWorker,
        scheduler: VerifierSchedulerAdapter,
        state_bridge: VerifierStateBridge,
        verifier_sampler: DSSDVerifierSampler,
    ) -> None:
        self.vllm_config = vllm_config
        self.worker = worker
        self.scheduler = scheduler
        self.state_bridge = state_bridge
        self.verifier_sampler = verifier_sampler
        self.gamma = self._resolve_fixed_gamma()
        self.sessions: dict[str, VerifierSession] = {}

    @property
    def model_runner(self) -> GPUModelRunner:
        return self.worker.model_runner

    def open_session(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        lora_request: LoRARequest | None = None,
    ) -> VerifierOpenSessionResult:
        session = VerifierSession(
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
            token_ids=list(prompt_token_ids),
            lora_request=lora_request,
        )

        scheduler_output = self.scheduler.build_open_session_step(session)
        self._execute(scheduler_output)
        state = self.model_runner.take_execute_model_state()
        hidden_states = self._take_hidden_states(state.hidden_states)
        sampler_output = self.model_runner.sample_without_postprocess(
            hidden_states,
            state.input_batch,
            grammar_output=None,
        )
        if sampler_output.sampled_token_ids.numel() == 0:
            raise RuntimeError("bootstrap 阶段没有采到首个 token")

        bootstrap_token_id = int(sampler_output.sampled_token_ids[0, 0].item())
        self._postprocess_prefill_only(state.input_batch)
        self.state_bridge.finish_prefill_without_commit(session)
        self.sessions[req_id] = session
        return VerifierOpenSessionResult(
            req_id=req_id,
            bootstrap_token_id=bootstrap_token_id,
        )

    def verify_round(
        self,
        session: VerifierSession,
        request: VerifierRoundRequest,
    ) -> VerifierRoundResult:
        self.state_bridge.prepare_round(
            session,
            request,
            self.model_runner,
            gamma=self.gamma,
        )
        scheduler_output = self.scheduler.build_verify_step(session, request)
        self._execute(scheduler_output)

        state = self.model_runner.take_execute_model_state()
        hidden_states = self._take_hidden_states(state.hidden_states)
        sample_hidden_states = hidden_states[state.input_batch.logits_indices]
        logits = self.model_runner.model.compute_logits(sample_hidden_states)
        sampler_output, raw_result = self.verifier_sampler(
            logits,
            state.input_batch,
            request,
        )

        self.state_bridge.commit_committed_token_before_postprocess(
            session,
            request.committed_token_id,
            self.model_runner,
        )

        device = state.input_batch.seq_lens.device
        accepted_len = raw_result.accepted_len.to(device=device, dtype=torch.int32)
        draft_len = torch.tensor(
            [len(request.draft_token_ids)],
            device=device,
            dtype=torch.int32,
        )
        self.model_runner.postprocess(
            state.input_batch,
            sampler_output.sampled_token_ids.to(device=device, dtype=torch.int64),
            sampler_output.num_sampled.to(device=device, dtype=torch.int32),
            draft_len - accepted_len.view(1),
        )
        result = raw_result.to_round_result()
        self.state_bridge.set_round_result(session, result)
        return result

    def close_session(self, session: VerifierSession) -> None:
        self.scheduler.free_blocks(session)
        scheduler_output = self.scheduler.build_close_step(session.req_id)
        self._execute(scheduler_output)
        self.state_bridge.remove_round_state(session)
        self.sessions.pop(session.req_id, None)

    def _resolve_fixed_gamma(self) -> int:
        runner_gamma = self.model_runner.num_speculative_steps
        sampler_gamma = self.verifier_sampler.num_speculative_steps
        if sampler_gamma != runner_gamma:
            raise ValueError(
                "verifier engine requires a fixed gamma shared by the sampler "
                "and model_runner"
            )
        return runner_gamma

    def _execute(self, scheduler_output: SchedulerOutput) -> Any | None:
        output = self.worker.execute_model(scheduler_output)
        if isinstance(output, AsyncModelRunnerOutput):
            return output.get_output()
        return output

    @staticmethod
    def _take_hidden_states(hidden_states: object) -> torch.Tensor:
        if not isinstance(hidden_states, torch.Tensor):
            raise NotImplementedError(
                "VerifierDecodeEngine currently only supports the last PP rank"
            )
        return cast(torch.Tensor, hidden_states)

    def _postprocess_prefill_only(self, input_batch) -> None:
        device = input_batch.seq_lens.device
        sampled_tokens = torch.full(
            (input_batch.num_reqs, self.gamma + 1),
            fill_value=-1,
            dtype=torch.int64,
            device=device,
        )
        zeros = torch.zeros(
            input_batch.num_reqs,
            dtype=torch.int32,
            device=device,
        )
        self.model_runner.postprocess(
            input_batch,
            sampled_tokens,
            zeros,
            zeros,
        )
