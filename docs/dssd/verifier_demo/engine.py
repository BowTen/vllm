from __future__ import annotations

from typing import cast

import torch

from vllm.config import VllmConfig
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.worker.gpu.input_batch import InputBatch
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
    """真正绑定 vLLM 执行流的 verifier 端执行层。"""

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

    def init_device(self) -> None:
        self.worker.init_device()

    def load_model(self) -> None:
        self.worker.load_model()

    def init_kv_cache(self) -> None:
        # 真实 vLLM 初始化链路是：
        # determine_available_memory -> 构建 KVCacheConfig -> initialize_from_config
        # 这里不展开，只保留接口位置。
        raise NotImplementedError("demo 不负责构造 KVCacheConfig")

    def _resolve_fixed_gamma(self) -> int:
        runner_gamma = self.model_runner.num_speculative_steps
        sampler_gamma = self.verifier_sampler.num_speculative_steps
        if sampler_gamma != runner_gamma:
            raise ValueError(
                "verifier demo 的 gamma 必须在初始化时固定，且 sampler 和 "
                "model_runner 必须使用同一个 num_speculative_steps"
            )
        return runner_gamma

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
        bootstrap_token_id = self.run_prefill(session)
        self.sessions[req_id] = session
        return VerifierOpenSessionResult(
            req_id=req_id,
            bootstrap_token_id=bootstrap_token_id,
        )

    def run_prefill(self, session: VerifierSession) -> int:
        scheduler_output = self.scheduler.build_open_session_step(session)
        self._execute(scheduler_output)
        input_batch, hidden_states = self._take_execute_state()
        bootstrap_token_id = self._sample_bootstrap_token_without_commit(
            hidden_states,
            input_batch,
        )
        self._postprocess_prefill_only(input_batch)
        self.state_bridge.finish_prefill_without_commit(session)
        return bootstrap_token_id

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
        result = self._sample_with_dssd(session, request)
        self.state_bridge.set_round_result(session, result)
        return result

    def close_session(self, session: VerifierSession) -> None:
        self.scheduler.free_blocks(session)
        scheduler_output = self.scheduler.build_close_step(session.req_id)
        self._execute(scheduler_output)
        self.state_bridge.remove_round_state(session)
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

    def _take_execute_state(self) -> tuple[InputBatch, torch.Tensor]:
        state = self.model_runner.execute_model_state
        if state is None:
            raise RuntimeError("执行采样前必须先 execute_model")
        self.model_runner.execute_model_state = None

        if not isinstance(state.hidden_states, torch.Tensor):
            raise NotImplementedError("demo 只覆盖单卡/最后一个 PP rank")
        return state.input_batch, cast(torch.Tensor, state.hidden_states)

    def _sample_bootstrap_token_without_commit(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
    ) -> int:
        sampler_output, _, _ = self.model_runner.sample(
            hidden_states,
            input_batch,
            grammar_output=None,
        )
        if sampler_output.sampled_token_ids.numel() == 0:
            raise RuntimeError("bootstrap 阶段没有采到首个 token")
        return int(sampler_output.sampled_token_ids[0, 0].item())

    def _postprocess_prefill_only(self, input_batch: InputBatch) -> None:
        # bootstrap 阶段只把 prompt 标记为“已计算”，不提交任何输出 token。
        width = max(1, self.model_runner.num_speculative_steps + 1)
        device = input_batch.seq_lens.device
        sampled_tokens = torch.full(
            (input_batch.num_reqs, width),
            fill_value=-1,
            dtype=torch.int64,
            device=device,
        )
        zeros = torch.zeros(input_batch.num_reqs, dtype=torch.int32, device=device)
        self.model_runner.postprocess(
            input_batch,
            sampled_tokens,
            zeros,
            zeros,
        )

    def _sample_with_dssd(
        self,
        session: VerifierSession,
        request: VerifierRoundRequest,
    ) -> VerifierRoundResult:
        input_batch, hidden_states = self._take_execute_state()
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        logits = self.model_runner.model.compute_logits(sample_hidden_states)

        sampler_output, result = self.verifier_sampler(
            logits, input_batch, request
        )
        num_sampled = self.verifier_sampler.build_num_sampled(
            input_batch, result.accepted_len
        )
        num_rejected = self.verifier_sampler.build_num_rejected(
            input_batch,
            result.accepted_len,
            draft_len=len(request.draft_token_ids),
        )
        self.state_bridge.commit_committed_token_before_postprocess(
            session,
            request.committed_token_id,
            self.model_runner,
        )
        self.model_runner.postprocess(
            input_batch,
            sampler_output.sampled_token_ids.to(
                device=input_batch.seq_lens.device, dtype=torch.int64
            ),
            num_sampled,
            num_rejected,
        )
        return result

    def _install_dssd_sampler_hook(self) -> None:
        # 第一版不建议全局 monkeypatch GPUModelRunner.sample_tokens。
        # 更清晰的做法是在 verify_round 内部显式走 _sample_with_dssd。
        return None
