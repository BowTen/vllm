from __future__ import annotations

from typing import Any

from vllm.config import VllmConfig
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams

from .types import (
    VerifierOpenSessionResult,
    VerifierRoundRequest,
    VerifierRoundResult,
    VerifierSession,
)


class VerifierDecodeEngineV1:
    def __init__(
        self,
        vllm_config: VllmConfig,
        worker,
        scheduler,
        state_bridge,
        verifier_sampler,
    ) -> None:
        self.vllm_config = vllm_config
        self.worker = worker
        self.scheduler = scheduler
        self.state_bridge = state_bridge
        self.verifier_sampler = verifier_sampler
        self.gamma = int(worker.model_runner.num_spec_tokens)
        self.sessions: dict[str, VerifierSession] = {}

    @property
    def model_runner(self):
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
        self._execute(self.scheduler.build_open_session_step(session))
        state = self.model_runner.take_execute_model_state()
        bootstrap_token_id = self.verifier_sampler.sample_bootstrap(
            logits=state.logits,
            sampling_metadata=self.model_runner.input_batch.sampling_metadata,
        )
        self.state_bridge.finish_prefill_without_commit(
            session,
            self.model_runner,
        )
        self.sessions[req_id] = session
        return VerifierOpenSessionResult(
            req_id=req_id,
            bootstrap_token_id=bootstrap_token_id,
        )

    def generate_local(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        lora_request: LoRARequest | None = None,
    ) -> list[int]:
        max_tokens = sampling_params.max_tokens or 0
        if max_tokens <= 0:
            return []

        opened = self.open_session(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        session = self.sessions[req_id]
        bootstrap_token_id = opened.bootstrap_token_id
        self.state_bridge.inject_local_token(
            session,
            bootstrap_token_id,
            self.model_runner,
            computed_delta=1,
        )
        output_token_ids = [bootstrap_token_id]

        try:
            if self._should_stop_local_generation(
                sampling_params,
                bootstrap_token_id,
            ):
                return output_token_ids

            while len(output_token_ids) < max_tokens:
                next_token_id = self.decode_one_local(
                    session,
                    output_token_ids[-1],
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

    def decode_one_local(
        self,
        session: VerifierSession,
        input_token_id: int,
    ) -> int:
        self.state_bridge.prepare_local_decode(
            session,
            input_token_id,
            self.model_runner,
        )
        self._execute(self.scheduler.build_decode_step(session))
        return self._sample_local_token(session)

    def verify_round(
        self,
        session: VerifierSession,
        request: VerifierRoundRequest,
    ) -> VerifierRoundResult:
        snapshot = self._snapshot_round_state(session)
        try:
            self.state_bridge.begin_round(
                session,
                request,
                self.model_runner,
                gamma=self.gamma,
            )
            self._execute(self.scheduler.build_verify_step(session, request))
            state = self.model_runner.take_execute_model_state()
            result = self.verifier_sampler.verify_round(
                logits=state.logits,
                spec_decode_metadata=state.spec_decode_metadata,
                sampling_metadata=self.model_runner.input_batch.sampling_metadata,
                request=request,
            )
            self.state_bridge.finish_round(
                session,
                request,
                result,
                self.model_runner,
            )
            return result
        except Exception:
            self._rollback_round_state(session, snapshot)
            raise

    def close_session(self, session: VerifierSession) -> None:
        self.scheduler.free_blocks(session)
        self._execute(self.scheduler.build_close_step(session.req_id))
        self.state_bridge.remove_round_state(session)
        self.sessions.pop(session.req_id, None)
        if getattr(self.model_runner, "execute_model_state", None) is not None:
            self.model_runner.execute_model_state = None

    def _execute(self, scheduler_output: Any) -> Any | None:
        return self.worker.execute_model(scheduler_output)

    def _sample_local_token(self, session: VerifierSession) -> int:
        state = self.model_runner.take_execute_model_state()
        token_id = self.verifier_sampler.sample_bootstrap(
            logits=state.logits,
            sampling_metadata=self.model_runner.input_batch.sampling_metadata,
        )
        self.state_bridge.commit_local_token(
            session,
            token_id,
            self.model_runner,
        )
        return int(token_id)

    @staticmethod
    def _should_stop_local_generation(
        sampling_params: SamplingParams,
        token_id: int,
    ) -> bool:
        if sampling_params.ignore_eos:
            return False
        eos_token_id = sampling_params._eos_token_id
        return eos_token_id is not None and int(token_id) == int(eos_token_id)

    def _snapshot_round_state(
        self,
        session: VerifierSession,
    ) -> dict[str, Any]:
        model_runner = self.model_runner
        req_state = model_runner.requests[session.req_id]
        input_batch = model_runner.input_batch
        req_idx = input_batch.req_id_to_index[session.req_id]
        return {
            "session_token_ids": list(session.token_ids),
            "session_num_computed_tokens": session.num_computed_tokens,
            "session_total_len": session.total_len,
            "req_output_token_ids": list(req_state.output_token_ids),
            "req_num_computed_tokens": req_state.num_computed_tokens,
            "token_ids_cpu_row": input_batch.token_ids_cpu[req_idx].copy(),
            "is_token_ids_row": (
                input_batch.is_token_ids[req_idx].copy()
                if hasattr(input_batch, "is_token_ids")
                else None
            ),
            "num_tokens_no_spec": input_batch.num_tokens_no_spec[req_idx].copy(),
            "num_computed_tokens_cpu":
            input_batch.num_computed_tokens_cpu[req_idx].copy(),
        }

    def _rollback_round_state(
        self,
        session: VerifierSession,
        snapshot: dict[str, Any],
    ) -> None:
        model_runner = self.model_runner
        req_state = model_runner.requests[session.req_id]
        input_batch = model_runner.input_batch
        req_idx = input_batch.req_id_to_index[session.req_id]

        try:
            session.token_ids[:] = snapshot["session_token_ids"]
            session.num_computed_tokens = snapshot["session_num_computed_tokens"]
            session.total_len = snapshot["session_total_len"]

            req_state.output_token_ids[:] = snapshot["req_output_token_ids"]
            req_state.num_computed_tokens = snapshot["req_num_computed_tokens"]

            input_batch.token_ids_cpu[req_idx] = snapshot["token_ids_cpu_row"]
            if snapshot["is_token_ids_row"] is not None:
                input_batch.is_token_ids[req_idx] = snapshot["is_token_ids_row"]
            input_batch.num_tokens_no_spec[req_idx] = snapshot[
                "num_tokens_no_spec"]
            input_batch.num_computed_tokens_cpu[req_idx] = snapshot[
                "num_computed_tokens_cpu"]
            if hasattr(input_batch, "req_output_token_ids"):
                input_batch.req_output_token_ids[req_idx] = (
                    req_state.output_token_ids)
        finally:
            if getattr(model_runner, "execute_model_state", None) is not None:
                model_runner.execute_model_state = None
            self.state_bridge.remove_round_state(session)
