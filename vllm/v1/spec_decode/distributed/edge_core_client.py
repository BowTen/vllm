# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import contextlib
import queue
from concurrent.futures import Future as ConcurrentFuture
from dataclasses import dataclass
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any, Callable

import aiohttp
import msgspec.msgpack
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreOutputs,
    EngineCoreRequest,
    FinishReason,
)
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.executor import Executor
from vllm.v1.spec_decode.distributed.protocol import (
    CloseSessionRequest,
    DraftProposal,
    MSGPACK_ENCODER,
    OpenSessionRequest,
    OpenSessionResponse,
    ResyncSessionRequest,
    ResyncSessionResponse,
    SamplingMetadata,
    VerificationResult,
)
from vllm.v1.spec_decode.distributed.runtime import EdgeDraftRunner, deserialize_probs
from vllm.v1.spec_decode.distributed.sampling import (
    is_terminal_token,
    sample_from_probs,
)

logger = init_logger(__name__)


@dataclass
class EdgeSessionState:
    request_id: str
    prompt_len: int
    sampling: SamplingMetadata
    generator: torch.Generator
    accepted_prefix_token_ids: list[int]
    version: int = 0
    proposal_id: int = 0


class VerifierRPCClient:
    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._session: aiohttp.ClientSession | None = None

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def open_session(
        self,
        request: OpenSessionRequest,
    ) -> OpenSessionResponse:
        return await self._post("open_session", request, OpenSessionResponse)

    async def verify_proposal(
        self,
        proposal: DraftProposal,
    ) -> VerificationResult:
        return await self._post("verify_proposal", proposal, VerificationResult)

    async def resync_session(
        self,
        request: ResyncSessionRequest,
    ) -> ResyncSessionResponse:
        return await self._post("resync_session", request, ResyncSessionResponse)

    async def close_session(self, request: CloseSessionRequest) -> None:
        await self._post("close_session", request, type(None))

    async def _post(
        self,
        path: str,
        payload: Any,
        response_type: Any,
    ) -> Any:
        session = await self._get_session()
        async with session.post(
            f"{self.base_url}/{path}",
            data=MSGPACK_ENCODER.encode(payload),
            headers={"content-type": "application/msgpack"},
        ) as response:
            response.raise_for_status()
            raw = await response.read()
        if response_type is type(None):
            return None
        return msgspec.msgpack.decode(raw, type=response_type)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout, trust_env=True)
        return self._session


class DistributedSpecEdgeCoreClient(EngineCoreClient):
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_count: int = 1,
        client_index: int = 0,
        sync_mode: bool = False,
    ) -> None:
        del executor_class, log_stats, client_count
        self.vllm_config = vllm_config
        self.client_index = client_index
        self._sync_mode = sync_mode
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        assert self.speculative_config.verifier_url is not None
        self.engine_ranks_managed = [0]
        self.resources = SimpleNamespace(engine_dead=False)
        self._paused = False
        self._outputs: asyncio.Queue[EngineCoreOutputs | Exception] = asyncio.Queue()
        self._sync_outputs: queue.Queue[EngineCoreOutputs | Exception] | None = None
        self._sessions: dict[str, EdgeSessionState] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_ready = Event()
        self._loop_thread: Thread | None = None
        self._shutdown_task: asyncio.Task[None] | None = None
        self._draft_runner = EdgeDraftRunner(vllm_config)
        self._verifier = VerifierRPCClient(
            self.speculative_config.verifier_url,
            self.speculative_config.verifier_timeout_s,
        )
        if self._sync_mode:
            self._sync_outputs = queue.Queue()
            self._start_background_loop()

    def shutdown(self):
        if self._sync_mode:
            if self._loop is not None:
                with contextlib.suppress(Exception):
                    self._run_coroutine_sync(self._shutdown_async())
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread is not None:
                self._loop_thread.join(timeout=5)
            return

        try:
            loop = asyncio.get_running_loop()
            if self._shutdown_task is None or self._shutdown_task.done():
                self._shutdown_task = loop.create_task(self._shutdown_async())
        except RuntimeError:
            with contextlib.suppress(Exception):
                asyncio.run(self._shutdown_async())

    async def shutdown_async(self) -> None:
        await self._shutdown_async()

    def get_output(self) -> EngineCoreOutputs:
        if not self._sync_mode or self._sync_outputs is None:
            raise NotImplementedError("DistributedSpecEdgeCoreClient is async-only.")
        outputs = self._sync_outputs.get()
        if isinstance(outputs, Exception):
            self.resources.engine_dead = True
            raise outputs
        return outputs

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return ("generate",)

    def add_request(self, request: EngineCoreRequest) -> None:
        if not self._sync_mode:
            raise NotImplementedError("DistributedSpecEdgeCoreClient is async-only.")
        self._run_coroutine_sync(self.add_request_async(request))

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        del is_start, profile_prefix

    def reset_mm_cache(self) -> None:
        return None

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        del reset_running_requests, reset_connector
        return False

    def reset_encoder_cache(self) -> None:
        return None

    def sleep(self, level: int = 1, mode: str = "abort") -> None:
        del level, mode
        self._paused = True

    def wake_up(self, tags: list[str] | None = None) -> None:
        del tags
        self._paused = False

    def is_sleeping(self) -> bool:
        return self._paused

    def execute_dummy_batch(self) -> None:
        return None

    async def execute_dummy_batch_async(self) -> None:
        return None

    def abort_requests(self, request_ids: list[str]) -> None:
        if self._sync_mode:
            self._run_coroutine_sync(self.abort_requests_async(request_ids))
            return
        self._cancel_request_tasks(request_ids)

    def add_lora(self, lora_request: LoRARequest) -> bool:
        del lora_request
        return False

    def remove_lora(self, lora_id: int) -> bool:
        del lora_id
        return False

    def list_loras(self) -> set[int]:
        return set()

    def pin_lora(self, lora_id: int) -> bool:
        del lora_id
        return False

    def save_sharded_state(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        raise NotImplementedError("save_sharded_state is not supported.")

    def collective_rpc(
        self,
        method: str | Callable[..., Any],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[Any]:
        del method, timeout, args, kwargs
        raise NotImplementedError("collective_rpc is not supported.")

    def dp_engines_running(self) -> bool:
        return False

    async def scale_elastic_ep(self, new_data_parallel_size: int) -> None:
        del new_data_parallel_size
        raise NotImplementedError("Elastic scaling is not supported.")

    async def get_output_async(self) -> EngineCoreOutputs:
        if self._sync_mode:
            return await asyncio.to_thread(self.get_output)
        outputs = await self._outputs.get()
        if isinstance(outputs, Exception):
            self.resources.engine_dead = True
            raise outputs
        return outputs

    async def get_supported_tasks_async(self) -> tuple[SupportedTask, ...]:
        return ("generate",)

    async def add_request_async(self, request: EngineCoreRequest) -> None:
        if self._paused:
            raise RuntimeError("Distributed speculative decoding is paused.")
        self._validate_request(request)
        assert request.prompt_token_ids is not None
        sampling = SamplingMetadata.from_sampling_params(
            request.sampling_params  # type: ignore[arg-type]
        )
        seed = sampling.seed
        if seed is None:
            seed = torch.seed()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        session = EdgeSessionState(
            request_id=request.request_id,
            prompt_len=len(request.prompt_token_ids),
            sampling=sampling,
            generator=generator,
            accepted_prefix_token_ids=list(request.prompt_token_ids),
        )
        self._sessions[request.request_id] = session
        task = asyncio.create_task(
            self._run_request(request, session),
            name=f"distributed-spec-{request.request_id}",
        )
        self._tasks[request.request_id] = task

    async def profile_async(
        self, is_start: bool = True, profile_prefix: str | None = None
    ) -> None:
        del is_start, profile_prefix

    async def reset_mm_cache_async(self) -> None:
        return None

    async def reset_prefix_cache_async(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        del reset_running_requests, reset_connector
        return False

    async def reset_encoder_cache_async(self) -> None:
        return None

    async def sleep_async(self, level: int = 1, mode: str = "abort") -> None:
        del level, mode
        self._paused = True

    async def wake_up_async(self, tags: list[str] | None = None) -> None:
        del tags
        self._paused = False

    async def is_sleeping_async(self) -> bool:
        return self._paused

    async def abort_requests_async(self, request_ids: list[str]) -> None:
        self._cancel_request_tasks(request_ids)
        for request_id in request_ids:
            self._sessions.pop(request_id, None)
            with contextlib.suppress(Exception):
                await self._verifier.close_session(
                    CloseSessionRequest(session_id=request_id)
                )

    async def add_lora_async(self, lora_request: LoRARequest) -> bool:
        return self.add_lora(lora_request)

    async def remove_lora_async(self, lora_id: int) -> bool:
        return self.remove_lora(lora_id)

    async def list_loras_async(self) -> set[int]:
        return self.list_loras()

    async def pin_lora_async(self, lora_id: int) -> bool:
        return self.pin_lora(lora_id)

    async def save_sharded_state_async(
        self, path: str, pattern: str | None = None, max_size: int | None = None
    ) -> None:
        self.save_sharded_state(path, pattern, max_size)

    async def collective_rpc_async(
        self,
        method: str | Callable[..., Any],
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> list[Any]:
        return self.collective_rpc(method, timeout, args, kwargs)

    async def pause_scheduler_async(
        self, mode: str = "abort", clear_cache: bool = True
    ) -> None:
        del clear_cache
        self._paused = True
        if mode == "abort":
            await self.abort_requests_async(list(self._tasks))

    async def resume_scheduler_async(self) -> None:
        self._paused = False

    async def is_scheduler_paused_async(self) -> bool:
        return self._paused

    async def _run_request(
        self,
        request: EngineCoreRequest,
        session: EdgeSessionState,
    ) -> None:
        try:
            response = await self._verifier.open_session(
                OpenSessionRequest(
                    session_id=request.request_id,
                    prompt_token_ids=list(session.accepted_prefix_token_ids),
                    sampling_metadata=session.sampling,
                    initial_version=session.version,
                )
            )
            if (
                response.vocab_size is not None
                and response.vocab_size != self._draft_runner.vocab_size
            ):
                raise ValueError(
                    "Draft and verifier vocab sizes differ: "
                    f"{self._draft_runner.vocab_size} != {response.vocab_size}."
                )

            while True:
                finish_reason = self._maybe_finish_without_new_tokens(session)
                if finish_reason is not None:
                    await self._emit_output(
                        request.request_id,
                        [],
                        finish_reason=finish_reason,
                    )
                    return

                proposal_output = await self._draft_runner.propose(
                    session_id=request.request_id,
                    proposal_id=session.proposal_id,
                    base_version=session.version,
                    accepted_prefix_token_ids=session.accepted_prefix_token_ids,
                    prompt_len=session.prompt_len,
                    sampling=session.sampling,
                    generator=session.generator,
                )
                proposal = proposal_output.proposal
                if not proposal.draft_token_ids:
                    await self._emit_output(
                        request.request_id,
                        [],
                        finish_reason=FinishReason.LENGTH,
                    )
                    return

                verification = await self._verifier.verify_proposal(proposal)
                (
                    new_token_ids,
                    finish_reason,
                    stop_reason,
                    needs_resync,
                ) = self._apply_verification_result(session, verification)

                if needs_resync and finish_reason is None:
                    await self._verifier.resync_session(
                        ResyncSessionRequest(
                            session_id=request.request_id,
                            accepted_prefix_token_ids=list(
                                session.accepted_prefix_token_ids
                            ),
                            edge_version=session.version,
                        )
                    )

                await self._emit_output(
                    request.request_id,
                    new_token_ids,
                    finish_reason=finish_reason,
                    stop_reason=stop_reason,
                )
                session.proposal_id += 1

                if finish_reason is not None:
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception(
                "Distributed speculative request %s failed.",
                request.request_id,
            )
            await self._publish_output(exc)
        finally:
            close_task = asyncio.create_task(
                self._verifier.close_session(
                    CloseSessionRequest(session_id=request.request_id)
                )
            )
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError:
                with contextlib.suppress(Exception):
                    await asyncio.shield(close_task)
                raise
            except Exception:
                pass
            self._tasks.pop(request.request_id, None)
            self._sessions.pop(request.request_id, None)

    def _apply_verification_result(
        self,
        session: EdgeSessionState,
        verification: VerificationResult,
    ) -> tuple[list[int], FinishReason | None, int | str | None, bool]:
        if verification.base_version != session.version:
            raise ValueError(
                f"Session {session.request_id} verification version mismatch: "
                f"got {verification.base_version}, expected {session.version}."
            )
        committed = list(verification.accepted_token_ids)
        needs_resync = verification.reject_pos is not None
        if needs_resync:
            probs = deserialize_probs(
                verification.target_probs_at_reject_pos or b"",
                self._draft_runner.vocab_size,
            )
            committed.append(sample_from_probs(probs, session.generator))
        elif verification.bonus_token_id is not None:
            committed.append(verification.bonus_token_id)

        emitted, finish_reason, stop_reason = self._trim_committed_tokens(
            session, committed
        )
        session.accepted_prefix_token_ids.extend(emitted)
        session.version += len(emitted)
        return emitted, finish_reason, stop_reason, needs_resync

    def _trim_committed_tokens(
        self,
        session: EdgeSessionState,
        committed: list[int],
    ) -> tuple[list[int], FinishReason | None, int | str | None]:
        emitted: list[int] = []
        for token_id in committed:
            emitted.append(token_id)
            output_len_after = (
                len(session.accepted_prefix_token_ids) - session.prompt_len
            ) + len(emitted)
            if is_terminal_token(token_id, session.sampling, output_len_after):
                stop_reason = (
                    token_id if token_id in session.sampling.stop_token_ids else None
                )
                return emitted, FinishReason.STOP, stop_reason
            if (
                session.sampling.max_tokens is not None
                and output_len_after >= session.sampling.max_tokens
            ):
                return emitted, FinishReason.LENGTH, None
        return emitted, None, None

    def _maybe_finish_without_new_tokens(
        self, session: EdgeSessionState
    ) -> FinishReason | None:
        if session.sampling.max_tokens is None:
            return None
        output_len = len(session.accepted_prefix_token_ids) - session.prompt_len
        if output_len >= session.sampling.max_tokens:
            return FinishReason.LENGTH
        return None

    async def _emit_output(
        self,
        request_id: str,
        new_token_ids: list[int],
        finish_reason: FinishReason | None = None,
        stop_reason: int | str | None = None,
    ) -> None:
        await self._publish_output(
            EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id=request_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        stop_reason=stop_reason,
                    )
                ]
            )
        )

    async def _publish_output(self, item: EngineCoreOutputs | Exception) -> None:
        if self._sync_mode:
            assert self._sync_outputs is not None
            self._sync_outputs.put_nowait(item)
            return
        await self._outputs.put(item)

    async def _shutdown_async(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()
        self._sessions.clear()
        await self._verifier.close()

    def _cancel_request_tasks(self, request_ids: list[str]) -> None:
        for request_id in request_ids:
            if task := self._tasks.pop(request_id, None):
                task.cancel()

    def _start_background_loop(self) -> None:
        def run_event_loop() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._loop_ready.set()
            try:
                loop.run_forever()
            finally:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                with contextlib.suppress(Exception):
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                with contextlib.suppress(Exception):
                    loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

        self._loop_thread = Thread(
            target=run_event_loop,
            name="DistributedSpecEdgeCoreClientLoop",
            daemon=True,
        )
        self._loop_thread.start()
        self._loop_ready.wait()

    def _run_coroutine_sync(self, coro: Any) -> Any:
        if self._loop is None:
            raise RuntimeError("Distributed speculative loop is not initialized.")
        future: ConcurrentFuture[Any] = asyncio.run_coroutine_threadsafe(
            coro, self._loop
        )
        return future.result()

    def _validate_request(self, request: EngineCoreRequest) -> None:
        if request.prompt_embeds is not None:
            raise NotImplementedError(
                "Distributed draft-model speculative decoding does not support "
                "prompt_embeds."
            )
        if request.mm_features:
            raise NotImplementedError(
                "Distributed draft-model speculative decoding only supports "
                "text-only requests."
            )
        if request.sampling_params is None:
            raise NotImplementedError(
                "Distributed draft-model speculative decoding only supports "
                "generation requests."
            )
        params = request.sampling_params
        assert isinstance(params, SamplingParams)
        if params.logprobs is not None or params.prompt_logprobs is not None:
            raise NotImplementedError(
                "Distributed draft-model speculative decoding does not support "
                "logprobs yet."
            )
        if params.structured_outputs is not None:
            raise NotImplementedError(
                "Distributed draft-model speculative decoding does not support "
                "structured outputs yet."
            )
        if params.bad_words:
            raise NotImplementedError(
                "Distributed draft-model speculative decoding does not support "
                "bad_words yet."
            )
