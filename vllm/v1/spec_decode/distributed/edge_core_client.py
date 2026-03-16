# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import contextlib
import queue
from concurrent.futures import Future as ConcurrentFuture
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any, Callable

import aiohttp
import msgspec.msgpack

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
from vllm.v1.outputs import LogprobsLists, LogprobsTensors
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.executor import Executor
from vllm.v1.spec_decode.distributed.errors import (
    VerifierQueueTimeoutError,
    VerifierSessionMissingError,
)
from vllm.v1.spec_decode.distributed.edge_session import (
    EdgeSessionCore,
    EdgeSessionState,
)
from vllm.v1.spec_decode.distributed.protocol import (
    CloseSessionRequest,
    DraftProposal,
    MSGPACK_ENCODER,
    OpenSessionRequest,
    OpenSessionResponse,
    ResyncSessionRequest,
    ResyncSessionResponse,
    VerificationResult,
)
from vllm.v1.spec_decode.distributed.runtime import EdgeDraftRunner
from vllm.v1.spec_decode.distributed.structured_output import StructuredOutputFactory

logger = init_logger(__name__)


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
            if response.status == 409:
                raise VerifierSessionMissingError(await response.text())
            if response.status == 408:
                raise VerifierQueueTimeoutError(await response.text())
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
        self._structured_output_factory: StructuredOutputFactory | None = None
        structured_output_factory = None
        draft_model_config = getattr(self.speculative_config, "draft_model_config", None)
        if draft_model_config is not None:
            self._structured_output_factory = StructuredOutputFactory(
                model_name=draft_model_config.model,
                trust_remote_code=draft_model_config.trust_remote_code,
                num_speculative_tokens=self.speculative_config.num_speculative_tokens,
                vocab_size=self._draft_runner.vocab_size,
            )
            structured_output_factory = self._structured_output_factory
        self._session_core = EdgeSessionCore(
            self._draft_runner,
            self._verifier,
            structured_output_factory=structured_output_factory,
            num_speculative_tokens=self.speculative_config.num_speculative_tokens,
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
        session = self._session_core.create_session(request)
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
        await self._session_core.abort_sessions(request_ids)

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
            async for step in self._session_core.run_request(request, session):
                await self._emit_output(
                    request.request_id,
                    step.new_token_ids,
                    new_logprobs=step.new_logprobs,
                    new_prompt_logprobs_tensors=step.new_prompt_logprobs_tensors,
                    finish_reason=step.finish_reason,
                    stop_reason=step.stop_reason,
                )
                if step.finish_reason is not None:
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
                self._session_core.close_session(request.request_id)
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

    async def _emit_output(
        self,
        request_id: str,
        new_token_ids: list[int],
        new_logprobs: LogprobsLists | None = None,
        new_prompt_logprobs_tensors: LogprobsTensors | None = None,
        finish_reason: FinishReason | None = None,
        stop_reason: int | str | None = None,
    ) -> None:
        await self._publish_output(
            EngineCoreOutputs(
                outputs=[
                    EngineCoreOutput(
                        request_id=request_id,
                        new_token_ids=new_token_ids,
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=new_prompt_logprobs_tensors,
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
        self._session_core.shutdown()
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
        if (
            params.structured_outputs is not None
            and not params.structured_outputs.all_constraints_none()
        ):
            if self._structured_output_factory is None:
                raise NotImplementedError(
                    "Distributed draft-model speculative decoding requires a "
                    "local draft model config to use structured outputs."
                )
            self._structured_output_factory.resolve_sampling_params(
                params,
                getattr(self.vllm_config, "structured_outputs_config", None),
            )
