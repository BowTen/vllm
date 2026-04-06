# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import httpx
from fastapi import FastAPI
import msgspec
import pytest
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.sampling_params import SamplingParams
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.dssd.edge.coordinator import DSSDRoundCoordinator
from vllm.v1.dssd.transport import HTTPDSSDTransport
from vllm.v1.dssd.verifier.service import DSSDVerifierService
from vllm.v1.dssd.worker.draft_runner import DraftRoundResult
from vllm.entrypoints.serve.dssd.api_router import attach_router


MODEL_NAME = "facebook/opt-125m"
PROMPT = "Hello from DSSD"
ENGINE_MAX_MODEL_LEN = 32
ENGINE_MAX_NUM_BATCHED_TOKENS = 32
ENGINE_GPU_MEMORY_UTILIZATION = 0.1
ENGINE_KV_CACHE_MEMORY_BYTES = 128 * 1024 * 1024

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="DSSD real-model e2e requires CUDA",
)


class _CountingEngineClient:
    def __init__(self, client, *, tokenizer) -> None:
        self._client = client
        self.renderer = SimpleNamespace(
            get_tokenizer=lambda: tokenizer,
            tokenizer=tokenizer,
        )
        self.model_config = getattr(client, "model_config", None)
        self.vllm_config = getattr(client, "vllm_config", None)
        self.draft_requests = []
        self.verify_requests = []
        self.create_session_requests = []
        self.close_session_requests = []
        self.commit_requests = []

    async def dssd_draft_round_async(self, request):
        self.draft_requests.append(request)
        result = await self._client.engine_core.call_utility_async(
            "dssd_draft_round", request
        )
        if isinstance(result, DraftRoundResult):
            return result
        return msgspec.convert(result, type=DraftRoundResult)

    async def dssd_verify_round_async(self, request):
        self.verify_requests.append(request)
        return await self._client.engine_core.call_utility_async(
            "dssd_verify_round", request
        )

    async def dssd_create_verifier_session_async(self, request):
        self.create_session_requests.append(request)
        return await self._client.engine_core.call_utility_async(
            "dssd_create_verifier_session", request
        )

    async def dssd_close_verifier_session_async(self, request):
        self.close_session_requests.append(request)
        return await self._client.engine_core.call_utility_async(
            "dssd_close_verifier_session", request
        )

    async def dssd_commit_verifier_tokens_async(self, request):
        self.commit_requests.append(request)
        return await self._client.engine_core.call_utility_async(
            "dssd_commit_verifier_tokens", request
        )

    def __getattr__(self, name):
        return getattr(self._client, name)


class _RealServing:
    def __init__(self, *, model_name: str, tokenizer, prompt_token_ids: list[int]):
        self.models = SimpleNamespace(
            model_name=lambda *_args, **_kwargs: model_name
        )
        self.renderer = SimpleNamespace(
            get_tokenizer=lambda: tokenizer,
            tokenizer=tokenizer,
        )
        self.default_sampling_params = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "max_tokens": 1,
        }
        self.model_config = SimpleNamespace(max_model_len=ENGINE_MAX_MODEL_LEN)
        self.override_max_tokens = None
        self._prompt_token_ids = prompt_token_ids

    async def render_chat_request(self, request):
        del request
        return [], [{"prompt_token_ids": list(self._prompt_token_ids)}]

    def create_error_response(self, message: str, **kwargs):
        return {"message": message, **kwargs}


def _attach_sampling_params(request):
    def _to_sampling_params(max_tokens, _default_sampling_params):
        return SamplingParams(
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            max_tokens=max_tokens,
            seed=0,
        )

    request.to_sampling_params = _to_sampling_params
    return request


def _make_engine_args(
    *,
    role: str,
    verifier_url: str | None = None,
    attention_backend: AttentionBackendEnum | None = None,
) -> AsyncEngineArgs:
    kwargs = dict(
        model=MODEL_NAME,
        enforce_eager=True,
        max_model_len=ENGINE_MAX_MODEL_LEN,
        max_num_batched_tokens=ENGINE_MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=1,
        gpu_memory_utilization=ENGINE_GPU_MEMORY_UTILIZATION,
        kv_cache_memory_bytes=ENGINE_KV_CACHE_MEMORY_BYTES,
        tensor_parallel_size=1,
        block_size=16,
        disable_log_stats=True,
        async_scheduling=False,
    )
    if attention_backend is not None:
        kwargs["attention_backend"] = attention_backend
    return AsyncEngineArgs(
        **kwargs,
        dssd_config={
            "enabled": True,
            "role": role,
            "gamma": 1,
            "verifier_url": verifier_url,
        },
    )


@pytest.mark.asyncio
async def test_dssd_real_model_e2e_executes_edge_and_verifier_models():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    prompt_token_ids = tokenizer(PROMPT, add_special_tokens=False)["input_ids"]

    original_is_cuda = current_platform.is_cuda
    current_platform.is_cuda = lambda: False
    try:
        async with build_async_engine_client_from_engine_args(
            _make_engine_args(
                role="edge",
                verifier_url="http://testserver",
                attention_backend=AttentionBackendEnum.TRITON_ATTN,
            )
        ) as raw_edge_engine:
            async with build_async_engine_client_from_engine_args(
                _make_engine_args(
                    role="verifier",
                    attention_backend=AttentionBackendEnum.TRITON_ATTN,
                )
            ) as raw_verifier_engine:
                edge_engine = _CountingEngineClient(
                    raw_edge_engine,
                    tokenizer=tokenizer,
                )
                verifier_engine = _CountingEngineClient(
                    raw_verifier_engine,
                    tokenizer=tokenizer,
                )

                app = FastAPI()
                app.state.dssd_verifier_service = DSSDVerifierService(
                    verifier_engine,
                    verifier_engine.vllm_config,
                )
                attach_router(app)

                transport = HTTPDSSDTransport(
                    base_url="http://testserver",
                    network_simulation=SimpleNamespace(
                        latency_ms=0.0,
                        bandwidth_mbps=None,
                        jitter_ms=0.0,
                    ),
                    client=httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=app),
                        base_url="http://testserver",
                    ),
                )
                try:
                    coordinator = DSSDRoundCoordinator(
                        edge_engine=edge_engine,
                        transport=transport,
                        gamma=1,
                    )
                    response = await coordinator.create_chat_completion(
                        request=_attach_sampling_params(
                            SimpleNamespace(
                                stream=False,
                                max_tokens=1,
                                stop_token_ids=[],
                                user="edge-user",
                                skip_special_tokens=True,
                            )
                        ),
                        raw_request=None,
                        serving=_RealServing(
                            model_name=MODEL_NAME,
                            tokenizer=tokenizer,
                            prompt_token_ids=prompt_token_ids,
                        ),
                    )
                finally:
                    await transport.aclose()
    finally:
        current_platform.is_cuda = original_is_cuda

    assert not isinstance(response, dict), response
    assert edge_engine.draft_requests
    assert verifier_engine.verify_requests
    assert verifier_engine.create_session_requests
    assert response.choices[0].message.content is not None
    assert response.choices[0].message.content != ""
    assert response.usage.prompt_tokens == len(prompt_token_ids)
    assert response.usage.completion_tokens == 1
    assert response.choices[0].finish_reason in {"length", "stop"}


@pytest.mark.asyncio
async def test_dssd_real_model_e2e_streams_sse_chunks():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )
    prompt_token_ids = tokenizer(PROMPT, add_special_tokens=False)["input_ids"]

    original_is_cuda = current_platform.is_cuda
    current_platform.is_cuda = lambda: False
    try:
        async with build_async_engine_client_from_engine_args(
            _make_engine_args(
                role="edge",
                verifier_url="http://testserver",
                attention_backend=AttentionBackendEnum.TRITON_ATTN,
            )
        ) as raw_edge_engine:
            async with build_async_engine_client_from_engine_args(
                _make_engine_args(
                    role="verifier",
                    attention_backend=AttentionBackendEnum.TRITON_ATTN,
                )
            ) as raw_verifier_engine:
                edge_engine = _CountingEngineClient(
                    raw_edge_engine,
                    tokenizer=tokenizer,
                )
                verifier_engine = _CountingEngineClient(
                    raw_verifier_engine,
                    tokenizer=tokenizer,
                )

                app = FastAPI()
                app.state.dssd_verifier_service = DSSDVerifierService(
                    verifier_engine,
                    verifier_engine.vllm_config,
                )
                attach_router(app)

                transport = HTTPDSSDTransport(
                    base_url="http://testserver",
                    network_simulation=SimpleNamespace(
                        latency_ms=0.0,
                        bandwidth_mbps=None,
                        jitter_ms=0.0,
                    ),
                    client=httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=app),
                        base_url="http://testserver",
                    ),
                )
                try:
                    coordinator = DSSDRoundCoordinator(
                        edge_engine=edge_engine,
                        transport=transport,
                        gamma=1,
                    )
                    stream = coordinator.create_chat_completion_stream(
                        request=_attach_sampling_params(
                            SimpleNamespace(
                                stream=True,
                                max_tokens=1,
                                stop_token_ids=[],
                                user="edge-user",
                                skip_special_tokens=True,
                            )
                        ),
                        raw_request=None,
                        serving=_RealServing(
                            model_name=MODEL_NAME,
                            tokenizer=tokenizer,
                            prompt_token_ids=prompt_token_ids,
                        ),
                    )
                    chunks = []
                    async for chunk in stream:
                        chunks.append(chunk)
                finally:
                    await transport.aclose()
    finally:
        current_platform.is_cuda = original_is_cuda

    assert chunks
    assert chunks[-1] == "data: [DONE]\n\n"

    payload_chunks = [
        json.loads(chunk.removeprefix("data: ").rstrip("\n"))
        for chunk in chunks
        if chunk != "data: [DONE]\n\n"
    ]
    assert payload_chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert payload_chunks[-1]["choices"][0]["finish_reason"] in {"length", "stop"}
