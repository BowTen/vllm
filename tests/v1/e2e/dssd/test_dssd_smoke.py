# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI


ROOT = Path(__file__).resolve().parents[4]
VLLM_DIR = ROOT / "vllm"
DSSD_DIR = VLLM_DIR / "v1" / "dssd"
DSSD_EDGE_DIR = DSSD_DIR / "edge"
DSSD_VERIFIER_DIR = DSSD_DIR / "verifier"
DSSD_SERVE_DIR = VLLM_DIR / "entrypoints" / "serve" / "dssd"


def _install_package_stub(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class _ChatMessage:
    role: str
    content: str | None = None


@dataclass
class _ChatCompletionResponseChoice:
    index: int
    message: _ChatMessage
    finish_reason: str | None = "stop"


@dataclass
class _UsageInfo:
    prompt_tokens: int = 0
    completion_tokens: int | None = 0
    total_tokens: int = 0


@dataclass
class _ChatCompletionResponse:
    id: str
    model: str
    choices: list[_ChatCompletionResponseChoice]
    usage: _UsageInfo
    created: int = 0
    object: str = "chat.completion"


_install_package_stub("vllm", VLLM_DIR)
_install_package_stub("vllm.v1", VLLM_DIR / "v1")
_install_package_stub("vllm.v1.dssd", DSSD_DIR)
_install_package_stub("vllm.v1.dssd.edge", DSSD_EDGE_DIR)
_install_package_stub("vllm.v1.dssd.verifier", DSSD_VERIFIER_DIR)
_install_package_stub("vllm.entrypoints", VLLM_DIR / "entrypoints")
_install_package_stub("vllm.entrypoints.openai", VLLM_DIR / "entrypoints" / "openai")
_install_package_stub(
    "vllm.entrypoints.openai.chat_completion",
    VLLM_DIR / "entrypoints" / "openai" / "chat_completion",
)
_install_package_stub(
    "vllm.entrypoints.openai.engine",
    VLLM_DIR / "entrypoints" / "openai" / "engine",
)
_install_package_stub("vllm.entrypoints.serve", VLLM_DIR / "entrypoints" / "serve")
_install_package_stub("vllm.entrypoints.serve.dssd", DSSD_SERVE_DIR)

chat_protocol_module = types.ModuleType(
    "vllm.entrypoints.openai.chat_completion.protocol"
)
chat_protocol_module.ChatMessage = _ChatMessage
chat_protocol_module.ChatCompletionResponseChoice = _ChatCompletionResponseChoice
chat_protocol_module.ChatCompletionResponse = _ChatCompletionResponse
sys.modules["vllm.entrypoints.openai.chat_completion.protocol"] = (
    chat_protocol_module
)

engine_protocol_module = types.ModuleType(
    "vllm.entrypoints.openai.engine.protocol"
)
engine_protocol_module.ErrorResponse = dict
engine_protocol_module.UsageInfo = _UsageInfo
sys.modules["vllm.entrypoints.openai.engine.protocol"] = engine_protocol_module

protocol_module = _load_module(
    "vllm.v1.dssd.protocol",
    DSSD_DIR / "protocol.py",
)
metrics_module = _load_module(
    "vllm.v1.dssd.metrics",
    DSSD_DIR / "metrics.py",
)
transport_module = _load_module(
    "vllm.v1.dssd.transport",
    DSSD_DIR / "transport.py",
)
session_module = _load_module(
    "vllm.v1.dssd.verifier.session",
    DSSD_VERIFIER_DIR / "session.py",
)
service_module = _load_module(
    "vllm.v1.dssd.verifier.service",
    DSSD_VERIFIER_DIR / "service.py",
)
router_module = _load_module(
    "vllm.entrypoints.serve.dssd.api_router",
    DSSD_SERVE_DIR / "api_router.py",
)
resample_module = _load_module(
    "vllm.v1.dssd.worker.resample",
    DSSD_DIR / "worker" / "resample.py",
)
coordinator_module = _load_module(
    "vllm.v1.dssd.edge.coordinator",
    DSSD_EDGE_DIR / "coordinator.py",
)

BindVerifierResponse = protocol_module.BindVerifierResponse
CreateSessionResponse = protocol_module.CreateSessionResponse
VerifyRoundResponse = protocol_module.VerifyRoundResponse
DSSDRequestMetrics = metrics_module.DSSDRequestMetrics
HTTPDSSDTransport = transport_module.HTTPDSSDTransport
DSSDVerifierService = service_module.DSSDVerifierService
attach_router = router_module.attach_router
DSSDRoundCoordinator = coordinator_module.DSSDRoundCoordinator


def test_metrics_track_communication_bytes():
    metrics = DSSDRequestMetrics(request_id="req-1")

    metrics.record_uplink(32)
    metrics.record_downlink(128)

    assert metrics.uplink_bytes == 32
    assert metrics.downlink_bytes == 128


@pytest.mark.asyncio
async def test_dssd_smoke_runs_single_http_round_trip():
    class _StubVerifierEngine:
        def __init__(self) -> None:
            self.model_config = SimpleNamespace(model="target-model")

        async def dssd_verify_round_async(self, request):
            return VerifyRoundResponse(
                verifier_session_id=request.verifier_session_id,
                seq_no=request.seq_no,
                accepted_count=1,
                all_accepted=True,
                bonus_token_id=9,
            )

    class _StubEdgeEngine:
        async def dssd_draft_round_async(self, request):
            return SimpleNamespace(
                draft_token_ids=[3],
                q_values=[0.7],
                q_dists_handle="edge-1:0",
                q_distributions=[[0.3, 0.7]],
            )

    class _Tokenizer:
        def decode(self, token_ids, skip_special_tokens=False):
            del skip_special_tokens
            return "|".join(f"tok{token_id}" for token_id in token_ids)

    class _StubServing:
        def __init__(self) -> None:
            self.models = SimpleNamespace(
                model_name=lambda *_args, **_kwargs: "edge-model"
            )
            self.renderer = SimpleNamespace(
                get_tokenizer=lambda: _Tokenizer(),
                tokenizer=_Tokenizer(),
            )

        async def render_chat_request(self, request):
            del request
            return [], [{"prompt_token_ids": [11, 12]}]

        def create_error_response(self, message: str, **kwargs):
            return {"message": message, **kwargs}

    app = FastAPI()
    app.state.dssd_verifier_service = DSSDVerifierService(
        _StubVerifierEngine(),
        SimpleNamespace(
            dssd_config=SimpleNamespace(enabled=True, role="verifier", gamma=1),
        ),
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
            edge_engine=_StubEdgeEngine(),
            transport=transport,
            gamma=1,
        )
        response = await coordinator.create_chat_completion(
            request=SimpleNamespace(
                stream=False,
                max_tokens=2,
                stop_token_ids=[],
                user="edge-user",
            ),
            raw_request=None,
            serving=_StubServing(),
        )
    finally:
        await transport.aclose()

    assert response.choices[0].message.content == "tok3|tok9"
    assert response.usage.prompt_tokens == 2
    assert response.usage.completion_tokens == 2
    assert response.choices[0].finish_reason == "length"
    assert app.state.dssd_verifier_service.session_manager._sessions == {}
