# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

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


def _attach_sampling_params(request):
    def _to_sampling_params(max_tokens, default_sampling_params):
        return types.SimpleNamespace(
            temperature=default_sampling_params["temperature"],
            top_p=default_sampling_params["top_p"],
            top_k=default_sampling_params["top_k"],
            min_p=default_sampling_params["min_p"],
            seed=999,
            max_tokens=max_tokens,
        )

    request.to_sampling_params = _to_sampling_params
    return request


@contextmanager
def _smoke_modules() -> Iterator[SimpleNamespace]:
    saved_vllm_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "vllm" or name.startswith("vllm.")
    }

    try:
        _install_package_stub("vllm", VLLM_DIR)
        _install_package_stub("vllm.v1", VLLM_DIR / "v1")
        _install_package_stub("vllm.v1.dssd", DSSD_DIR)
        _install_package_stub("vllm.v1.dssd.engine", DSSD_DIR / "engine")
        _install_package_stub("vllm.v1.dssd.edge", DSSD_EDGE_DIR)
        _install_package_stub("vllm.v1.dssd.verifier", DSSD_VERIFIER_DIR)
        _install_package_stub("vllm.entrypoints", VLLM_DIR / "entrypoints")
        _install_package_stub(
            "vllm.entrypoints.openai", VLLM_DIR / "entrypoints" / "openai"
        )
        _install_package_stub(
            "vllm.entrypoints.openai.chat_completion",
            VLLM_DIR / "entrypoints" / "openai" / "chat_completion",
        )
        _install_package_stub(
            "vllm.entrypoints.openai.engine",
            VLLM_DIR / "entrypoints" / "openai" / "engine",
        )
        _install_package_stub(
            "vllm.entrypoints.serve", VLLM_DIR / "entrypoints" / "serve"
        )
        _install_package_stub("vllm.entrypoints.serve.dssd", DSSD_SERVE_DIR)

        sampling_params_module = types.ModuleType("vllm.sampling_params")

        class _SamplingParams:
            pass

        sampling_params_module.SamplingParams = _SamplingParams
        sys.modules["vllm.sampling_params"] = sampling_params_module

        entrypoints_utils_module = types.ModuleType("vllm.entrypoints.utils")

        def _get_max_tokens(
            max_model_len,
            max_tokens,
            input_length,
            default_sampling_params,
            override_max_tokens=None,
        ):
            del input_length
            if max_model_len < 0:
                raise ValueError("max_model_len must be non-negative")
            fallback_max_tokens = (
                max_tokens
                if max_tokens is not None
                else default_sampling_params.get("max_tokens")
            )
            return min(
                value
                for value in (
                    max_model_len,
                    fallback_max_tokens,
                    override_max_tokens,
                )
                if value is not None
            )

        entrypoints_utils_module.get_max_tokens = _get_max_tokens
        sys.modules["vllm.entrypoints.utils"] = entrypoints_utils_module

        chat_protocol_module = types.ModuleType(
            "vllm.entrypoints.openai.chat_completion.protocol"
        )
        chat_protocol_module.ChatMessage = _ChatMessage
        chat_protocol_module.ChatCompletionResponseChoice = (
            _ChatCompletionResponseChoice
        )
        chat_protocol_module.ChatCompletionResponse = _ChatCompletionResponse
        sys.modules["vllm.entrypoints.openai.chat_completion.protocol"] = (
            chat_protocol_module
        )

        engine_protocol_module = types.ModuleType(
            "vllm.entrypoints.openai.engine.protocol"
        )
        engine_protocol_module.ErrorResponse = dict
        engine_protocol_module.UsageInfo = _UsageInfo
        sys.modules["vllm.entrypoints.openai.engine.protocol"] = (
            engine_protocol_module
        )

        protocol_module = _load_module(
            "vllm.v1.dssd.protocol",
            DSSD_DIR / "protocol.py",
        )
        _load_module(
            "vllm.v1.dssd.engine.batch_planner",
            DSSD_DIR / "engine" / "batch_planner.py",
        )
        metrics_module = _load_module(
            "vllm.v1.dssd.metrics",
            DSSD_DIR / "metrics.py",
        )
        transport_module = _load_module(
            "vllm.v1.dssd.transport",
            DSSD_DIR / "transport.py",
        )
        _load_module(
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
        _load_module(
            "vllm.v1.dssd.worker.resample",
            DSSD_DIR / "worker" / "resample.py",
        )
        coordinator_module = _load_module(
            "vllm.v1.dssd.edge.coordinator",
            DSSD_EDGE_DIR / "coordinator.py",
        )

        yield SimpleNamespace(
            BindVerifierResponse=protocol_module.BindVerifierResponse,
            CreateSessionResponse=protocol_module.CreateSessionResponse,
            VerifyRoundResponse=protocol_module.VerifyRoundResponse,
            DSSDRequestMetrics=metrics_module.DSSDRequestMetrics,
            HTTPDSSDTransport=transport_module.HTTPDSSDTransport,
            DSSDVerifierService=service_module.DSSDVerifierService,
            attach_router=router_module.attach_router,
            DSSDRoundCoordinator=coordinator_module.DSSDRoundCoordinator,
        )
    finally:
        for name in list(sys.modules):
            if name == "vllm" or name.startswith("vllm."):
                if name not in saved_vllm_modules:
                    sys.modules.pop(name, None)
        sys.modules.update(saved_vllm_modules)


@pytest.fixture
def smoke_modules() -> Iterator[SimpleNamespace]:
    with _smoke_modules() as modules:
        yield modules


def test_metrics_track_communication_bytes(smoke_modules):
    metrics = smoke_modules.DSSDRequestMetrics(request_id="req-1")

    metrics.record_uplink(32)
    metrics.record_downlink(128)

    assert metrics.uplink_bytes == 32
    assert metrics.downlink_bytes == 128


def test_metrics_export_request_and_round_summaries(smoke_modules):
    metrics = smoke_modules.DSSDRequestMetrics(request_id="req-metrics")

    metrics.record_uplink(32)
    metrics.record_downlink(128)
    metrics.record_round(
        seq_no=0,
        accepted_count=2,
        draft_latency_ms=1.5,
        verify_latency_ms=2.5,
        round_trip_latency_ms=4.0,
    )
    metrics.record_round(
        seq_no=1,
        accepted_count=1,
        reject_index=1,
        draft_latency_ms=1.0,
        verify_latency_ms=3.0,
        round_trip_latency_ms=4.5,
    )
    metrics.finalize_request(total_latency_ms=12.0)

    exported = metrics.export(
        run_metadata={
            "mode": "dssd",
            "edge_model_id": "edge-model",
            "verifier_model_id": "target-model",
        }
    )

    assert exported["request"]["request_id"] == "req-metrics"
    assert exported["request"]["uplink_bytes"] == 32
    assert exported["request"]["downlink_bytes"] == 128
    assert exported["request"]["accepted_tokens"] == 3
    assert exported["request"]["rejected_rounds"] == 1
    assert exported["request"]["latency_ms"] == {
        "total": 12.0,
        "draft_total": 2.5,
        "verify_total": 5.5,
        "round_trip_total": 8.5,
    }
    assert exported["rounds"] == [
        {
            "seq_no": 0,
            "accepted_count": 2,
            "reject_index": None,
            "draft_latency_ms": 1.5,
            "verify_latency_ms": 2.5,
            "round_trip_latency_ms": 4.0,
        },
        {
            "seq_no": 1,
            "accepted_count": 1,
            "reject_index": 1,
            "draft_latency_ms": 1.0,
            "verify_latency_ms": 3.0,
            "round_trip_latency_ms": 4.5,
        },
    ]
    assert exported["run"] == {
        "mode": "dssd",
        "edge_model_id": "edge-model",
        "verifier_model_id": "target-model",
    }


@pytest.mark.asyncio
async def test_dssd_smoke_runs_two_http_round_trips(smoke_modules):
    class _StubVerifierEngine:
        def __init__(self) -> None:
            self.model_config = SimpleNamespace(model="target-model")
            self.verify_requests = []

        async def dssd_verify_round_async(self, request):
            self.verify_requests.append(request)
            if request.seq_no == 0:
                return smoke_modules.VerifyRoundResponse(
                    verifier_session_id=request.verifier_session_id,
                    seq_no=request.seq_no,
                    accepted_count=1,
                    all_accepted=True,
                    bonus_token_id=9,
                    finished=False,
                )
            assert request.seq_no == 1
            assert request.prefix_delta_token_ids == []
            return smoke_modules.VerifyRoundResponse(
                verifier_session_id=request.verifier_session_id,
                seq_no=request.seq_no,
                accepted_count=1,
                all_accepted=True,
                bonus_token_id=10,
                finished=True,
                finish_reason="stop",
            )

    class _StubEdgeEngine:
        def __init__(self) -> None:
            self.draft_requests = []

        async def dssd_draft_round_async(self, request):
            self.draft_requests.append(request)
            if request.seq_no == 0:
                assert request.committed_token_ids == []
                return SimpleNamespace(
                    draft_token_ids=[3],
                    q_values=[0.7],
                    q_dists_handle="edge-1:0",
                    q_distributions=[[0.3, 0.7]],
                )
            assert request.seq_no == 1
            assert request.committed_token_ids == [3, 9]
            return SimpleNamespace(
                draft_token_ids=[4],
                q_values=[0.6],
                q_dists_handle="edge-1:1",
                q_distributions=[[0.4, 0.6]],
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
            self.default_sampling_params = {
                "temperature": 0.6,
                "top_p": 0.75,
                "top_k": 11,
                "min_p": 0.03,
                "max_tokens": 16,
            }
            self.model_config = SimpleNamespace(max_model_len=128)
            self.override_max_tokens = None

        async def render_chat_request(self, request):
            del request
            return [], [{"prompt_token_ids": [11, 12]}]

        def create_error_response(self, message: str, **kwargs):
            return {"message": message, **kwargs}

    app = FastAPI()
    verifier_engine = _StubVerifierEngine()
    app.state.dssd_verifier_service = smoke_modules.DSSDVerifierService(
        verifier_engine,
        SimpleNamespace(
            dssd_config=SimpleNamespace(enabled=True, role="verifier", gamma=1),
        ),
    )
    smoke_modules.attach_router(app)

    transport = smoke_modules.HTTPDSSDTransport(
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
        edge_engine = _StubEdgeEngine()
        coordinator = smoke_modules.DSSDRoundCoordinator(
            edge_engine=edge_engine,
            transport=transport,
            gamma=1,
        )
        response = await coordinator.create_chat_completion(
            request=_attach_sampling_params(SimpleNamespace(
                stream=False,
                max_tokens=4,
                stop_token_ids=[],
                user="edge-user",
            )),
            raw_request=None,
            serving=_StubServing(),
        )
    finally:
        await transport.aclose()

    assert [request.seq_no for request in edge_engine.draft_requests] == [0, 1]
    assert [request.seq_no for request in verifier_engine.verify_requests] == [0, 1]
    assert response.choices[0].message.content == "tok3|tok9|tok4|tok10"
    assert response.usage.prompt_tokens == 2
    assert response.usage.completion_tokens == 4
    assert response.choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_dssd_smoke_exports_experiment_result_to_raw_request_state(
    smoke_modules,
):
    class _StubVerifierEngine:
        def __init__(self) -> None:
            self.model_config = SimpleNamespace(model="target-model")

        async def dssd_verify_round_async(self, request):
            if request.seq_no == 0:
                return smoke_modules.VerifyRoundResponse(
                    verifier_session_id=request.verifier_session_id,
                    seq_no=request.seq_no,
                    accepted_count=1,
                    all_accepted=True,
                    bonus_token_id=9,
                    finished=False,
                )
            return smoke_modules.VerifyRoundResponse(
                verifier_session_id=request.verifier_session_id,
                seq_no=request.seq_no,
                accepted_count=1,
                all_accepted=False,
                reject_index=0,
                reject_target_probs=[0.8, 0.2],
                finished=True,
                finish_reason="stop",
            )

    class _StubEdgeEngine:
        async def dssd_draft_round_async(self, request):
            if request.seq_no == 0:
                return SimpleNamespace(
                    draft_token_ids=[3],
                    q_values=[0.7],
                    q_dists_handle="edge-1:0",
                    q_distributions=[[0.3, 0.7]],
                )
            return SimpleNamespace(
                draft_token_ids=[4],
                q_values=[0.6],
                q_dists_handle="edge-1:1",
                q_distributions=[[0.4, 0.6]],
            )

    class _Coordinator(smoke_modules.DSSDRoundCoordinator):
        def _resample_reject_token(self, *, verify_response, q_distributions):
            del verify_response, q_distributions
            return 42

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
            self.default_sampling_params = {
                "temperature": 0.6,
                "top_p": 0.75,
                "top_k": 11,
                "min_p": 0.03,
                "max_tokens": 16,
            }
            self.model_config = SimpleNamespace(max_model_len=128)
            self.override_max_tokens = None

        async def render_chat_request(self, request):
            del request
            return [], [{"prompt_token_ids": [11, 12]}]

        def create_error_response(self, message: str, **kwargs):
            return {"message": message, **kwargs}

    app = FastAPI()
    app.state.dssd_verifier_service = smoke_modules.DSSDVerifierService(
        _StubVerifierEngine(),
        SimpleNamespace(
            dssd_config=SimpleNamespace(enabled=True, role="verifier", gamma=1),
        ),
    )
    smoke_modules.attach_router(app)

    transport = smoke_modules.HTTPDSSDTransport(
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
    raw_request = SimpleNamespace(state=SimpleNamespace())
    try:
        coordinator = _Coordinator(
            edge_engine=_StubEdgeEngine(),
            transport=transport,
            gamma=1,
        )
        await coordinator.create_chat_completion(
            request=_attach_sampling_params(SimpleNamespace(
                stream=False,
                max_tokens=4,
                stop_token_ids=[],
                user="edge-user",
            )),
            raw_request=raw_request,
            serving=_StubServing(),
        )
    finally:
        await transport.aclose()

    exported = raw_request.state.dssd_experiment_result
    assert exported["request"]["accepted_tokens"] == 2
    assert exported["request"]["rejected_rounds"] == 1
    assert exported["request"]["uplink_bytes"] > 0
    assert exported["request"]["downlink_bytes"] > 0
    assert exported["request"]["latency_ms"]["total"] >= 0.0
    assert [round_entry["accepted_count"] for round_entry in exported["rounds"]] == [
        1,
        1,
    ]
    assert exported["rounds"][1]["reject_index"] == 0
    assert exported["run"] == {
        "mode": "dssd",
        "edge_model_id": "edge-model",
        "verifier_model_id": "target-model",
        "sampling": {
            "temperature": 0.6,
            "top_p": 0.75,
            "top_k": 11,
            "min_p": 0.03,
            "max_tokens": 4,
        },
        "network": {
            "latency_ms": 0.0,
            "bandwidth_mbps": None,
            "jitter_ms": 0.0,
        },
    }


@pytest.mark.asyncio
async def test_dssd_smoke_exports_experiment_result_to_sink_file(
    smoke_modules,
    tmp_path: Path,
):
    class _StubVerifierEngine:
        def __init__(self) -> None:
            self.model_config = SimpleNamespace(model="target-model")

        async def dssd_verify_round_async(self, request):
            return smoke_modules.VerifyRoundResponse(
                verifier_session_id=request.verifier_session_id,
                seq_no=request.seq_no,
                accepted_count=1,
                all_accepted=True,
                bonus_token_id=9,
                finished=True,
                finish_reason="stop",
            )

    class _StubEdgeEngine:
        async def dssd_draft_round_async(self, request):
            del request
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
            self.default_sampling_params = {
                "temperature": 0.6,
                "top_p": 0.75,
                "top_k": 11,
                "min_p": 0.03,
                "max_tokens": 16,
            }
            self.model_config = SimpleNamespace(max_model_len=128)
            self.override_max_tokens = None

        async def render_chat_request(self, request):
            del request
            return [], [{"prompt_token_ids": [11, 12]}]

        def create_error_response(self, message: str, **kwargs):
            return {"message": message, **kwargs}

    app = FastAPI()
    app.state.dssd_verifier_service = smoke_modules.DSSDVerifierService(
        _StubVerifierEngine(),
        SimpleNamespace(
            dssd_config=SimpleNamespace(enabled=True, role="verifier", gamma=1),
        ),
    )
    smoke_modules.attach_router(app)

    transport = smoke_modules.HTTPDSSDTransport(
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
    sink_path = tmp_path / "dssd-results.jsonl"
    try:
        coordinator = smoke_modules.DSSDRoundCoordinator(
            edge_engine=_StubEdgeEngine(),
            transport=transport,
            gamma=1,
            experiment_result_path=str(sink_path),
        )
        await coordinator.create_chat_completion(
            request=_attach_sampling_params(SimpleNamespace(
                stream=False,
                max_tokens=4,
                stop_token_ids=[],
                user="edge-user",
            )),
            raw_request=SimpleNamespace(state=SimpleNamespace()),
            serving=_StubServing(),
        )
    finally:
        await transport.aclose()

    assert sink_path.exists()
    lines = sink_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    exported = json.loads(lines[0])
    assert exported["request"]["accepted_tokens"] == 1
    assert exported["run"]["mode"] == "dssd"


@pytest.mark.asyncio
async def test_dssd_smoke_forwards_residual_token_as_prefix_delta(smoke_modules):
    class _StubVerifierEngine:
        def __init__(self) -> None:
            self.model_config = SimpleNamespace(model="target-model")
            self.verify_requests = []

        async def dssd_verify_round_async(self, request):
            self.verify_requests.append(request)
            if request.seq_no == 0:
                return smoke_modules.VerifyRoundResponse(
                    verifier_session_id=request.verifier_session_id,
                    seq_no=request.seq_no,
                    accepted_count=1,
                    all_accepted=False,
                    reject_index=1,
                    reject_target_probs=[0.8, 0.2],
                    finished=False,
                )
            assert request.seq_no == 1
            assert request.prefix_delta_token_ids == [42]
            return smoke_modules.VerifyRoundResponse(
                verifier_session_id=request.verifier_session_id,
                seq_no=request.seq_no,
                accepted_count=1,
                all_accepted=True,
                bonus_token_id=10,
                finished=True,
                finish_reason="stop",
            )

    class _StubEdgeEngine:
        def __init__(self) -> None:
            self.draft_requests = []

        async def dssd_draft_round_async(self, request):
            self.draft_requests.append(request)
            if request.seq_no == 0:
                assert request.committed_token_ids == []
                return SimpleNamespace(
                    draft_token_ids=[3, 4],
                    q_values=[0.7, 0.8],
                    q_dists_handle="edge-1:0",
                    q_distributions=[[0.3, 0.7], [0.2, 0.8]],
                )
            assert request.seq_no == 1
            assert request.committed_token_ids == [3, 42]
            return SimpleNamespace(
                draft_token_ids=[5],
                q_values=[0.6],
                q_dists_handle="edge-1:1",
                q_distributions=[[0.4, 0.6]],
            )

    class _Coordinator(smoke_modules.DSSDRoundCoordinator):
        def _resample_reject_token(self, *, verify_response, q_distributions):
            del verify_response, q_distributions
            return 42

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
            self.default_sampling_params = {
                "temperature": 0.6,
                "top_p": 0.75,
                "top_k": 11,
                "min_p": 0.03,
                "max_tokens": 16,
            }
            self.model_config = SimpleNamespace(max_model_len=128)
            self.override_max_tokens = None

        async def render_chat_request(self, request):
            del request
            return [], [{"prompt_token_ids": [11, 12]}]

        def create_error_response(self, message: str, **kwargs):
            return {"message": message, **kwargs}

    app = FastAPI()
    verifier_engine = _StubVerifierEngine()
    app.state.dssd_verifier_service = smoke_modules.DSSDVerifierService(
        verifier_engine,
        SimpleNamespace(
            dssd_config=SimpleNamespace(enabled=True, role="verifier", gamma=1),
        ),
    )
    smoke_modules.attach_router(app)

    transport = smoke_modules.HTTPDSSDTransport(
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
        edge_engine = _StubEdgeEngine()
        coordinator = _Coordinator(
            edge_engine=edge_engine,
            transport=transport,
            gamma=1,
        )
        response = await coordinator.create_chat_completion(
            request=_attach_sampling_params(SimpleNamespace(
                stream=False,
                max_tokens=8,
                stop_token_ids=[],
                user="edge-user",
            )),
            raw_request=None,
            serving=_StubServing(),
        )
    finally:
        await transport.aclose()

    assert [request.seq_no for request in edge_engine.draft_requests] == [0, 1]
    assert [request.seq_no for request in verifier_engine.verify_requests] == [0, 1]
    assert response.choices[0].message.content == "tok3|tok42|tok5|tok10"
    assert response.choices[0].finish_reason == "stop"
