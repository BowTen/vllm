# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import msgspec
import pytest


ROOT = Path(__file__).resolve().parents[3]
VLLM_DIR = ROOT / "vllm"
DSSD_EDGE_DIR = VLLM_DIR / "v1" / "dssd" / "edge"


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
class _DeltaMessage:
    role: str | None = None
    content: str | None = None


@dataclass
class _ChatCompletionResponseStreamChoice:
    index: int
    delta: _DeltaMessage
    logprobs: object | None = None
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    token_ids: list[int] | None = None


@dataclass
class _ChatCompletionStreamResponse:
    id: str
    model: str
    choices: list[_ChatCompletionResponseStreamChoice]
    usage: _UsageInfo | None = None
    created: int = 0
    object: str = "chat.completion.chunk"
    prompt_token_ids: list[int] | None = None

    def model_dump_json(self, exclude_unset: bool = False) -> str:
        del exclude_unset
        return json.dumps(self, default=lambda obj: obj.__dict__)


@dataclass
class _ChatCompletionResponse:
    id: str
    model: str
    choices: list[_ChatCompletionResponseChoice]
    usage: _UsageInfo
    created: int = 0
    object: str = "chat.completion"


coordinator_module = None
protocol_module = None
DSSDRoundCoordinator = None
BindVerifierResponse = None
CreateSessionResponse = None
DraftRoundRequest = None
VerifyRoundResponse = None
SamplingParams = None


@contextmanager
def _round_coordinator_modules() -> Iterator[SimpleNamespace]:
    saved_vllm_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "vllm" or name.startswith("vllm.")
    }

    try:
        _install_package_stub("vllm", VLLM_DIR)
        _install_package_stub("vllm.v1", VLLM_DIR / "v1")
        _install_package_stub("vllm.v1.dssd", VLLM_DIR / "v1" / "dssd")
        _install_package_stub("vllm.v1.dssd.edge", DSSD_EDGE_DIR)
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

        sampling_params_module = types.ModuleType("vllm.sampling_params")

        class _SamplingParams(msgspec.Struct, omit_defaults=True):
            temperature: float = 1.0
            top_p: float = 1.0
            top_k: int = 0
            min_p: float = 0.0
            seed: int | None = None
            max_tokens: int | None = 16

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

        def _should_include_usage(stream_options, enable_force_include_usage):
            del stream_options, enable_force_include_usage
            return False, False

        entrypoints_utils_module.should_include_usage = _should_include_usage
        sys.modules["vllm.entrypoints.utils"] = entrypoints_utils_module

        chat_protocol_module = types.ModuleType(
            "vllm.entrypoints.openai.chat_completion.protocol"
        )
        chat_protocol_module.ChatMessage = _ChatMessage
        chat_protocol_module.ChatCompletionResponseChoice = (
            _ChatCompletionResponseChoice
        )
        chat_protocol_module.ChatCompletionResponse = _ChatCompletionResponse
        chat_protocol_module.ChatCompletionResponseStreamChoice = (
            _ChatCompletionResponseStreamChoice
        )
        chat_protocol_module.ChatCompletionStreamResponse = (
            _ChatCompletionStreamResponse
        )
        sys.modules["vllm.entrypoints.openai.chat_completion.protocol"] = (
            chat_protocol_module
        )

        engine_protocol_module = types.ModuleType(
            "vllm.entrypoints.openai.engine.protocol"
        )
        engine_protocol_module.ErrorResponse = dict
        engine_protocol_module.DeltaMessage = _DeltaMessage
        engine_protocol_module.UsageInfo = _UsageInfo
        sys.modules["vllm.entrypoints.openai.engine.protocol"] = (
            engine_protocol_module
        )

        coordinator_module = _load_module(
            "vllm.v1.dssd.edge.coordinator",
            DSSD_EDGE_DIR / "coordinator.py",
        )
        protocol_module = _load_module(
            "vllm.v1.dssd.protocol",
            VLLM_DIR / "v1" / "dssd" / "protocol.py",
        )

        yield SimpleNamespace(
            coordinator_module=coordinator_module,
            protocol_module=protocol_module,
            DSSDRoundCoordinator=coordinator_module.DSSDRoundCoordinator,
            BindVerifierResponse=protocol_module.BindVerifierResponse,
            CreateSessionResponse=protocol_module.CreateSessionResponse,
            DraftRoundRequest=protocol_module.DraftRoundRequest,
            VerifyRoundResponse=protocol_module.VerifyRoundResponse,
            SamplingParams=sampling_params_module.SamplingParams,
        )
    finally:
        for name in list(sys.modules):
            if name == "vllm" or name.startswith("vllm."):
                if name not in saved_vllm_modules:
                    sys.modules.pop(name, None)
        sys.modules.update(saved_vllm_modules)


@pytest.fixture(scope="module", autouse=True)
def _install_round_coordinator_modules() -> Iterator[None]:
    global coordinator_module
    global protocol_module
    global DSSDRoundCoordinator
    global BindVerifierResponse
    global CreateSessionResponse
    global DraftRoundRequest
    global VerifyRoundResponse
    global SamplingParams

    with _round_coordinator_modules() as modules:
        coordinator_module = modules.coordinator_module
        protocol_module = modules.protocol_module
        DSSDRoundCoordinator = modules.DSSDRoundCoordinator
        BindVerifierResponse = modules.BindVerifierResponse
        CreateSessionResponse = modules.CreateSessionResponse
        DraftRoundRequest = modules.DraftRoundRequest
        VerifyRoundResponse = modules.VerifyRoundResponse
        SamplingParams = modules.SamplingParams
        yield


def _attach_sampling_params(
    request,
    *,
    temperature: float = 0.6,
    top_p: float = 0.75,
    top_k: int = 11,
    min_p: float = 0.03,
    seed: int = 999,
):
    def _to_sampling_params(max_tokens, default_sampling_params):
        del default_sampling_params
        return SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            seed=seed,
            max_tokens=max_tokens,
        )

    request.to_sampling_params = _to_sampling_params
    return request


def test_round_coordinator_commits_bonus_token_on_full_accept():
    coordinator = DSSDRoundCoordinator(edge_engine=None, transport=None)
    response = VerifyRoundResponse(
        verifier_session_id="vs-1",
        seq_no=0,
        accepted_count=2,
        all_accepted=True,
        bonus_token_id=99,
        reject_index=None,
        reject_target_probs=None,
        finished=False,
        finish_reason=None,
    )

    committed = coordinator._build_committed_tokens(
        [10, 11],
        response,
        resampled_token=None,
    )

    assert committed == [10, 11, 99]


def test_round_coordinator_uses_resampled_token_after_reject():
    coordinator = DSSDRoundCoordinator(edge_engine=None, transport=None)
    response = VerifyRoundResponse(
        verifier_session_id="vs-1",
        seq_no=0,
        accepted_count=1,
        all_accepted=False,
        bonus_token_id=None,
        reject_index=1,
        reject_target_probs=[0.2, 0.8],
        finished=False,
        finish_reason=None,
    )

    committed = coordinator._build_committed_tokens(
        [10, 11],
        response,
        resampled_token=42,
    )

    assert committed == [10, 42]


def test_round_coordinator_enters_dssd_control_plane_before_failing_closed():
    class _StubTransport:
        def __init__(self) -> None:
            self.bind_calls = []
            self.session_calls = []

        async def bind_verifier(self, request):
            self.bind_calls.append(request)
            return BindVerifierResponse(
                binding_id="bind-1",
                protocol_version=request.protocol_version,
                verifier_model_id="target-model",
                tokenizer_hash=request.tokenizer_hash,
                vocab_hash=request.vocab_hash,
                supported_gamma_max=request.supported_gamma_max,
                capabilities={"transport": "http"},
            )

        async def create_session(self, request):
            self.session_calls.append(request)
            return CreateSessionResponse(
                verifier_session_id="vs-1",
                accepted_prompt_len=len(request.prompt_token_ids),
                expires_at=None,
            )

    class _StubServing:
        def __init__(self) -> None:
            self.models = SimpleNamespace(model_name=lambda *_args, **_kwargs: "edge-model")
            self.default_sampling_params = {
                "temperature": 0.6,
                "top_p": 0.75,
                "top_k": 11,
                "min_p": 0.03,
                "max_tokens": 16,
            }
            self.model_config = SimpleNamespace(max_model_len=128)

        async def render_chat_request(self, request):
            del request
            return [], [{"prompt_token_ids": [1, 2, 3]}]

        def create_error_response(self, message: str, **kwargs):
            return {"message": message, **kwargs}

    _StubServing.default_sampling_params = {
        "temperature": 0.6,
        "top_p": 0.75,
        "top_k": 11,
        "min_p": 0.03,
        "max_tokens": 16,
    }
    _StubServing.model_config = SimpleNamespace(max_model_len=128)

    coordinator = DSSDRoundCoordinator(
        edge_engine=SimpleNamespace(),
        transport=_StubTransport(),
    )

    request = SimpleNamespace(
        stream=False,
        max_tokens=16,
        stop_token_ids=[2],
        user="edge-user",
    )

    import asyncio

    result = asyncio.run(
        coordinator.create_chat_completion(
            request=request,
            raw_request=SimpleNamespace(),
            serving=_StubServing(),
        )
    )

    assert len(coordinator.transport.bind_calls) == 0
    assert len(coordinator.transport.session_calls) == 0
    assert result["status_code"] == HTTPStatus.NOT_IMPLEMENTED
    assert "draft_round" in result["message"]


def test_round_coordinator_rejects_unresolvable_sampling_params():
    class _StubTransport:
        def __init__(self) -> None:
            self.bind_calls = []
            self.session_calls = []

        async def bind_verifier(self, request):
            self.bind_calls.append(request)
            raise AssertionError("bind_verifier should not be called")

        async def create_session(self, request):
            self.session_calls.append(request)
            raise AssertionError("create_session should not be called")

    class _StubEdgeEngine:
        async def dssd_draft_round_async(self, request):
            raise AssertionError("dssd_draft_round_async should not be called")

    class _StubServing:
        def __init__(self) -> None:
            self.models = SimpleNamespace(model_name=lambda *_args, **_kwargs: "edge-model")
            self.default_sampling_params = {
                "temperature": 0.6,
                "top_p": 0.75,
                "top_k": 11,
                "min_p": 0.03,
                "max_tokens": 16,
            }
            self.model_config = SimpleNamespace(max_model_len=128)

        async def render_chat_request(self, request):
            del request
            return [], [{"prompt_token_ids": [1, 2, 3]}]

        def create_error_response(self, message: str, **kwargs):
            return {"message": message, **kwargs}

    coordinator = DSSDRoundCoordinator(
        edge_engine=_StubEdgeEngine(),
        transport=_StubTransport(),
    )

    result = asyncio.run(
        coordinator.create_chat_completion(
            request=SimpleNamespace(
                stream=False,
                max_tokens=16,
                stop_token_ids=[],
                user="edge-user",
            ),
            raw_request=None,
            serving=_StubServing(),
        )
    )

    assert result["status_code"] == HTTPStatus.BAD_REQUEST
    assert "resolvable sampling params" in result["message"]
    assert len(coordinator.transport.bind_calls) == 0
    assert len(coordinator.transport.session_calls) == 0


def test_round_coordinator_passes_resolved_sampling_params_to_draft_round():
    draft_requests = []
    to_sampling_params_calls = []

    class _StubTransport:
        async def bind_verifier(self, request):
            return BindVerifierResponse(
                binding_id="bind-1",
                protocol_version=request.protocol_version,
                verifier_model_id="target-model",
                tokenizer_hash=request.tokenizer_hash,
                vocab_hash=request.vocab_hash,
                supported_gamma_max=request.supported_gamma_max,
            )

        async def create_session(self, request):
            return CreateSessionResponse(
                verifier_session_id="vs-1",
                accepted_prompt_len=len(request.prompt_token_ids),
            )

        async def verify_round(self, request):
            return VerifyRoundResponse(
                verifier_session_id=request.verifier_session_id,
                seq_no=request.seq_no,
                accepted_count=1,
                all_accepted=True,
                finished=True,
                finish_reason="stop",
            )

        async def close_session(self, request):
            del request
            return {"closed": True}

    class _StubEdgeEngine:
        async def dssd_draft_round_async(self, request):
            draft_requests.append(request)
            return SimpleNamespace(
                draft_token_ids=[7],
                q_values=[0.9],
                q_dists_handle="handle-1",
                q_distributions=[[0.1, 0.9]],
            )

    class _Tokenizer:
        def decode(self, token_ids, skip_special_tokens=False):
            del skip_special_tokens
            return "|".join(f"tok{token_id}" for token_id in token_ids)

    class _StubServing:
        def __init__(self) -> None:
            self.models = SimpleNamespace(model_name=lambda *_args, **_kwargs: "edge-model")
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
            return [], [{"prompt_token_ids": [11, 12, 13]}]

        def create_error_response(self, message: str, **kwargs):
            return {"message": message, **kwargs}

    request = SimpleNamespace(
        stream=False,
        max_tokens=4,
        max_completion_tokens=None,
        stop_token_ids=[],
        user="edge-user",
    )

    def _to_sampling_params(max_tokens, default_sampling_params):
        to_sampling_params_calls.append((max_tokens, dict(default_sampling_params)))
        return SamplingParams(
            temperature=default_sampling_params["temperature"],
            top_p=default_sampling_params["top_p"],
            top_k=default_sampling_params["top_k"],
            min_p=default_sampling_params["min_p"],
            seed=999,
            max_tokens=max_tokens,
        )

    request.to_sampling_params = _to_sampling_params

    coordinator = DSSDRoundCoordinator(
        edge_engine=_StubEdgeEngine(),
        transport=_StubTransport(),
    )

    response = asyncio.run(
        coordinator.create_chat_completion(
            request=request,
            raw_request=None,
            serving=_StubServing(),
        )
    )

    assert response.choices[0].message.content == "tok7"
    assert to_sampling_params_calls == [
        (
            4,
            {
                "temperature": 0.6,
                "top_p": 0.75,
                "top_k": 11,
                "min_p": 0.03,
                "max_tokens": 16,
            },
        )
    ]
    assert len(draft_requests) == 1
    assert draft_requests[0].sampling_params == SamplingParams(
        temperature=0.6,
        top_p=0.75,
        top_k=11,
        min_p=0.03,
        seed=999,
        max_tokens=4,
    )


def test_round_coordinator_rejects_incompatible_verifier_gamma():
    class _StubTransport:
        def __init__(self) -> None:
            self.session_calls = []

        async def bind_verifier(self, request):
            return BindVerifierResponse(
                binding_id="bind-1",
                protocol_version=request.protocol_version,
                verifier_model_id="target-model",
                tokenizer_hash=request.tokenizer_hash,
                vocab_hash=request.vocab_hash,
                supported_gamma_max=1,
            )

        async def create_session(self, request):
            self.session_calls.append(request)
            raise AssertionError("create_session should not be called")

    class _StubServing:
        def __init__(self) -> None:
            self.models = SimpleNamespace(model_name=lambda *_args, **_kwargs: "edge-model")
            self.default_sampling_params = {
                "temperature": 0.6,
                "top_p": 0.75,
                "top_k": 11,
                "min_p": 0.03,
                "max_tokens": 16,
            }
            self.model_config = SimpleNamespace(max_model_len=128)

        async def render_chat_request(self, request):
            del request
            return [], [{"prompt_token_ids": [1, 2, 3]}]

        def create_error_response(self, message: str, **kwargs):
            return {"message": message, **kwargs}

    coordinator = DSSDRoundCoordinator(
        edge_engine=SimpleNamespace(dssd_draft_round_async=lambda *_args, **_kwargs: None),
        transport=_StubTransport(),
        gamma=4,
    )

    result = asyncio.run(
        coordinator.create_chat_completion(
            request=_attach_sampling_params(SimpleNamespace(
                stream=False,
                max_tokens=16,
                stop_token_ids=[],
                user="edge-user",
            )),
            raw_request=None,
            serving=_StubServing(),
        )
    )

    assert result["status_code"] == HTTPStatus.SERVICE_UNAVAILABLE
    assert "supported_gamma_max" in result["message"]


def test_round_coordinator_returns_response_on_full_accept():
    class _StubTransport:
        def __init__(self) -> None:
            self.close_calls = []

        async def bind_verifier(self, request):
            return BindVerifierResponse(
                binding_id="bind-1",
                protocol_version=request.protocol_version,
                verifier_model_id="target-model",
                tokenizer_hash=request.tokenizer_hash,
                vocab_hash=request.vocab_hash,
                supported_gamma_max=request.supported_gamma_max,
            )

        async def create_session(self, request):
            return CreateSessionResponse(
                verifier_session_id="vs-1",
                accepted_prompt_len=len(request.prompt_token_ids),
            )

        async def verify_round(self, request):
            assert request.q_values == [0.7, 0.8]
            return VerifyRoundResponse(
                verifier_session_id=request.verifier_session_id,
                seq_no=request.seq_no,
                accepted_count=2,
                all_accepted=True,
                bonus_token_id=9,
            )

        async def close_session(self, request):
            self.close_calls.append(request)
            return {"closed": True}

    class _StubEdgeEngine:
        async def dssd_draft_round_async(self, request):
            assert request.gamma == 2
            assert request.prompt_token_ids == [11, 12]
            return SimpleNamespace(
                draft_token_ids=[3, 4],
                q_values=[0.7, 0.8],
                q_dists_handle="handle-1",
                q_distributions=[[0.3, 0.7], [0.2, 0.8]],
            )

    class _Tokenizer:
        def decode(self, token_ids, skip_special_tokens=False):
            del skip_special_tokens
            return "|".join(f"tok{token_id}" for token_id in token_ids)

    class _StubServing:
        def __init__(self) -> None:
            self.models = SimpleNamespace(model_name=lambda *_args, **_kwargs: "edge-model")
            self.renderer = SimpleNamespace(
                get_tokenizer=lambda: _Tokenizer(),
                tokenizer=_Tokenizer(),
            )

        async def render_chat_request(self, request):
            del request
            return [], [{"prompt_token_ids": [11, 12]}]

        def create_error_response(self, message: str, **kwargs):
            return {"message": message, **kwargs}

        def create_streaming_error_response(self, message: str, **kwargs):
            return str(self.create_error_response(message, **kwargs))

    _StubServing.default_sampling_params = {
        "temperature": 0.6,
        "top_p": 0.75,
        "top_k": 11,
        "min_p": 0.03,
        "max_tokens": 16,
    }
    _StubServing.model_config = SimpleNamespace(max_model_len=128)

    transport = _StubTransport()
    class _Coordinator(DSSDRoundCoordinator):
        def _resample_reject_token(self, *, verify_response, q_distributions):
            del verify_response, q_distributions
            return 42

    coordinator = _Coordinator(
        edge_engine=_StubEdgeEngine(),
        transport=transport,
        gamma=2,
    )

    response = asyncio.run(
        coordinator.create_chat_completion(
            request=_attach_sampling_params(SimpleNamespace(
                stream=False,
                max_tokens=2,
                stop_token_ids=[],
                user="edge-user",
            )),
            raw_request=None,
            serving=_StubServing(),
        )
    )

    assert response.model == "edge-model"
    assert response.choices[0].message.content == "tok3|tok4"
    assert response.choices[0].finish_reason == "length"
    assert response.usage.prompt_tokens == 2
    assert response.usage.completion_tokens == 2
    assert len(transport.close_calls) == 1


def test_round_coordinator_returns_response_after_reject_resample():
    class _StubTransport:
        def __init__(self) -> None:
            self.close_calls = []

        async def bind_verifier(self, request):
            return BindVerifierResponse(
                binding_id="bind-1",
                protocol_version=request.protocol_version,
                verifier_model_id="target-model",
                tokenizer_hash=request.tokenizer_hash,
                vocab_hash=request.vocab_hash,
                supported_gamma_max=request.supported_gamma_max,
            )

        async def create_session(self, request):
            return CreateSessionResponse(
                verifier_session_id="vs-1",
                accepted_prompt_len=len(request.prompt_token_ids),
            )

        async def verify_round(self, request):
            return VerifyRoundResponse(
                verifier_session_id=request.verifier_session_id,
                seq_no=request.seq_no,
                accepted_count=1,
                all_accepted=False,
                reject_index=1,
                reject_target_probs=[0.9, 0.1],
            )

        async def close_session(self, request):
            self.close_calls.append(request)
            return {"closed": True}

    class _StubEdgeEngine:
        async def dssd_draft_round_async(self, request):
            del request
            return SimpleNamespace(
                draft_token_ids=[1, 0],
                q_values=[0.8, 0.8],
                q_dists_handle="handle-1",
                q_distributions=[[0.2, 0.8], [0.8, 0.2]],
            )

    class _Tokenizer:
        def decode(self, token_ids, skip_special_tokens=False):
            del skip_special_tokens
            return "|".join(f"tok{token_id}" for token_id in token_ids)

    class _StubServing:
        def __init__(self) -> None:
            self.models = SimpleNamespace(model_name=lambda *_args, **_kwargs: "edge-model")
            self.renderer = SimpleNamespace(
                get_tokenizer=lambda: _Tokenizer(),
                tokenizer=_Tokenizer(),
            )

        async def render_chat_request(self, request):
            del request
            return [], [{"prompt_token_ids": [11, 12]}]

        def create_error_response(self, message: str, **kwargs):
            return {"message": message, **kwargs}

    _StubServing.default_sampling_params = {
        "temperature": 0.6,
        "top_p": 0.75,
        "top_k": 11,
        "min_p": 0.03,
        "max_tokens": 16,
    }
    _StubServing.model_config = SimpleNamespace(max_model_len=128)

    transport = _StubTransport()
    coordinator = DSSDRoundCoordinator(
        edge_engine=_StubEdgeEngine(),
        transport=transport,
        gamma=2,
    )

    response = asyncio.run(
        coordinator.create_chat_completion(
            request=_attach_sampling_params(SimpleNamespace(
                stream=False,
                max_tokens=2,
                stop_token_ids=[],
                user="edge-user",
            )),
            raw_request=None,
            serving=_StubServing(),
        )
    )

    assert response.model == "edge-model"
    assert response.choices[0].message.content == "tok1|tok0"
    assert response.usage.completion_tokens == 2
    assert response.choices[0].finish_reason == "length"
    assert len(transport.close_calls) == 1


def test_round_coordinator_streams_only_verified_delta_after_reject_resample():
    class _StubTransport:
        def __init__(self) -> None:
            self.close_calls = []

        async def bind_verifier(self, request):
            return BindVerifierResponse(
                binding_id="bind-1",
                protocol_version=request.protocol_version,
                verifier_model_id="target-model",
                tokenizer_hash=request.tokenizer_hash,
                vocab_hash=request.vocab_hash,
                supported_gamma_max=request.supported_gamma_max,
            )

        async def create_session(self, request):
            return CreateSessionResponse(
                verifier_session_id="vs-1",
                accepted_prompt_len=len(request.prompt_token_ids),
            )

        async def verify_round(self, request):
            if request.seq_no == 0:
                return VerifyRoundResponse(
                    verifier_session_id=request.verifier_session_id,
                    seq_no=0,
                    accepted_count=1,
                    all_accepted=False,
                    reject_index=1,
                    reject_target_probs=[0.9, 0.1],
                )
            assert request.seq_no == 1
            return VerifyRoundResponse(
                verifier_session_id=request.verifier_session_id,
                seq_no=1,
                accepted_count=2,
                all_accepted=True,
                bonus_token_id=8,
                finished=True,
                finish_reason="stop",
            )

        async def close_session(self, request):
            self.close_calls.append(request)
            return {"closed": True}

    class _StubEdgeEngine:
        async def dssd_draft_round_async(self, request):
            if request.seq_no == 0:
                assert request.committed_token_ids == []
                return SimpleNamespace(
                    draft_token_ids=[3, 4],
                    q_values=[0.7, 0.8],
                    q_dists_handle="handle-1",
                    q_distributions=[[0.3, 0.7], [0.2, 0.8]],
                )
            assert request.seq_no == 1
            assert request.committed_token_ids == [3, 42]
            return SimpleNamespace(
                draft_token_ids=[5, 6],
                q_values=[0.6, 0.5],
                q_dists_handle="handle-2",
                q_distributions=[[0.4, 0.6], [0.5, 0.5]],
            )

    class _Coordinator(DSSDRoundCoordinator):
        def _resample_reject_token(self, *, verify_response, q_distributions):
            del verify_response, q_distributions
            return 42

    class _Tokenizer:
        def decode(self, token_ids, skip_special_tokens=False):
            del skip_special_tokens
            return "|".join(f"tok{token_id}" for token_id in token_ids)

    class _StubServing:
        def __init__(self) -> None:
            self.models = SimpleNamespace(model_name=lambda *_args, **_kwargs: "edge-model")
            self.renderer = SimpleNamespace(
                get_tokenizer=lambda: _Tokenizer(),
                tokenizer=_Tokenizer(),
            )

        async def render_chat_request(self, request):
            del request
            return [], [{"prompt_token_ids": [11, 12]}]

        def create_error_response(self, message: str, **kwargs):
            return {"message": message, **kwargs}

    _StubServing.default_sampling_params = {
        "temperature": 0.6,
        "top_p": 0.75,
        "top_k": 11,
        "min_p": 0.03,
        "max_tokens": 16,
    }
    _StubServing.model_config = SimpleNamespace(max_model_len=128)

    transport = _StubTransport()
    coordinator = _Coordinator(
        edge_engine=_StubEdgeEngine(),
        transport=transport,
        gamma=2,
    )
    coordinator._resample_reject_token = lambda **_kwargs: 42  # type: ignore[method-assign]

    async def _collect():
        serving = _StubServing()
        prepared = await coordinator._prepare_chat_completion(
            request=_attach_sampling_params(SimpleNamespace(
                stream=True,
                max_tokens=8,
                stop_token_ids=[],
                user="edge-user",
                skip_special_tokens=True,
            )),
            raw_request=None,
            serving=serving,
        )
        assert not isinstance(prepared, dict)
        stream = coordinator._iter_verified_round_deltas(
            request=_attach_sampling_params(SimpleNamespace(
                stream=True,
                max_tokens=8,
                stop_token_ids=[],
                user="edge-user",
                skip_special_tokens=True,
            )),
            raw_request=None,
            serving=serving,
            prompt_token_ids=prepared.prompt_token_ids,
            resolved_sampling_params=prepared.resolved_sampling_params,
            request_id=prepared.request_id,
            binding=prepared.binding,
        )
        return [item async for item in stream]

    import asyncio

    deltas = asyncio.run(_collect())

    assert [delta.seq_no for delta in deltas] == [0, 1]
    assert [delta.delta_token_ids for delta in deltas] == [[3, 42], [5, 6, 8]]
    assert deltas[-1].finished is True
    assert deltas[-1].finish_reason == "stop"
    assert len(transport.close_calls) == 1


def test_round_coordinator_stream_iterator_fail_closed_on_seq_mismatch():
    class _StubTransport:
        def __init__(self) -> None:
            self.close_calls = []

        async def bind_verifier(self, request):
            return BindVerifierResponse(
                binding_id="bind-1",
                protocol_version=request.protocol_version,
                verifier_model_id="target-model",
                tokenizer_hash=request.tokenizer_hash,
                vocab_hash=request.vocab_hash,
                supported_gamma_max=request.supported_gamma_max,
            )

        async def create_session(self, request):
            return CreateSessionResponse(
                verifier_session_id="vs-1",
                accepted_prompt_len=len(request.prompt_token_ids),
            )

        async def verify_round(self, request):
            del request
            return VerifyRoundResponse(
                verifier_session_id="vs-1",
                seq_no=1,
                accepted_count=1,
                all_accepted=True,
            )

        async def close_session(self, request):
            self.close_calls.append(request)
            return {"closed": True}

    class _StubEdgeEngine:
        async def dssd_draft_round_async(self, request):
            del request
            return SimpleNamespace(
                draft_token_ids=[3],
                q_values=[0.7],
                q_dists_handle="handle-1",
                q_distributions=[[0.3, 0.7]],
            )

    class _Tokenizer:
        def decode(self, token_ids, skip_special_tokens=False):
            del skip_special_tokens
            return "|".join(f"tok{token_id}" for token_id in token_ids)

    class _StubServing:
        def __init__(self) -> None:
            self.models = SimpleNamespace(model_name=lambda *_args, **_kwargs: "edge-model")
            self.renderer = SimpleNamespace(
                get_tokenizer=lambda: _Tokenizer(),
                tokenizer=_Tokenizer(),
            )

        async def render_chat_request(self, request):
            del request
            return [], [{"prompt_token_ids": [11, 12]}]

        def create_error_response(self, message: str, **kwargs):
            return {"message": message, **kwargs}

    _StubServing.default_sampling_params = {
        "temperature": 0.6,
        "top_p": 0.75,
        "top_k": 11,
        "min_p": 0.03,
        "max_tokens": 16,
    }
    _StubServing.model_config = SimpleNamespace(max_model_len=128)

    transport = _StubTransport()
    coordinator = DSSDRoundCoordinator(
        edge_engine=_StubEdgeEngine(),
        transport=transport,
    )

    async def _collect():
        serving = _StubServing()
        prepared = await coordinator._prepare_chat_completion(
            request=_attach_sampling_params(SimpleNamespace(
                stream=True,
                max_tokens=8,
                stop_token_ids=[],
                user="edge-user",
                skip_special_tokens=True,
            )),
            raw_request=None,
            serving=serving,
        )
        assert not isinstance(prepared, dict)
        stream = coordinator._iter_verified_round_deltas(
            request=_attach_sampling_params(SimpleNamespace(
                stream=True,
                max_tokens=8,
                stop_token_ids=[],
                user="edge-user",
                skip_special_tokens=True,
            )),
            raw_request=None,
            serving=serving,
            prompt_token_ids=prepared.prompt_token_ids,
            resolved_sampling_params=prepared.resolved_sampling_params,
            request_id=prepared.request_id,
            binding=prepared.binding,
        )
        return [item async for item in stream]

    import asyncio

    with pytest.raises(ValueError, match="mismatched seq_no"):
        asyncio.run(_collect())

    assert len(transport.close_calls) == 1


def test_round_coordinator_applies_stop_token_before_response():
    class _StubTransport:
        def __init__(self) -> None:
            self.close_calls = []

        async def bind_verifier(self, request):
            return BindVerifierResponse(
                binding_id="bind-1",
                protocol_version=request.protocol_version,
                verifier_model_id="target-model",
                tokenizer_hash=request.tokenizer_hash,
                vocab_hash=request.vocab_hash,
                supported_gamma_max=request.supported_gamma_max,
            )

        async def create_session(self, request):
            return CreateSessionResponse(
                verifier_session_id="vs-1",
                accepted_prompt_len=len(request.prompt_token_ids),
            )

        async def verify_round(self, request):
            del request
            return VerifyRoundResponse(
                verifier_session_id="vs-1",
                seq_no=0,
                accepted_count=2,
                all_accepted=True,
                bonus_token_id=9,
            )

        async def close_session(self, request):
            self.close_calls.append(request)
            return {"closed": True}

    class _StubEdgeEngine:
        async def dssd_draft_round_async(self, request):
            del request
            return SimpleNamespace(
                draft_token_ids=[3, 2],
                q_values=[0.8, 0.6],
                q_dists_handle="handle-1",
                q_distributions=[[0.2, 0.8], [0.4, 0.6]],
            )

    class _Tokenizer:
        def decode(self, token_ids, skip_special_tokens=False):
            del skip_special_tokens
            return "|".join(f"tok{token_id}" for token_id in token_ids)

    class _StubServing:
        def __init__(self) -> None:
            self.models = SimpleNamespace(model_name=lambda *_args, **_kwargs: "edge-model")
            self.renderer = SimpleNamespace(
                get_tokenizer=lambda: _Tokenizer(),
                tokenizer=_Tokenizer(),
            )

        async def render_chat_request(self, request):
            del request
            return [], [{"prompt_token_ids": [11, 12]}]

        def create_error_response(self, message: str, **kwargs):
            return {"message": message, **kwargs}

    _StubServing.default_sampling_params = {
        "temperature": 0.6,
        "top_p": 0.75,
        "top_k": 11,
        "min_p": 0.03,
        "max_tokens": 16,
    }
    _StubServing.model_config = SimpleNamespace(max_model_len=128)

    transport = _StubTransport()
    coordinator = DSSDRoundCoordinator(
        edge_engine=_StubEdgeEngine(),
        transport=transport,
        gamma=2,
    )

    response = asyncio.run(
        coordinator.create_chat_completion(
            request=_attach_sampling_params(SimpleNamespace(
                stream=False,
                max_tokens=4,
                stop_token_ids=[2],
                user="edge-user",
            )),
            raw_request=None,
            serving=_StubServing(),
        )
    )

    assert response.model == "edge-model"
    assert response.choices[0].message.content == "tok3"
    assert response.choices[0].finish_reason == "stop"
    assert len(transport.close_calls) == 1


def test_round_coordinator_rejects_missing_q_distribution_fail_closed():
    class _StubTransport:
        def __init__(self) -> None:
            self.close_calls = []

        async def bind_verifier(self, request):
            return BindVerifierResponse(
                binding_id="bind-1",
                protocol_version=request.protocol_version,
                verifier_model_id="target-model",
                tokenizer_hash=request.tokenizer_hash,
                vocab_hash=request.vocab_hash,
                supported_gamma_max=request.supported_gamma_max,
            )

        async def create_session(self, request):
            return CreateSessionResponse(
                verifier_session_id="vs-1",
                accepted_prompt_len=len(request.prompt_token_ids),
            )

        async def verify_round(self, request):
            del request
            return VerifyRoundResponse(
                verifier_session_id="vs-1",
                seq_no=0,
                accepted_count=1,
                all_accepted=False,
                reject_index=1,
                reject_target_probs=[0.9, 0.1],
            )

        async def close_session(self, request):
            self.close_calls.append(request)
            return {"closed": True}

    class _StubEdgeEngine:
        async def dssd_draft_round_async(self, request):
            del request
            return SimpleNamespace(
                draft_token_ids=[1, 0],
                q_values=[0.8, 0.8],
                q_dists_handle="handle-1",
                q_distributions=None,
            )

    class _StubServing:
        def __init__(self) -> None:
            self.models = SimpleNamespace(model_name=lambda *_args, **_kwargs: "edge-model")
            self.renderer = SimpleNamespace(get_tokenizer=lambda: None, tokenizer=None)

        async def render_chat_request(self, request):
            del request
            return [], [{"prompt_token_ids": [11, 12]}]

        def create_error_response(self, message: str, **kwargs):
            return {"message": message, **kwargs}

    _StubServing.default_sampling_params = {
        "temperature": 0.6,
        "top_p": 0.75,
        "top_k": 11,
        "min_p": 0.03,
        "max_tokens": 16,
    }
    _StubServing.model_config = SimpleNamespace(max_model_len=128)

    transport = _StubTransport()
    coordinator = DSSDRoundCoordinator(
        edge_engine=_StubEdgeEngine(),
        transport=transport,
        gamma=2,
    )

    response = asyncio.run(
        coordinator.create_chat_completion(
            request=_attach_sampling_params(SimpleNamespace(
                stream=False,
                max_tokens=2,
                stop_token_ids=[],
                user="edge-user",
            )),
            raw_request=None,
            serving=_StubServing(),
        )
    )

    assert response["status_code"] == HTTPStatus.INTERNAL_SERVER_ERROR
    assert "resample" in response["message"]
    assert len(transport.close_calls) == 1


def test_round_coordinator_runs_two_full_accept_rounds():
    draft_requests = []
    verify_requests = []

    class _StubTransport:
        def __init__(self) -> None:
            self.close_calls = []

        async def bind_verifier(self, request):
            return BindVerifierResponse(
                binding_id="bind-1",
                protocol_version=request.protocol_version,
                verifier_model_id="target-model",
                tokenizer_hash=request.tokenizer_hash,
                vocab_hash=request.vocab_hash,
                supported_gamma_max=request.supported_gamma_max,
            )

        async def create_session(self, request):
            return CreateSessionResponse(
                verifier_session_id="vs-1",
                accepted_prompt_len=len(request.prompt_token_ids),
            )

        async def verify_round(self, request):
            verify_requests.append(request)
            if request.seq_no == 0:
                return VerifyRoundResponse(
                    verifier_session_id=request.verifier_session_id,
                    seq_no=0,
                    accepted_count=2,
                    all_accepted=True,
                    bonus_token_id=9,
                    finished=False,
                )
            assert request.seq_no == 1
            return VerifyRoundResponse(
                verifier_session_id=request.verifier_session_id,
                seq_no=1,
                accepted_count=2,
                all_accepted=True,
                bonus_token_id=8,
                finished=True,
                finish_reason="stop",
            )

        async def close_session(self, request):
            self.close_calls.append(request)
            return {"closed": True}

    class _StubEdgeEngine:
        async def dssd_draft_round_async(self, request):
            draft_requests.append(request)
            if request.seq_no == 0:
                assert request.committed_token_ids == []
                return SimpleNamespace(
                    draft_token_ids=[3, 4],
                    q_values=[0.7, 0.8],
                    q_dists_handle="handle-1",
                    q_distributions=[[0.3, 0.7], [0.2, 0.8]],
                )
            assert request.seq_no == 1
            assert request.committed_token_ids == [3, 4, 9]
            return SimpleNamespace(
                draft_token_ids=[5, 6],
                q_values=[0.6, 0.5],
                q_dists_handle="handle-2",
                q_distributions=[[0.4, 0.6], [0.5, 0.5]],
            )

    class _Tokenizer:
        def decode(self, token_ids, skip_special_tokens=False):
            del skip_special_tokens
            return "|".join(f"tok{token_id}" for token_id in token_ids)

    class _StubServing:
        def __init__(self) -> None:
            self.models = SimpleNamespace(model_name=lambda *_args, **_kwargs: "edge-model")
            self.renderer = SimpleNamespace(
                get_tokenizer=lambda: _Tokenizer(),
                tokenizer=_Tokenizer(),
            )

        async def render_chat_request(self, request):
            del request
            return [], [{"prompt_token_ids": [11, 12]}]

        def create_error_response(self, message: str, **kwargs):
            return {"message": message, **kwargs}

    _StubServing.default_sampling_params = {
        "temperature": 0.6,
        "top_p": 0.75,
        "top_k": 11,
        "min_p": 0.03,
        "max_tokens": 16,
    }
    _StubServing.model_config = SimpleNamespace(max_model_len=128)

    transport = _StubTransport()
    coordinator = DSSDRoundCoordinator(
        edge_engine=_StubEdgeEngine(),
        transport=transport,
        gamma=2,
    )

    response = asyncio.run(
        coordinator.create_chat_completion(
            request=_attach_sampling_params(SimpleNamespace(
                stream=False,
                max_tokens=8,
                stop_token_ids=[],
                user="edge-user",
            )),
            raw_request=None,
            serving=_StubServing(),
        )
    )

    assert [request.seq_no for request in draft_requests] == [0, 1]
    assert [request.seq_no for request in verify_requests] == [0, 1]
    assert verify_requests[0].prefix_delta_token_ids == []
    assert verify_requests[1].prefix_delta_token_ids == []
    assert response.choices[0].message.content == "tok3|tok4|tok9|tok5|tok6|tok8"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.completion_tokens == 6
    assert len(transport.close_calls) == 1


def test_round_coordinator_sends_resample_token_as_next_round_prefix_delta():
    draft_requests = []
    verify_requests = []

    class _StubTransport:
        def __init__(self) -> None:
            self.close_calls = []

        async def bind_verifier(self, request):
            return BindVerifierResponse(
                binding_id="bind-1",
                protocol_version=request.protocol_version,
                verifier_model_id="target-model",
                tokenizer_hash=request.tokenizer_hash,
                vocab_hash=request.vocab_hash,
                supported_gamma_max=request.supported_gamma_max,
            )

        async def create_session(self, request):
            return CreateSessionResponse(
                verifier_session_id="vs-1",
                accepted_prompt_len=len(request.prompt_token_ids),
            )

        async def verify_round(self, request):
            verify_requests.append(request)
            if request.seq_no == 0:
                return VerifyRoundResponse(
                    verifier_session_id=request.verifier_session_id,
                    seq_no=0,
                    accepted_count=1,
                    all_accepted=False,
                    reject_index=1,
                    reject_target_probs=[0.8, 0.2],
                    finished=False,
                )
            assert request.seq_no == 1
            return VerifyRoundResponse(
                verifier_session_id=request.verifier_session_id,
                seq_no=1,
                accepted_count=2,
                all_accepted=True,
                bonus_token_id=8,
                finished=True,
                finish_reason="stop",
            )

        async def close_session(self, request):
            self.close_calls.append(request)
            return {"closed": True}

    class _StubEdgeEngine:
        async def dssd_draft_round_async(self, request):
            draft_requests.append(request)
            if request.seq_no == 0:
                assert request.committed_token_ids == []
                return SimpleNamespace(
                    draft_token_ids=[3, 4],
                    q_values=[0.7, 0.8],
                    q_dists_handle="handle-1",
                    q_distributions=[[0.3, 0.7], [0.2, 0.8]],
                )
            assert request.seq_no == 1
            assert request.committed_token_ids == [3, 42]
            return SimpleNamespace(
                draft_token_ids=[5, 6],
                q_values=[0.6, 0.5],
                q_dists_handle="handle-2",
                q_distributions=[[0.4, 0.6], [0.5, 0.5]],
            )

    class _Coordinator(DSSDRoundCoordinator):
        def _resample_reject_token(self, *, verify_response, q_distributions):
            del verify_response, q_distributions
            return 42

    class _Tokenizer:
        def decode(self, token_ids, skip_special_tokens=False):
            del skip_special_tokens
            return "|".join(f"tok{token_id}" for token_id in token_ids)

    class _StubServing:
        def __init__(self) -> None:
            self.models = SimpleNamespace(model_name=lambda *_args, **_kwargs: "edge-model")
            self.renderer = SimpleNamespace(
                get_tokenizer=lambda: _Tokenizer(),
                tokenizer=_Tokenizer(),
            )

        async def render_chat_request(self, request):
            del request
            return [], [{"prompt_token_ids": [11, 12]}]

        def create_error_response(self, message: str, **kwargs):
            return {"message": message, **kwargs}

    _StubServing.default_sampling_params = {
        "temperature": 0.6,
        "top_p": 0.75,
        "top_k": 11,
        "min_p": 0.03,
        "max_tokens": 16,
    }
    _StubServing.model_config = SimpleNamespace(max_model_len=128)

    transport = _StubTransport()
    coordinator = _Coordinator(
        edge_engine=_StubEdgeEngine(),
        transport=transport,
        gamma=2,
    )

    response = asyncio.run(
        coordinator.create_chat_completion(
            request=_attach_sampling_params(SimpleNamespace(
                stream=False,
                max_tokens=8,
                stop_token_ids=[],
                user="edge-user",
            )),
            raw_request=None,
            serving=_StubServing(),
        )
    )

    assert [request.seq_no for request in draft_requests] == [0, 1]
    assert [request.seq_no for request in verify_requests] == [0, 1]
    assert verify_requests[0].prefix_delta_token_ids == []
    assert verify_requests[1].prefix_delta_token_ids == [42]
    assert verify_requests[1].draft_token_ids == [5, 6]
    assert response.choices[0].message.content == "tok3|tok42|tok5|tok6|tok8"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.completion_tokens == 5
    assert len(transport.close_calls) == 1


def test_round_coordinator_fail_closed_on_zero_progress_round():
    class _StubTransport:
        def __init__(self) -> None:
            self.close_calls = []

        async def bind_verifier(self, request):
            return BindVerifierResponse(
                binding_id="bind-1",
                protocol_version=request.protocol_version,
                verifier_model_id="target-model",
                tokenizer_hash=request.tokenizer_hash,
                vocab_hash=request.vocab_hash,
                supported_gamma_max=request.supported_gamma_max,
            )

        async def create_session(self, request):
            return CreateSessionResponse(
                verifier_session_id="vs-1",
                accepted_prompt_len=len(request.prompt_token_ids),
            )

        async def verify_round(self, request):
            del request
            return VerifyRoundResponse(
                verifier_session_id="vs-1",
                seq_no=0,
                accepted_count=0,
                all_accepted=True,
                bonus_token_id=None,
                finished=False,
            )

        async def close_session(self, request):
            self.close_calls.append(request)
            return {"closed": True}

    class _StubEdgeEngine:
        async def dssd_draft_round_async(self, request):
            del request
            return SimpleNamespace(
                draft_token_ids=[3, 4],
                q_values=[0.7, 0.8],
                q_dists_handle="handle-1",
                q_distributions=[[0.3, 0.7], [0.2, 0.8]],
            )

    class _StubServing:
        def __init__(self) -> None:
            self.models = SimpleNamespace(model_name=lambda *_args, **_kwargs: "edge-model")
            self.renderer = SimpleNamespace(get_tokenizer=lambda: None, tokenizer=None)

        async def render_chat_request(self, request):
            del request
            return [], [{"prompt_token_ids": [11, 12]}]

        def create_error_response(self, message: str, **kwargs):
            return {"message": message, **kwargs}

    _StubServing.default_sampling_params = {
        "temperature": 0.6,
        "top_p": 0.75,
        "top_k": 11,
        "min_p": 0.03,
        "max_tokens": 16,
    }
    _StubServing.model_config = SimpleNamespace(max_model_len=128)

    transport = _StubTransport()
    coordinator = DSSDRoundCoordinator(
        edge_engine=_StubEdgeEngine(),
        transport=transport,
        gamma=2,
    )

    response = asyncio.run(
        coordinator.create_chat_completion(
            request=_attach_sampling_params(SimpleNamespace(
                stream=False,
                max_tokens=8,
                stop_token_ids=[],
                user="edge-user",
            )),
            raw_request=None,
            serving=_StubServing(),
        )
    )

    assert response["status_code"] == HTTPStatus.INTERNAL_SERVER_ERROR
    assert "zero-progress" in response["message"]
    assert len(transport.close_calls) == 1


def test_round_coordinator_streaming_preserves_transport_error_status():
    class _StubServing:
        async def render_chat_request(self, request):
            raise AssertionError("render_chat_request should not be called")

        def create_error_response(self, message: str, **kwargs):
            return {"message": message, **kwargs}

        def create_streaming_error_response(self, message: str, **kwargs):
            return json.dumps({"message": str(message), **kwargs})

    coordinator = DSSDRoundCoordinator(
        edge_engine=SimpleNamespace(),
        transport=None,
    )

    async def _collect():
        stream = coordinator.create_chat_completion_stream(
            request=SimpleNamespace(stream=True, max_tokens=8, stop_token_ids=[]),
            raw_request=None,
            serving=_StubServing(),
        )
        return [chunk async for chunk in stream]

    chunks = asyncio.run(_collect())

    assert len(chunks) == 2
    error_payload = json.loads(chunks[0].removeprefix("data: ").strip())
    assert error_payload["status_code"] == HTTPStatus.SERVICE_UNAVAILABLE
    assert chunks[1] == "data: [DONE]\n\n"


def test_round_coordinator_streaming_preserves_internal_error_status():
    class _StubTransport:
        async def bind_verifier(self, request):
            return BindVerifierResponse(
                binding_id="bind-1",
                protocol_version=request.protocol_version,
                verifier_model_id="target-model",
                tokenizer_hash=request.tokenizer_hash,
                vocab_hash=request.vocab_hash,
                supported_gamma_max=request.supported_gamma_max,
            )

        async def create_session(self, request):
            return CreateSessionResponse(
                verifier_session_id="vs-1",
                accepted_prompt_len=len(request.prompt_token_ids),
            )

        async def verify_round(self, request):
            del request
            return VerifyRoundResponse(
                verifier_session_id="vs-1",
                seq_no=0,
                accepted_count=0,
                all_accepted=True,
                bonus_token_id=None,
                finished=False,
            )

        async def close_session(self, request):
            del request
            return {"closed": True}

    class _StubEdgeEngine:
        async def dssd_draft_round_async(self, request):
            del request
            return SimpleNamespace(
                draft_token_ids=[3, 4],
                q_values=[0.7, 0.8],
                q_dists_handle="handle-1",
                q_distributions=[[0.3, 0.7], [0.2, 0.8]],
            )

    class _StubTokenizer:
        def decode(self, token_ids, skip_special_tokens=False):
            del skip_special_tokens
            return "|".join(f"tok{token_id}" for token_id in token_ids)

    class _StubServing:
        def __init__(self) -> None:
            self.models = SimpleNamespace(
                model_name=lambda *_args, **_kwargs: "edge-model"
            )
            self.renderer = SimpleNamespace(
                get_tokenizer=lambda: _StubTokenizer(),
                tokenizer=_StubTokenizer(),
            )

        async def render_chat_request(self, request):
            del request
            return [], [{"prompt_token_ids": [11, 12]}]

        def create_error_response(self, message: str, **kwargs):
            return {"message": message, **kwargs}

        def create_streaming_error_response(self, message: str, **kwargs):
            return json.dumps({"message": str(message), **kwargs})

    _StubServing.default_sampling_params = {
        "temperature": 0.6,
        "top_p": 0.75,
        "top_k": 11,
        "min_p": 0.03,
        "max_tokens": 16,
    }
    _StubServing.model_config = SimpleNamespace(max_model_len=128)

    coordinator = DSSDRoundCoordinator(
        edge_engine=_StubEdgeEngine(),
        transport=_StubTransport(),
        gamma=2,
    )

    async def _collect():
        stream = coordinator.create_chat_completion_stream(
            request=_attach_sampling_params(
                SimpleNamespace(
                    stream=True,
                    max_tokens=8,
                    stop_token_ids=[],
                    user="edge-user",
                    skip_special_tokens=True,
                )
            ),
            raw_request=None,
            serving=_StubServing(),
        )
        return [chunk async for chunk in stream]

    chunks = asyncio.run(_collect())

    assert chunks[0].startswith("data: ")
    error_payload = json.loads(chunks[-2].removeprefix("data: ").strip())
    assert error_payload["status_code"] == HTTPStatus.INTERNAL_SERVER_ERROR
    assert "zero-progress" in error_payload["message"]
    assert chunks[-1] == "data: [DONE]\n\n"
