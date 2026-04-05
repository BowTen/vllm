# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path
from types import SimpleNamespace


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
class _ChatCompletionResponse:
    id: str
    model: str
    choices: list[_ChatCompletionResponseChoice]
    usage: _UsageInfo
    created: int = 0
    object: str = "chat.completion"


_install_package_stub("vllm", VLLM_DIR)
_install_package_stub("vllm.v1", VLLM_DIR / "v1")
_install_package_stub("vllm.v1.dssd", VLLM_DIR / "v1" / "dssd")
_install_package_stub("vllm.v1.dssd.edge", DSSD_EDGE_DIR)
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

coordinator_module = _load_module(
    "vllm.v1.dssd.edge.coordinator",
    DSSD_EDGE_DIR / "coordinator.py",
)
protocol_module = _load_module(
    "vllm.v1.dssd.protocol",
    VLLM_DIR / "v1" / "dssd" / "protocol.py",
)

DSSDRoundCoordinator = coordinator_module.DSSDRoundCoordinator
BindVerifierResponse = protocol_module.BindVerifierResponse
CreateSessionResponse = protocol_module.CreateSessionResponse
DraftRoundRequest = protocol_module.DraftRoundRequest
VerifyRoundResponse = protocol_module.VerifyRoundResponse


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

        async def render_chat_request(self, request):
            del request
            return [], [{"prompt_token_ids": [1, 2, 3]}]

        def create_error_response(self, message: str, **kwargs):
            return {"message": message, **kwargs}

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

    import asyncio

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

    transport = _StubTransport()
    coordinator = DSSDRoundCoordinator(
        edge_engine=_StubEdgeEngine(),
        transport=transport,
        gamma=2,
    )

    import asyncio

    response = asyncio.run(
        coordinator.create_chat_completion(
            request=SimpleNamespace(
                stream=False,
                max_tokens=2,
                stop_token_ids=[],
                user="edge-user",
            ),
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

    transport = _StubTransport()
    coordinator = DSSDRoundCoordinator(
        edge_engine=_StubEdgeEngine(),
        transport=transport,
        gamma=2,
    )

    import asyncio

    response = asyncio.run(
        coordinator.create_chat_completion(
            request=SimpleNamespace(
                stream=False,
                max_tokens=2,
                stop_token_ids=[],
                user="edge-user",
            ),
            raw_request=None,
            serving=_StubServing(),
        )
    )

    assert response.model == "edge-model"
    assert response.choices[0].message.content == "tok1|tok0"
    assert response.usage.completion_tokens == 2
    assert response.choices[0].finish_reason == "length"
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

    transport = _StubTransport()
    coordinator = DSSDRoundCoordinator(
        edge_engine=_StubEdgeEngine(),
        transport=transport,
        gamma=2,
    )

    import asyncio

    response = asyncio.run(
        coordinator.create_chat_completion(
            request=SimpleNamespace(
                stream=False,
                max_tokens=4,
                stop_token_ids=[2],
                user="edge-user",
            ),
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

    transport = _StubTransport()
    coordinator = DSSDRoundCoordinator(
        edge_engine=_StubEdgeEngine(),
        transport=transport,
        gamma=2,
    )

    import asyncio

    response = asyncio.run(
        coordinator.create_chat_completion(
            request=SimpleNamespace(
                stream=False,
                max_tokens=2,
                stop_token_ids=[],
                user="edge-user",
            ),
            raw_request=None,
            serving=_StubServing(),
        )
    )

    assert response["status_code"] == HTTPStatus.INTERNAL_SERVER_ERROR
    assert "resample" in response["message"]
    assert len(transport.close_calls) == 1
