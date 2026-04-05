# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
import types
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


_install_package_stub("vllm", VLLM_DIR)
_install_package_stub("vllm.v1", VLLM_DIR / "v1")
_install_package_stub("vllm.v1.dssd", VLLM_DIR / "v1" / "dssd")
_install_package_stub("vllm.v1.dssd.edge", DSSD_EDGE_DIR)

coordinator_module = _load_module(
    "vllm.v1.dssd.edge.coordinator",
    DSSD_EDGE_DIR / "coordinator.py",
)
protocol_module = _load_module(
    "vllm.v1.dssd.protocol",
    VLLM_DIR / "v1" / "dssd" / "protocol.py",
)

DSSDRoundCoordinator = coordinator_module.DSSDRoundCoordinator
VerifyRoundResponse = protocol_module.VerifyRoundResponse
BindVerifierResponse = protocol_module.BindVerifierResponse
CreateSessionResponse = protocol_module.CreateSessionResponse


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
            self.engine_client = SimpleNamespace(
                vllm_config=SimpleNamespace(
                    dssd_config=SimpleNamespace(gamma=4),
                )
            )
            self.rendered = False

        async def render_chat_request(self, request):
            self.rendered = True
            return [], [{"prompt_token_ids": [1, 2, 3]}]

        def create_error_response(self, message: str, **kwargs):
            return {"message": message, **kwargs}

    coordinator = DSSDRoundCoordinator(edge_engine=SimpleNamespace(), transport=_StubTransport())
    request = SimpleNamespace(
        stream=False,
        max_tokens=16,
        stop_token_ids=[2],
        user="edge-user",
    )
    raw_request = SimpleNamespace()
    serving = _StubServing()

    import asyncio

    result = asyncio.run(
        coordinator.create_chat_completion(
            request=request,
            raw_request=raw_request,
            serving=serving,
        )
    )

    assert serving.rendered is True
    assert len(coordinator.transport.bind_calls) == 1
    assert len(coordinator.transport.session_calls) == 1
    assert result["status_code"] == HTTPStatus.NOT_IMPLEMENTED
    assert "round loop" in result["message"]
