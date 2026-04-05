# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[3]
VLLM_DIR = ROOT / "vllm"
DSSD_DIR = VLLM_DIR / "v1" / "dssd"
VERIFIER_DIR = DSSD_DIR / "verifier"


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
_install_package_stub("vllm.v1.dssd", DSSD_DIR)
_install_package_stub("vllm.v1.dssd.verifier", VERIFIER_DIR)

protocol_module = _load_module(
    "vllm.v1.dssd.protocol",
    DSSD_DIR / "protocol.py",
)
session_module = _load_module(
    "vllm.v1.dssd.verifier.session",
    VERIFIER_DIR / "session.py",
)
service_module = _load_module(
    "vllm.v1.dssd.verifier.service",
    VERIFIER_DIR / "service.py",
)

BindVerifierRequest = protocol_module.BindVerifierRequest
CreateSessionRequest = protocol_module.CreateSessionRequest
VerifyRoundRequest = protocol_module.VerifyRoundRequest
VerifyRoundResponse = protocol_module.VerifyRoundResponse
VerifierForwardResult = protocol_module.VerifierForwardResult
VerifierSessionInitRequest = protocol_module.VerifierSessionInitRequest
VerifierCommitRequest = protocol_module.VerifierCommitRequest
DSSDVerifierService = service_module.DSSDVerifierService


@pytest.mark.asyncio
async def test_verifier_service_retries_failed_round_without_dup_prefix_delta():
    class _FlakyEngine:
        def __init__(self) -> None:
            self.model_config = SimpleNamespace(model="target-model")
            self.calls = 0

        async def dssd_verify_round_async(self, request):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary failure")
            return VerifyRoundResponse(
                verifier_session_id=request.verifier_session_id,
                seq_no=request.seq_no,
                accepted_count=1,
                all_accepted=True,
                bonus_token_id=9,
            )

    service = DSSDVerifierService(
        _FlakyEngine(),
        SimpleNamespace(
            dssd_config=SimpleNamespace(enabled=True, role="verifier", gamma=2),
        ),
    )
    bind_response = await service.bind_verifier(
        BindVerifierRequest(
            protocol_version="v1alpha1",
            edge_instance_id="edge-1",
            tokenizer_hash="tok",
            vocab_hash="voc",
            supported_gamma_max=2,
        )
    )
    session_response = await service.create_session(
        CreateSessionRequest(
            binding_id=bind_response.binding_id,
            request_id="req-1",
            prompt_token_ids=[10],
            sampling_params_digest="sp-1",
            max_new_tokens=4,
            stop_token_ids=[],
        )
    )
    request = VerifyRoundRequest(
        binding_id=bind_response.binding_id,
        verifier_session_id=session_response.verifier_session_id,
        seq_no=0,
        prefix_delta_token_ids=[99],
        draft_token_ids=[3],
        q_values=[0.7],
    )

    with pytest.raises(RuntimeError, match="temporary failure"):
        await service.verify_round(request)

    response = await service.verify_round(request)
    session = service.session_manager.get_session(session_response.verifier_session_id)

    assert response.accepted_count == 1
    assert session.committed_token_ids == [10, 99, 3, 9]


@pytest.mark.asyncio
async def test_verifier_service_builds_reject_response_from_forward_probs():
    class _Engine:
        def __init__(self) -> None:
            self.model_config = SimpleNamespace(model="target-model")
            self.commit_requests = []

        async def dssd_create_verifier_session_async(self, request):
            return True

        async def dssd_commit_verifier_tokens_async(self, request):
            self.commit_requests.append(request)
            return True

        async def dssd_verify_round_async(self, request):
            return VerifierForwardResult(
                verifier_session_id=request.verifier_session_id,
                seq_no=request.seq_no,
                seq_probs=[[0.1, 0.9], [0.2, 0.8]],
                bonus_probs=[0.3, 0.7],
            )

    class _FixedRNG:
        def __init__(self, draws: list[float]) -> None:
            self._draws = iter(draws)

        def random(self) -> float:
            return next(self._draws)

    service = DSSDVerifierService(
        _Engine(),
        SimpleNamespace(
            dssd_config=SimpleNamespace(enabled=True, role="verifier", gamma=2),
        ),
    )
    bind_response = await service.bind_verifier(
        BindVerifierRequest(
            protocol_version="v1alpha1",
            edge_instance_id="edge-1",
            tokenizer_hash="tok",
            vocab_hash="voc",
            supported_gamma_max=2,
        )
    )
    session_response = await service.create_session(
        CreateSessionRequest(
            binding_id=bind_response.binding_id,
            request_id="req-2",
            prompt_token_ids=[10],
            sampling_params_digest="sp-2",
            max_new_tokens=4,
            stop_token_ids=[],
        )
    )
    service.session_manager.get_session(session_response.verifier_session_id).rng = (
        _FixedRNG([0.05, 0.5])
    )

    response = await service.verify_round(
        VerifyRoundRequest(
            binding_id=bind_response.binding_id,
            verifier_session_id=session_response.verifier_session_id,
            seq_no=0,
            prefix_delta_token_ids=[],
            draft_token_ids=[1, 0],
            q_values=[0.8, 0.5],
        )
    )

    assert response.accepted_count == 1
    assert response.all_accepted is False
    assert response.reject_index == 1
    assert response.reject_target_probs == [0.2, 0.8]
    assert response.bonus_token_id is None
    assert service.session_manager.get_session(
        session_response.verifier_session_id
    ).committed_token_ids == [10, 1]
    assert len(service.engine_client.commit_requests) == 1
    commit_request = service.engine_client.commit_requests[0]
    assert isinstance(commit_request, VerifierCommitRequest)
    assert commit_request.token_ids == [1]


@pytest.mark.asyncio
async def test_verifier_service_builds_bonus_token_from_forward_probs():
    class _Engine:
        def __init__(self) -> None:
            self.model_config = SimpleNamespace(model="target-model")
            self.commit_requests = []

        async def dssd_create_verifier_session_async(self, request):
            return True

        async def dssd_commit_verifier_tokens_async(self, request):
            self.commit_requests.append(request)
            return True

        async def dssd_verify_round_async(self, request):
            return VerifierForwardResult(
                verifier_session_id=request.verifier_session_id,
                seq_no=request.seq_no,
                seq_probs=[[0.1, 0.9]],
                bonus_probs=[0.2, 0.8],
            )

    class _FixedRNG:
        def __init__(self, draws: list[float]) -> None:
            self._draws = iter(draws)

        def random(self) -> float:
            return next(self._draws)

    service = DSSDVerifierService(
        _Engine(),
        SimpleNamespace(
            dssd_config=SimpleNamespace(enabled=True, role="verifier", gamma=1),
        ),
    )
    bind_response = await service.bind_verifier(
        BindVerifierRequest(
            protocol_version="v1alpha1",
            edge_instance_id="edge-1",
            tokenizer_hash="tok",
            vocab_hash="voc",
            supported_gamma_max=1,
        )
    )
    session_response = await service.create_session(
        CreateSessionRequest(
            binding_id=bind_response.binding_id,
            request_id="req-3",
            prompt_token_ids=[11],
            sampling_params_digest="sp-3",
            max_new_tokens=4,
            stop_token_ids=[],
        )
    )
    service.session_manager.get_session(session_response.verifier_session_id).rng = (
        _FixedRNG([0.1, 0.3])
    )

    response = await service.verify_round(
        VerifyRoundRequest(
            binding_id=bind_response.binding_id,
            verifier_session_id=session_response.verifier_session_id,
            seq_no=0,
            prefix_delta_token_ids=[],
            draft_token_ids=[1],
            q_values=[0.4],
        )
    )

    assert response.accepted_count == 1
    assert response.all_accepted is True
    assert response.bonus_token_id == 1
    assert response.reject_index is None
    assert service.session_manager.get_session(
        session_response.verifier_session_id
    ).committed_token_ids == [11, 1, 1]
    assert len(service.engine_client.commit_requests) == 1
    assert service.engine_client.commit_requests[0].token_ids == [1, 1]


@pytest.mark.asyncio
async def test_verifier_service_syncs_engine_session_lifecycle():
    class _Engine:
        def __init__(self) -> None:
            self.model_config = SimpleNamespace(model="target-model")
            self.create_requests = []
            self.close_requests = []

        async def dssd_create_verifier_session_async(self, request):
            self.create_requests.append(request)
            return True

        async def dssd_close_verifier_session_async(self, request):
            self.close_requests.append(request)
            return True

    engine = _Engine()
    service = DSSDVerifierService(
        engine,
        SimpleNamespace(
            dssd_config=SimpleNamespace(enabled=True, role="verifier", gamma=2),
        ),
    )
    bind_response = await service.bind_verifier(
        BindVerifierRequest(
            protocol_version="v1alpha1",
            edge_instance_id="edge-1",
            tokenizer_hash="tok",
            vocab_hash="voc",
            supported_gamma_max=2,
        )
    )

    session_response = await service.create_session(
        CreateSessionRequest(
            binding_id=bind_response.binding_id,
            request_id="req-4",
            prompt_token_ids=[1, 2, 3],
            sampling_params_digest="sp-4",
            max_new_tokens=8,
            stop_token_ids=[],
        )
    )

    assert len(engine.create_requests) == 1
    create_request = engine.create_requests[0]
    assert isinstance(create_request, VerifierSessionInitRequest)
    assert create_request.verifier_session_id == session_response.verifier_session_id
    assert create_request.prompt_token_ids == [1, 2, 3]

    close_response = await service.close_session(
        protocol_module.CloseSessionRequest(
            verifier_session_id=session_response.verifier_session_id,
            reason="done",
        )
    )

    assert close_response.closed is True
    assert len(engine.close_requests) == 1
    assert (
        engine.close_requests[0].verifier_session_id
        == session_response.verifier_session_id
    )
