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
    assert session.committed_token_ids == [10, 99]
