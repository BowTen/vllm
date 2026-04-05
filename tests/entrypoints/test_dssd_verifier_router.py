# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[2]
VLLM_DIR = ROOT / "vllm"
DSSD_SERVE_DIR = VLLM_DIR / "entrypoints" / "serve" / "dssd"
DSSD_DIR = VLLM_DIR / "v1" / "dssd"


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
_install_package_stub("vllm.entrypoints", VLLM_DIR / "entrypoints")
_install_package_stub("vllm.entrypoints.serve", VLLM_DIR / "entrypoints" / "serve")
_install_package_stub("vllm.entrypoints.serve.dssd", DSSD_SERVE_DIR)
_install_package_stub("vllm.v1", VLLM_DIR / "v1")
_install_package_stub("vllm.v1.dssd", DSSD_DIR)

protocol_module = _load_module(
    "vllm.v1.dssd.protocol",
    DSSD_DIR / "protocol.py",
)

router_module = _load_module(
    "vllm.entrypoints.serve.dssd.api_router",
    DSSD_SERVE_DIR / "api_router.py",
)

attach_router = router_module.attach_router
BindVerifierRequest = protocol_module.BindVerifierRequest
CreateSessionRequest = protocol_module.CreateSessionRequest
VerifyRoundRequest = protocol_module.VerifyRoundRequest
CloseSessionRequest = protocol_module.CloseSessionRequest


def test_verifier_router_exposes_bind_endpoint():
    class _StubVerifierService:
        async def bind_verifier(self, request):
            assert isinstance(request, BindVerifierRequest)
            return {"binding_id": "bind-1"}

        async def create_session(self, request):
            assert isinstance(request, CreateSessionRequest)
            return {"verifier_session_id": "vs-1"}

        async def verify_round(self, request):
            assert isinstance(request, VerifyRoundRequest)
            return {
                "verifier_session_id": request.verifier_session_id,
                "seq_no": request.seq_no,
                "accepted_count": 0,
                "all_accepted": False,
                "reject_index": 0,
                "reject_target_probs": [1.0],
            }

        async def close_session(self, request):
            assert isinstance(request, CloseSessionRequest)
            return {"closed": True}

    app = FastAPI()
    app.state.dssd_verifier_service = _StubVerifierService()
    attach_router(app)

    client = TestClient(app)
    paths = {route.path for route in app.routes}

    assert "/server/dssd/bind" in paths
    response = client.post(
        "/server/dssd/bind",
        json={
            "protocol_version": "v1alpha1",
            "edge_instance_id": "edge-1",
            "tokenizer_hash": "tok",
            "vocab_hash": "voc",
            "supported_gamma_max": 4,
        },
    )
    assert response.status_code == 200
    assert response.json()["binding_id"] == "bind-1"

    response = client.post(
        "/server/dssd/sessions",
        json={
            "binding_id": "bind-1",
            "request_id": "req-1",
            "prompt_token_ids": [1, 2],
            "sampling_params_digest": "sp-1",
            "max_new_tokens": 8,
            "stop_token_ids": [2],
        },
    )
    assert response.status_code == 200
    assert response.json()["verifier_session_id"] == "vs-1"

    response = client.post(
        "/server/dssd/verify_round",
        json={
            "binding_id": "bind-1",
            "verifier_session_id": "vs-1",
            "seq_no": 0,
            "prefix_delta_token_ids": [],
            "draft_token_ids": [3],
            "q_values": [0.9],
        },
    )
    assert response.status_code == 200
    assert response.json()["accepted_count"] == 0

    response = client.post(
        "/server/dssd/close_session",
        json={
            "verifier_session_id": "vs-1",
            "reason": "done",
        },
    )
    assert response.status_code == 200
    assert response.json()["closed"] is True


def test_verifier_router_fails_closed_without_service():
    app = FastAPI()
    attach_router(app)

    client = TestClient(app)
    response = client.post(
        "/server/dssd/bind",
        json={
            "protocol_version": "v1alpha1",
            "edge_instance_id": "edge-1",
            "tokenizer_hash": "tok",
            "vocab_hash": "voc",
            "supported_gamma_max": 4,
        },
    )

    assert response.status_code == 503
