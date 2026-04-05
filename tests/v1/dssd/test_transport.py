# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI


ROOT = Path(__file__).resolve().parents[3]
VLLM_DIR = ROOT / "vllm"
DSSD_DIR = VLLM_DIR / "v1" / "dssd"
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


_install_package_stub("vllm", VLLM_DIR)
_install_package_stub("vllm.v1", VLLM_DIR / "v1")
_install_package_stub("vllm.v1.dssd", DSSD_DIR)
_install_package_stub("vllm.entrypoints", VLLM_DIR / "entrypoints")
_install_package_stub("vllm.entrypoints.serve", VLLM_DIR / "entrypoints" / "serve")
_install_package_stub("vllm.entrypoints.serve.dssd", DSSD_SERVE_DIR)

protocol_module = _load_module(
    "vllm.v1.dssd.protocol",
    DSSD_DIR / "protocol.py",
)
transport_module = _load_module(
    "vllm.v1.dssd.transport",
    DSSD_DIR / "transport.py",
)
router_module = _load_module(
    "vllm.entrypoints.serve.dssd.api_router",
    DSSD_SERVE_DIR / "api_router.py",
)

BindVerifierRequest = protocol_module.BindVerifierRequest
CreateSessionRequest = protocol_module.CreateSessionRequest
VerifyRoundRequest = protocol_module.VerifyRoundRequest
CloseSessionRequest = protocol_module.CloseSessionRequest
HTTPDSSDTransport = transport_module.HTTPDSSDTransport
attach_router = router_module.attach_router


@pytest.mark.asyncio
async def test_http_transport_round_trip_against_router():
    class _StubVerifierService:
        async def bind_verifier(self, request):
            return {
                "binding_id": "bind-1",
                "protocol_version": request.protocol_version,
                "verifier_model_id": "target-model",
                "tokenizer_hash": request.tokenizer_hash,
                "vocab_hash": request.vocab_hash,
                "supported_gamma_max": request.supported_gamma_max,
                "capabilities": {"transport": "http"},
            }

        async def create_session(self, request):
            return {
                "verifier_session_id": "vs-1",
                "accepted_prompt_len": len(request.prompt_token_ids),
                "expires_at": None,
            }

        async def verify_round(self, request):
            return {
                "verifier_session_id": request.verifier_session_id,
                "seq_no": request.seq_no,
                "accepted_count": 1,
                "all_accepted": True,
                "bonus_token_id": 9,
                "finished": False,
            }

        async def close_session(self, request):
            return {"closed": request.verifier_session_id == "vs-1"}

    app = FastAPI()
    app.state.dssd_verifier_service = _StubVerifierService()
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
        bind_response = await transport.bind_verifier(
            BindVerifierRequest(
                protocol_version="v1alpha1",
                edge_instance_id="edge-1",
                tokenizer_hash="tok",
                vocab_hash="voc",
                supported_gamma_max=4,
            )
        )
        session_response = await transport.create_session(
            CreateSessionRequest(
                binding_id=bind_response.binding_id,
                request_id="req-1",
                prompt_token_ids=[1, 2],
                sampling_params_digest="sp-1",
                max_new_tokens=8,
                stop_token_ids=[2],
            )
        )
        round_response = await transport.verify_round(
            VerifyRoundRequest(
                binding_id=bind_response.binding_id,
                verifier_session_id=session_response.verifier_session_id,
                seq_no=0,
                prefix_delta_token_ids=[],
                draft_token_ids=[3],
                q_values=[0.9],
            )
        )
        close_response = await transport.close_session(
            CloseSessionRequest(
                verifier_session_id=session_response.verifier_session_id,
                reason="done",
            )
        )
    finally:
        await transport.aclose()

    assert bind_response.binding_id == "bind-1"
    assert session_response.verifier_session_id == "vs-1"
    assert round_response.bonus_token_id == 9
    assert close_response.closed is True
