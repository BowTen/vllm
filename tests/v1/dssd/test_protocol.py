from __future__ import annotations

import importlib.util
import inspect
import sys
import types
from pathlib import Path

import msgspec.msgpack
import pytest


ROOT = Path(__file__).resolve().parents[3]
VLLM_DIR = ROOT / "vllm"
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
_install_package_stub("vllm.v1", VLLM_DIR / "v1")
_install_package_stub("vllm.v1.dssd", DSSD_DIR)

protocol = _load_module("vllm.v1.dssd.protocol", DSSD_DIR / "protocol.py")
transport = _load_module("vllm.v1.dssd.transport", DSSD_DIR / "transport.py")

BindVerifierResponse = protocol.BindVerifierResponse
DraftRoundRequest = protocol.DraftRoundRequest
CreateSessionRequest = protocol.CreateSessionRequest
VerifierSessionInitRequest = protocol.VerifierSessionInitRequest
VerifyRoundRequest = protocol.VerifyRoundRequest
DSSDVerifierExecutionRequest = protocol.DSSDVerifierExecutionRequest
VerifierForwardResult = protocol.VerifierForwardResult
VerifyRoundResponse = protocol.VerifyRoundResponse
DSSDTransport = transport.DSSDTransport


def test_bind_verifier_response_msgpack_round_trip():
    resp = BindVerifierResponse(
        binding_id="bind-1",
        protocol_version="1.0",
        verifier_model_id="qwen-verifier",
        tokenizer_hash="tok-hash",
        vocab_hash="vocab-hash",
        supported_gamma_max=4,
        capabilities={"network_transport": "http"},
    )
    restored = msgspec.msgpack.decode(
        msgspec.msgpack.encode(resp),
        type=BindVerifierResponse,
    )
    assert restored == resp


def test_verify_round_request_msgpack_round_trip():
    req = VerifyRoundRequest(
        binding_id="bind-1",
        verifier_session_id="vs-1",
        seq_no=3,
        prefix_delta_token_ids=[42],
        draft_token_ids=[7, 8, 9],
        q_values=[0.2, 0.3, 0.4],
    )
    restored = msgspec.msgpack.decode(
        msgspec.msgpack.encode(req),
        type=VerifyRoundRequest,
    )
    assert restored == req


def test_dssd_verifier_execution_request_msgpack_round_trip():
    req = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-1",
        seq_no=3,
        committed_token_ids=[11, 22, 42],
        draft_token_ids=[7, 8, 9],
        q_values=[0.2, 0.3, 0.4],
        prefix_delta_token_ids=[42],
    )
    restored = msgspec.msgpack.decode(
        msgspec.msgpack.encode(req),
        type=DSSDVerifierExecutionRequest,
    )
    assert restored == req


def test_create_session_request_msgpack_round_trip():
    req = CreateSessionRequest(
        binding_id="bind-1",
        request_id="req-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params_digest="sha256",
        max_new_tokens=16,
        stop_token_ids=[2],
    )
    restored = msgspec.msgpack.decode(
        msgspec.msgpack.encode(req),
        type=CreateSessionRequest,
    )
    assert restored == req


def test_draft_round_request_msgpack_round_trip():
    req = DraftRoundRequest(
        local_session_id="edge-1",
        prompt_token_ids=[1, 2, 3],
        committed_token_ids=[4],
        seq_no=2,
        gamma=4,
    )
    restored = msgspec.msgpack.decode(
        msgspec.msgpack.encode(req),
        type=DraftRoundRequest,
    )
    assert restored == req


def test_verifier_session_init_request_msgpack_round_trip():
    req = VerifierSessionInitRequest(
        verifier_session_id="vs-1",
        binding_id="bind-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params_digest="sha256",
    )
    restored = msgspec.msgpack.decode(
        msgspec.msgpack.encode(req),
        type=VerifierSessionInitRequest,
    )
    assert restored == req


def test_verify_round_response_reject_shape():
    resp = VerifyRoundResponse(
        verifier_session_id="vs-1",
        seq_no=3,
        accepted_count=1,
        all_accepted=False,
        bonus_token_id=None,
        reject_index=2,
        reject_target_probs=[0.1, 0.9],
        finished=False,
        finish_reason=None,
    )
    assert resp.reject_index == 2
    assert resp.reject_target_probs == [0.1, 0.9]


def test_verifier_forward_result_msgpack_round_trip():
    result = VerifierForwardResult(
        verifier_session_id="vs-1",
        seq_no=3,
        seq_probs=[[0.1, 0.9], [0.2, 0.8]],
        bonus_probs=[0.3, 0.7],
        finished=True,
        finish_reason="stop",
    )
    restored = msgspec.msgpack.decode(
        msgspec.msgpack.encode(result),
        type=VerifierForwardResult,
    )
    assert restored == result


def test_verify_round_request_requires_prefix_delta():
    with pytest.raises(TypeError):
        VerifyRoundRequest(
            binding_id="bind-1",
            verifier_session_id="vs-1",
            seq_no=3,
            draft_token_ids=[7, 8, 9],
            q_values=[0.2, 0.3, 0.4],
        )


def test_transport_is_abstract_and_async():
    with pytest.raises(TypeError):
        DSSDTransport()
    assert inspect.iscoroutinefunction(DSSDTransport.bind_verifier)
    assert inspect.iscoroutinefunction(DSSDTransport.create_session)
    assert inspect.iscoroutinefunction(DSSDTransport.verify_round)
    assert inspect.iscoroutinefunction(DSSDTransport.close_session)
