from __future__ import annotations

import importlib.util
import inspect
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import msgspec
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


protocol = None
transport = None
dssd_package = None
BindVerifierResponse = None
DraftRoundRequest = None
CreateSessionRequest = None
VerifierSessionInitRequest = None
VerifyRoundRequest = None
DSSDVerifierExecutionRequest = None
VerifierForwardResult = None
VerifyRoundResponse = None
DSSDTransport = None
SamplingParams = None


@contextmanager
def _protocol_modules() -> Iterator[SimpleNamespace]:
    saved_vllm_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "vllm" or name.startswith("vllm.")
    }

    try:
        _install_package_stub("vllm", VLLM_DIR)
        _install_package_stub("vllm.v1", VLLM_DIR / "v1")
        _install_package_stub("vllm.v1.dssd", DSSD_DIR)

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

        protocol_module = _load_module(
            "vllm.v1.dssd.protocol",
            DSSD_DIR / "protocol.py",
        )
        transport_module = _load_module(
            "vllm.v1.dssd.transport",
            DSSD_DIR / "transport.py",
        )
        dssd_package_module = _load_module(
            "vllm.v1.dssd",
            DSSD_DIR / "__init__.py",
        )

        yield SimpleNamespace(
            protocol=protocol_module,
            transport=transport_module,
            dssd_package=dssd_package_module,
            SamplingParams=sampling_params_module.SamplingParams,
        )
    finally:
        for name in list(sys.modules):
            if name == "vllm" or name.startswith("vllm."):
                if name not in saved_vllm_modules:
                    sys.modules.pop(name, None)
        sys.modules.update(saved_vllm_modules)


@pytest.fixture(scope="module", autouse=True)
def _install_protocol_modules() -> Iterator[None]:
    global protocol
    global transport
    global dssd_package
    global BindVerifierResponse
    global DraftRoundRequest
    global CreateSessionRequest
    global VerifierSessionInitRequest
    global VerifyRoundRequest
    global DSSDVerifierExecutionRequest
    global VerifierForwardResult
    global VerifyRoundResponse
    global DSSDTransport
    global SamplingParams

    with _protocol_modules() as modules:
        protocol = modules.protocol
        transport = modules.transport
        dssd_package = modules.dssd_package
        BindVerifierResponse = protocol.BindVerifierResponse
        DraftRoundRequest = protocol.DraftRoundRequest
        CreateSessionRequest = protocol.CreateSessionRequest
        VerifierSessionInitRequest = protocol.VerifierSessionInitRequest
        VerifyRoundRequest = protocol.VerifyRoundRequest
        DSSDVerifierExecutionRequest = protocol.DSSDVerifierExecutionRequest
        VerifierForwardResult = protocol.VerifierForwardResult
        VerifyRoundResponse = protocol.VerifyRoundResponse
        DSSDTransport = transport.DSSDTransport
        SamplingParams = modules.SamplingParams
        yield


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
    )
    restored = msgspec.msgpack.decode(
        msgspec.msgpack.encode(req),
        type=DSSDVerifierExecutionRequest,
    )
    assert restored == req


def test_dssd_verifier_execution_request_rejects_prefix_delta():
    with pytest.raises(TypeError):
        DSSDVerifierExecutionRequest(
            binding_id="bind-1",
            verifier_session_id="vs-1",
            seq_no=3,
            committed_token_ids=[11, 22, 42],
            draft_token_ids=[7, 8, 9],
            q_values=[0.2, 0.3, 0.4],
            prefix_delta_token_ids=[42],
        )


def test_package_keeps_internal_execution_request_without_public_export():
    assert dssd_package.DSSDVerifierExecutionRequest is DSSDVerifierExecutionRequest
    assert "DSSDVerifierExecutionRequest" not in dssd_package.__all__


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
    assert restored.sampling_params is None


def test_draft_round_request_msgpack_round_trip_with_sampling_params():
    req = DraftRoundRequest(
        local_session_id="edge-1",
        prompt_token_ids=[1, 2, 3],
        committed_token_ids=[4],
        seq_no=2,
        gamma=4,
        sampling_params=SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=12,
            min_p=0.05,
            seed=123,
            max_tokens=32,
        ),
    )

    restored = msgspec.msgpack.decode(
        msgspec.msgpack.encode(req),
        type=DraftRoundRequest,
    )

    assert restored.local_session_id == req.local_session_id
    assert restored.prompt_token_ids == req.prompt_token_ids
    assert restored.committed_token_ids == req.committed_token_ids
    assert restored.seq_no == req.seq_no
    assert restored.gamma == req.gamma
    assert restored.sampling_params is not None
    assert restored.sampling_params.temperature == req.sampling_params.temperature
    assert restored.sampling_params.top_p == req.sampling_params.top_p
    assert restored.sampling_params.top_k == req.sampling_params.top_k
    assert restored.sampling_params.min_p == req.sampling_params.min_p
    assert restored.sampling_params.seed == req.sampling_params.seed
    assert restored.sampling_params.max_tokens == req.sampling_params.max_tokens


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
