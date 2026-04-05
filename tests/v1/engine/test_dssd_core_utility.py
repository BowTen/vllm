# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import AsyncMock

import pytest


ROOT = Path(__file__).resolve().parents[3]
VLLM_DIR = ROOT / "vllm"
DSSD_DIR = VLLM_DIR / "v1" / "dssd"
ENGINE_DIR = VLLM_DIR / "v1" / "engine"


def _install_package_stub(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


def _install_module_stub(name: str, **attrs) -> None:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _DummyLogger:

    def __getattr__(self, _name: str):
        return lambda *args, **kwargs: None


def _noop_decorator(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


def _install_core_client_stubs() -> None:
    utility_type = types.SimpleNamespace(value=b"\x03")
    request_type = types.SimpleNamespace(UTILITY=utility_type)
    dummy_socket_type = type("Socket", (), {})
    dummy_context_type = type("Context", (), {})
    dummy_poller_type = type("Poller", (), {})
    dummy_frame_type = type("Frame", (), {})
    dummy_tracker_type = type("MessageTracker", (), {})
    dummy_again_type = type("Again", (Exception,), {})
    zmq_asyncio = types.ModuleType("zmq.asyncio")
    zmq_asyncio.Context = dummy_context_type
    zmq_asyncio.Poller = dummy_poller_type
    zmq_asyncio.Socket = dummy_socket_type

    _install_module_stub(
        "zmq",
        Again=dummy_again_type,
        Context=dummy_context_type,
        Frame=dummy_frame_type,
        MessageTracker=dummy_tracker_type,
        PAIR=0,
        POLLIN=1,
        PULL=2,
        ROUTER=3,
        Socket=dummy_socket_type,
        XSUB=4,
        NOBLOCK=5,
        Poller=dummy_poller_type,
        asyncio=zmq_asyncio,
    )
    sys.modules["zmq.asyncio"] = zmq_asyncio

    _install_module_stub(
        "vllm.config",
        VllmConfig=object,
        set_current_vllm_config=lambda *_args, **_kwargs: nullcontext(),
    )
    _install_module_stub("vllm.envs", VLLM_ENGINE_READY_TIMEOUT_S=60)
    _install_module_stub("vllm.logger", init_logger=lambda _name: _DummyLogger())
    _install_module_stub("vllm.lora.request", LoRARequest=object)
    _install_module_stub("vllm.multimodal", MULTIMODAL_REGISTRY=object())
    _install_module_stub("vllm.tasks", SupportedTask=str)
    _install_module_stub("vllm.tracing", instrument=_noop_decorator)
    _install_module_stub("vllm.utils.async_utils", in_loop=lambda: False)
    _install_module_stub(
        "vllm.utils.network_utils",
        close_sockets=lambda *args, **kwargs: None,
        get_open_zmq_inproc_path=lambda: "inproc://test",
        make_zmq_socket=lambda *args, **kwargs: None,
    )
    _install_module_stub(
        "vllm.v1.engine",
        EEP_NOTIFICATION_CALL_ID=0,
        EEPNotificationType=object,
        EngineCoreOutputs=object,
        EngineCoreRequest=object,
        EngineCoreRequestType=request_type,
        PauseMode=str,
        ReconfigureDistributedRequest=object,
        ReconfigureRankType=object,
        UtilityOutput=object,
    )
    _install_module_stub("vllm.v1.engine.coordinator", DPCoordinator=object)
    _install_module_stub(
        "vllm.v1.engine.core",
        EngineCore=object,
        EngineCoreProc=object,
    )
    _install_module_stub(
        "vllm.v1.engine.exceptions",
        EngineDeadError=RuntimeError,
    )
    _install_module_stub(
        "vllm.v1.engine.utils",
        CoreEngineActorManager=object,
        CoreEngineProcManager=object,
        get_engine_zmq_addresses=lambda *args, **kwargs: {},
        launch_core_engines=lambda *args, **kwargs: None,
    )
    _install_module_stub("vllm.v1.executor", Executor=object)
    _install_module_stub(
        "vllm.v1.pool.late_interaction",
        get_late_interaction_engine_index=lambda *args, **kwargs: 0,
    )
    _install_module_stub(
        "vllm.v1.serial_utils",
        MsgpackDecoder=object,
        MsgpackEncoder=object,
        bytestr=bytes,
        run_method=lambda obj, method, args, kwargs: getattr(obj, method)(*args, **kwargs),
    )


_install_package_stub("vllm", VLLM_DIR)
_install_package_stub("vllm.v1", VLLM_DIR / "v1")
_install_package_stub("vllm.v1.dssd", DSSD_DIR)
_install_package_stub("vllm.v1.dssd.engine", DSSD_DIR / "engine")
_install_package_stub("vllm.v1.engine", ENGINE_DIR)
_install_package_stub("vllm.lora", VLLM_DIR / "lora")
_install_package_stub("vllm.utils", VLLM_DIR / "utils")
_install_package_stub("vllm.v1.pool", VLLM_DIR / "v1" / "pool")

protocol = _load_module("vllm.v1.dssd.protocol", DSSD_DIR / "protocol.py")
session_store_module = _load_module(
    "vllm.v1.dssd.engine.session_store",
    DSSD_DIR / "engine" / "session_store.py",
)
_install_core_client_stubs()
core_client_module = _load_module(
    "vllm.v1.engine.core_client",
    ENGINE_DIR / "core_client.py",
)

VerifyRoundRequest = protocol.VerifyRoundRequest
VerifierSessionInitRequest = protocol.VerifierSessionInitRequest
CloseSessionRequest = protocol.CloseSessionRequest
DraftRoundRequest = protocol.DraftRoundRequest
DSSDSessionStore = session_store_module.DSSDSessionStore
AsyncMPClient = core_client_module.AsyncMPClient


@pytest.mark.asyncio
async def test_async_client_exposes_verify_round_utility():
    client = object.__new__(AsyncMPClient)
    client.call_utility_async = AsyncMock(return_value="ok")

    request = VerifyRoundRequest(
        binding_id="bind-1",
        verifier_session_id="vs-1",
        seq_no=0,
        prefix_delta_token_ids=[],
        draft_token_ids=[1, 2],
        q_values=[0.5, 0.4],
    )

    result = await AsyncMPClient.dssd_verify_round_async(client, request)

    assert result == "ok"
    client.call_utility_async.assert_awaited_once_with("dssd_verify_round", request)


@pytest.mark.asyncio
async def test_async_client_exposes_draft_round_utility():
    client = object.__new__(AsyncMPClient)
    client.call_utility_async = AsyncMock(return_value="draft-ok")

    request = DraftRoundRequest(
        local_session_id="edge-1",
        prompt_token_ids=[1, 2],
        committed_token_ids=[3],
        seq_no=0,
        gamma=2,
    )

    result = await AsyncMPClient.dssd_draft_round_async(client, request)

    assert result == "draft-ok"
    client.call_utility_async.assert_awaited_once_with("dssd_draft_round", request)


@pytest.mark.asyncio
async def test_async_client_exposes_create_verifier_session_utility():
    client = object.__new__(AsyncMPClient)
    client.call_utility_async = AsyncMock(return_value=True)

    request = VerifierSessionInitRequest(
        verifier_session_id="vs-1",
        binding_id="bind-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params_digest="sp-1",
    )

    result = await AsyncMPClient.dssd_create_verifier_session_async(client, request)

    assert result is True
    client.call_utility_async.assert_awaited_once_with(
        "dssd_create_verifier_session", request
    )


@pytest.mark.asyncio
async def test_async_client_exposes_close_verifier_session_utility():
    client = object.__new__(AsyncMPClient)
    client.call_utility_async = AsyncMock(return_value=True)

    request = CloseSessionRequest(
        verifier_session_id="vs-1",
        reason="done",
    )

    result = await AsyncMPClient.dssd_close_verifier_session_async(client, request)

    assert result is True
    client.call_utility_async.assert_awaited_once_with(
        "dssd_close_verifier_session", request
    )


def test_session_store_round_trip():
    store = DSSDSessionStore()
    store.put("edge-1", {"seq_no": 0})
    assert store.get("edge-1") == {"seq_no": 0}
    store.delete("edge-1")
    assert store.get("edge-1") is None
