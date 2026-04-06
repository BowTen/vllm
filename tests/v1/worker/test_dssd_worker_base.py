# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[3]
VLLM_DIR = ROOT / "vllm"
DSSD_DIR = VLLM_DIR / "v1" / "dssd"
WORKER_DIR = VLLM_DIR / "v1" / "worker"


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


def _install_stubs() -> None:
    base_field = type("BaseMultiModalField", (), {})
    flat_field = type("MultiModalFlatField", (base_field,), {})
    shared_field = type("MultiModalSharedField", (base_field,), {})
    batched_field = type("MultiModalBatchedField", (base_field,), {})

    _install_module_stub(
        "vllm.config",
        VllmConfig=object,
        set_current_vllm_config=lambda *_args, **_kwargs: nullcontext(),
    )
    _install_module_stub("vllm.envs")
    _install_module_stub("vllm.logger", init_logger=lambda _name: _DummyLogger())
    _install_module_stub("vllm.lora.request", LoRARequest=object)
    _install_module_stub("vllm.multimodal", MULTIMODAL_REGISTRY=object())
    _install_module_stub("vllm.tracing", instrument=_noop_decorator)
    _install_module_stub(
        "vllm.utils.import_utils",
        resolve_obj_by_qualname=lambda qualname: qualname,
    )
    _install_module_stub(
        "vllm.utils.system_utils",
        update_environment_variables=lambda _envs: None,
    )
    _install_module_stub(
        "vllm.v1.kv_cache_interface",
        KVCacheSpec=object,
    )
    _install_module_stub(
        "vllm.multimodal.inputs",
        BaseMultiModalField=base_field,
        MultiModalBatchedField=batched_field,
        MultiModalFieldConfig=object,
        MultiModalFieldElem=object,
        MultiModalFlatField=flat_field,
        MultiModalKwargsItem=object,
        MultiModalKwargsItems=object,
        MultiModalSharedField=shared_field,
        NestedTensors=object,
    )
    _install_module_stub(
        "vllm.utils.platform_utils",
        is_pin_memory_available=lambda: False,
    )
    _install_module_stub("vllm.v1.utils", tensor_data=lambda tensor: tensor)


_install_package_stub("vllm", VLLM_DIR)
_install_package_stub("vllm.v1", VLLM_DIR / "v1")
_install_package_stub("vllm.v1.dssd", DSSD_DIR)
_install_package_stub("vllm.v1.dssd.worker", DSSD_DIR / "worker")
_install_package_stub("vllm.v1.worker", WORKER_DIR)
_install_package_stub("vllm.lora", VLLM_DIR / "lora")
_install_package_stub("vllm.multimodal", VLLM_DIR / "multimodal")
_install_package_stub("vllm.utils", VLLM_DIR / "utils")

_install_stubs()

protocol_module = _load_module("vllm.v1.dssd.protocol", DSSD_DIR / "protocol.py")
draft_runner_module = _load_module(
    "vllm.v1.dssd.worker.draft_runner",
    DSSD_DIR / "worker" / "draft_runner.py",
)
serial_utils_module = _load_module(
    "vllm.v1.serial_utils",
    VLLM_DIR / "v1" / "serial_utils.py",
)
worker_base_module = _load_module(
    "vllm.v1.worker.worker_base",
    WORKER_DIR / "worker_base.py",
)

DraftRoundRequest = protocol_module.DraftRoundRequest
VerifyRoundRequest = protocol_module.VerifyRoundRequest
DSSDVerifierExecutionRequest = protocol_module.DSSDVerifierExecutionRequest
CloseSessionRequest = protocol_module.CloseSessionRequest
VerifierForwardResult = protocol_module.VerifierForwardResult
DraftRoundResult = draft_runner_module.DraftRoundResult
run_method = serial_utils_module.run_method
WorkerBase = worker_base_module.WorkerBase
WorkerWrapperBase = worker_base_module.WorkerWrapperBase


def _make_worker(*, model_runner) -> WorkerBase:
    worker = object.__new__(WorkerBase)
    worker.model_runner = model_runner
    return worker


def _make_wrapper(*, worker: WorkerBase) -> WorkerWrapperBase:
    wrapper = object.__new__(WorkerWrapperBase)
    wrapper.worker = worker
    return wrapper


def test_run_method_routes_dssd_draft_round_to_model_runner():
    request = DraftRoundRequest(
        local_session_id="edge-1",
        prompt_token_ids=[1, 2],
        committed_token_ids=[3],
        seq_no=0,
        gamma=2,
    )
    expected = DraftRoundResult(
        draft_token_ids=[9, 8],
        q_values=[0.9, 0.8],
        q_dists_handle="worker-handle",
        q_distributions=[[0.1, 0.9], [0.2, 0.8]],
    )
    model_runner = SimpleNamespace(
        dssd_draft_round=Mock(return_value=expected),
    )
    wrapper = _make_wrapper(worker=_make_worker(model_runner=model_runner))

    result = run_method(wrapper, "dssd_draft_round", (request,), {})

    assert result == expected
    model_runner.dssd_draft_round.assert_called_once_with(request)


def test_run_method_routes_dssd_verify_round_to_model_runner():
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-1",
        seq_no=2,
        committed_token_ids=[1, 2, 3, 4],
        draft_token_ids=[7, 8],
        q_values=[0.6, 0.4],
    )
    expected = VerifierForwardResult(
        verifier_session_id="vs-1",
        seq_no=2,
        seq_probs=[[0.1, 0.9], [0.2, 0.8]],
        bonus_probs=[0.3, 0.7],
    )
    model_runner = SimpleNamespace(
        dssd_verify_round=Mock(return_value=expected),
    )
    wrapper = _make_wrapper(worker=_make_worker(model_runner=model_runner))

    result = run_method(wrapper, "dssd_verify_round", (request,), {})

    assert result == expected
    model_runner.dssd_verify_round.assert_called_once_with(request)


def test_run_method_routes_dssd_close_verifier_session_to_model_runner():
    request = CloseSessionRequest(
        verifier_session_id="vs-1",
        reason="done",
    )
    model_runner = SimpleNamespace(
        dssd_close_verifier_session=Mock(return_value=True),
    )
    wrapper = _make_wrapper(worker=_make_worker(model_runner=model_runner))

    result = run_method(wrapper, "dssd_close_verifier_session", (request,), {})

    assert result is True
    model_runner.dssd_close_verifier_session.assert_called_once_with(request)
