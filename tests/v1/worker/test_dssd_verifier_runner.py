# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[3]
VLLM_DIR = ROOT / "vllm"
DSSD_DIR = VLLM_DIR / "v1" / "dssd"
DSSD_ENGINE_DIR = DSSD_DIR / "engine"
DSSD_EDGE_DIR = DSSD_DIR / "edge"


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
_install_package_stub("vllm.v1.dssd.edge", DSSD_EDGE_DIR)
_install_package_stub("vllm.v1.dssd.engine", DSSD_ENGINE_DIR)
_install_package_stub("vllm.v1.dssd.worker", DSSD_DIR / "worker")

_load_module("vllm.v1.dssd.edge.session", DSSD_EDGE_DIR / "session.py")
_load_module("vllm.v1.dssd.engine.batch_planner", DSSD_ENGINE_DIR / "batch_planner.py")
_load_module("vllm.v1.dssd.engine.session_store", DSSD_ENGINE_DIR / "session_store.py")
_load_module("vllm.v1.dssd.worker.draft_runner", DSSD_DIR / "worker" / "draft_runner.py")
DSSDSessionRunner = _load_module(
    "vllm.v1.dssd.engine.session_runner",
    DSSD_DIR / "engine" / "session_runner.py",
).DSSDSessionRunner
protocol_module = _load_module("vllm.v1.dssd.protocol", DSSD_DIR / "protocol.py")
VerifyRoundRequest = protocol_module.VerifyRoundRequest
VerifyRoundResponse = protocol_module.VerifyRoundResponse


def test_session_runner_uses_collective_rpc_for_verify_round():
    expected = VerifyRoundResponse(
        verifier_session_id="vs-1",
        seq_no=2,
        accepted_count=1,
        all_accepted=True,
        reject_index=None,
        reject_target_probs=None,
        finished=True,
        finish_reason="rpc-result",
    )
    model_executor = types.SimpleNamespace(
        collective_rpc=Mock(return_value=[expected]),
    )
    runner = DSSDSessionRunner(model_executor=model_executor)
    request = VerifyRoundRequest(
        binding_id="bind-1",
        verifier_session_id="vs-1",
        seq_no=2,
        prefix_delta_token_ids=[4],
        draft_token_ids=[7, 8],
        q_values=[0.6, 0.4],
    )

    result = runner.dssd_verify_round(request)

    assert result == expected
    model_executor.collective_rpc.assert_called_once_with(
        "dssd_verify_round",
        args=(request,),
    )
