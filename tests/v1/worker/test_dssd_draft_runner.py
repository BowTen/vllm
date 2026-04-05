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

edge_session_module = _load_module(
    "vllm.v1.dssd.edge.session",
    DSSD_EDGE_DIR / "session.py",
)
protocol_module = _load_module(
    "vllm.v1.dssd.protocol",
    DSSD_DIR / "protocol.py",
)
draft_runner_module = _load_module(
    "vllm.v1.dssd.worker.draft_runner",
    DSSD_DIR / "worker" / "draft_runner.py",
)
session_runner_module = _load_module(
    "vllm.v1.dssd.engine.session_runner",
    DSSD_ENGINE_DIR / "session_runner.py",
)

DraftRoundResult = draft_runner_module.DraftRoundResult
DSSDEdgeSessionState = edge_session_module.DSSDEdgeSessionState
DSSDSessionRunner = session_runner_module.DSSDSessionRunner
DraftRoundRequest = protocol_module.DraftRoundRequest
VerifyRoundRequest = protocol_module.VerifyRoundRequest
VerifyRoundResponse = protocol_module.VerifyRoundResponse


def test_draft_round_result_records_minimal_state():
    result = DraftRoundResult(
        draft_token_ids=[7, 8],
        q_values=[0.2, 0.8],
        q_dists_handle="draft-handle",
        q_distributions=[[0.8, 0.2], [0.2, 0.8]],
    )

    assert result.draft_token_ids == [7, 8]
    assert result.q_values == [0.2, 0.8]
    assert result.q_dists_handle == "draft-handle"
    assert result.q_distributions == [[0.8, 0.2], [0.2, 0.8]]


def test_session_runner_uses_collective_rpc_for_draft_round():
    expected = DraftRoundResult(
        draft_token_ids=[9, 8],
        q_values=[0.9, 0.8],
        q_dists_handle="rpc-handle",
        q_distributions=[[0.1, 0.9], [0.2, 0.8]],
    )
    model_executor = types.SimpleNamespace(
        collective_rpc=Mock(return_value=[expected]),
    )
    runner = DSSDSessionRunner(model_executor=model_executor)
    request = DraftRoundRequest(
        local_session_id="edge-1",
        prompt_token_ids=[1, 2, 3],
        committed_token_ids=[],
        seq_no=0,
        gamma=2,
    )
    runner.create_edge_session(
        DSSDEdgeSessionState(
            request_id="req-1",
            local_session_id="edge-1",
            verifier_binding_id="bind-1",
            verifier_session_id="vs-1",
            prompt_token_ids=[1, 2, 3],
        )
    )
    result = runner.dssd_draft_round(request)

    assert isinstance(result, DraftRoundResult)
    assert result == expected
    model_executor.collective_rpc.assert_called_once_with(
        "dssd_draft_round",
        args=(request,),
    )


def test_session_runner_returns_typed_verify_round_response():
    runner = DSSDSessionRunner()

    result = runner.dssd_verify_round(
        VerifyRoundRequest(
            binding_id="bind-1",
            verifier_session_id="vs-1",
            seq_no=2,
            prefix_delta_token_ids=[4],
            draft_token_ids=[7, 8],
            q_values=[0.6, 0.4],
        )
    )

    assert isinstance(result, VerifyRoundResponse)
    assert result.verifier_session_id == "vs-1"
    assert result.seq_no == 2
