# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

import torch


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
verifier_runner_module = _load_module(
    "vllm.v1.dssd.worker.verifier_runner",
    DSSD_DIR / "worker" / "verifier_runner.py",
)
session_runner_module = _load_module(
    "vllm.v1.dssd.engine.session_runner",
    DSSD_DIR / "engine" / "session_runner.py",
)
DSSDSessionRunner = session_runner_module.DSSDSessionRunner
protocol_module = _load_module("vllm.v1.dssd.protocol", DSSD_DIR / "protocol.py")
VerifyRoundRequest = protocol_module.VerifyRoundRequest
DSSDVerifierExecutionRequest = session_runner_module.DSSDVerifierExecutionRequest
VerifierSessionInitRequest = session_runner_module.VerifierSessionInitRequest
VerifierForwardResult = protocol_module.VerifierForwardResult
build_verifier_result = verifier_runner_module.build_verifier_result
build_verifier_result_from_logits = verifier_runner_module.build_verifier_result_from_logits
extract_forward_probs = verifier_runner_module.extract_forward_probs


def test_build_verifier_result_preserves_forward_prob_slices():
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-1",
        seq_no=2,
        committed_token_ids=[1, 2, 3, 4],
        prefix_delta_token_ids=[4],
        draft_token_ids=[7, 8],
        q_values=[0.6, 0.4],
    )

    result = build_verifier_result(
        request=request,
        target_probs=[[0.2, 0.8], [0.9, 0.1], [0.7, 0.3]],
        finished=True,
        finish_reason="stop",
    )

    assert result.verifier_session_id == "vs-1"
    assert result.seq_no == 2
    assert result.seq_probs == [[0.2, 0.8], [0.9, 0.1]]
    assert result.bonus_probs == [0.7, 0.3]
    assert result.finished is True
    assert result.finish_reason == "stop"


def test_extract_forward_probs_uses_target_and_bonus_indices():
    metadata = types.SimpleNamespace(
        target_logits_indices=torch.tensor([1, 3], dtype=torch.int32),
        bonus_logits_indices=torch.tensor([0], dtype=torch.int32),
    )
    logits = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [5.0, 5.0],
            [2.0, 0.0],
        ],
        dtype=torch.float32,
    )

    seq_probs, bonus_probs = extract_forward_probs(logits, metadata)

    assert len(seq_probs) == 2
    assert bonus_probs == [0.5, 0.5]
    assert seq_probs[0][1] > seq_probs[0][0]
    assert seq_probs[1][0] > seq_probs[1][1]


def test_build_verifier_result_from_logits_composes_helper_steps():
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-2",
        seq_no=4,
        committed_token_ids=[1, 2],
        prefix_delta_token_ids=[],
        draft_token_ids=[0, 1],
        q_values=[0.2, 0.4],
    )
    metadata = types.SimpleNamespace(
        target_logits_indices=torch.tensor([1, 3], dtype=torch.int32),
        bonus_logits_indices=torch.tensor([0], dtype=torch.int32),
    )
    logits = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [5.0, 5.0],
            [2.0, 0.0],
        ],
        dtype=torch.float32,
    )

    result = build_verifier_result_from_logits(
        request=request,
        logits=logits,
        metadata=metadata,
        finish_reason="spec-forward",
    )

    assert result.verifier_session_id == "vs-2"
    assert result.seq_no == 4
    assert len(result.seq_probs) == 2
    assert result.bonus_probs == [0.7310585975646973, 0.2689414322376251]
    assert result.finish_reason == "spec-forward"


def test_session_runner_uses_collective_rpc_for_verify_round():
    expected = VerifierForwardResult(
        verifier_session_id="vs-1",
        seq_no=2,
        seq_probs=[[0.1, 0.9], [0.2, 0.8]],
        bonus_probs=[0.3, 0.7],
        finished=True,
        finish_reason="rpc-result",
    )
    model_executor = types.SimpleNamespace(
        collective_rpc=Mock(return_value=[expected]),
    )
    runner = DSSDSessionRunner(model_executor=model_executor)
    runner.create_verifier_session(
        VerifierSessionInitRequest(
            verifier_session_id="vs-1",
            binding_id="bind-1",
            prompt_token_ids=[1, 2, 3],
            sampling_params_digest="sp-1",
        )
    )
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
    model_executor.collective_rpc.assert_called_once()
    called_request = model_executor.collective_rpc.call_args.kwargs["args"][0]
    assert isinstance(called_request, DSSDVerifierExecutionRequest)
    assert called_request.committed_token_ids == [1, 2, 3, 4]
    assert called_request.draft_token_ids == [7, 8]
    assert called_request.q_values == [0.6, 0.4]
    model_executor.collective_rpc.assert_called_once_with(
        "dssd_verify_round",
        args=(called_request,),
    )


def test_session_runner_keeps_committed_prefix_stable_across_retry():
    class _FailOnceExecutor:
        def __init__(self) -> None:
            self.calls = 0
            self.requests = []

        def collective_rpc(self, method, args=(), **kwargs):
            del method, kwargs
            self.calls += 1
            self.requests.append(args[0])
            if self.calls == 1:
                raise RuntimeError("temporary failure")
            return [VerifierForwardResult(
                verifier_session_id="vs-1",
                seq_no=0,
                seq_probs=[[0.0, 1.0]],
                bonus_probs=[1.0, 0.0],
            )]

    runner = DSSDSessionRunner(model_executor=_FailOnceExecutor())
    runner.create_verifier_session(
        VerifierSessionInitRequest(
            verifier_session_id="vs-1",
            binding_id="bind-1",
            prompt_token_ids=[1],
            sampling_params_digest="sp-1",
        )
    )
    request = VerifyRoundRequest(
        binding_id="bind-1",
        verifier_session_id="vs-1",
        seq_no=0,
        prefix_delta_token_ids=[9],
        draft_token_ids=[1],
        q_values=[0.5],
    )

    with pytest.raises(RuntimeError, match="temporary failure"):
        runner.dssd_verify_round(request)

    state = runner.verifier_sessions.get("vs-1")
    assert state is not None
    assert state.committed_token_ids == [1]

    runner.dssd_verify_round(request)
    assert state.committed_token_ids == [1, 9]
    assert [req.committed_token_ids for req in runner.model_executor.requests] == [
        [1, 9],
        [1, 9],
    ]
