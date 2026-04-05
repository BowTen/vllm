# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from vllm.v1.dssd.protocol import DraftRoundRequest, VerifyRoundRequest, VerifierForwardResult
from vllm.v1.dssd.worker.draft_runner import DraftRoundResult
from vllm.v1.serial_utils import run_method
from vllm.v1.worker.worker_base import WorkerBase, WorkerWrapperBase


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
    request = VerifyRoundRequest(
        binding_id="bind-1",
        verifier_session_id="vs-1",
        seq_no=2,
        prefix_delta_token_ids=[4],
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
