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
protocol_module = sys.modules["vllm.v1.dssd.protocol"]
VerifyRoundRequest = protocol_module.VerifyRoundRequest
DSSDVerifierExecutionRequest = session_runner_module.DSSDVerifierExecutionRequest
VerifierSessionInitRequest = session_runner_module.VerifierSessionInitRequest
VerifierForwardResult = protocol_module.VerifierForwardResult
build_verifier_result = verifier_runner_module.build_verifier_result
build_verifier_result_from_logits = verifier_runner_module.build_verifier_result_from_logits
build_verifier_replay_request_view = (
    verifier_runner_module.build_verifier_replay_request_view
)
build_verifier_replay_scheduler_output = (
    verifier_runner_module.build_verifier_replay_scheduler_output
)
extract_forward_probs = verifier_runner_module.extract_forward_probs
run_verifier_replay_forward = (
    verifier_runner_module.run_verifier_replay_forward
)


def test_protocol_module_is_shared_with_session_runner_imports():
    assert session_runner_module.VerifyRoundRequest is protocol_module.VerifyRoundRequest
    assert (session_runner_module.DSSDVerifierExecutionRequest
            is protocol_module.DSSDVerifierExecutionRequest)


def test_build_verifier_result_preserves_forward_prob_slices():
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-1",
        seq_no=2,
        committed_token_ids=[1, 2, 3, 4],
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


def test_session_runner_verify_round_selects_non_none_collective_reply():
    expected = VerifierForwardResult(
        verifier_session_id="vs-pp",
        seq_no=3,
        seq_probs=[[0.4, 0.6]],
        bonus_probs=[0.7, 0.3],
        finish_reason="gpu-replay-forward",
    )
    model_executor = types.SimpleNamespace(
        collective_rpc=Mock(return_value=[None, expected, None]),
    )
    runner = DSSDSessionRunner(model_executor=model_executor)
    runner.create_verifier_session(
        VerifierSessionInitRequest(
            verifier_session_id="vs-pp",
            binding_id="bind-1",
            prompt_token_ids=[1, 2],
            sampling_params_digest="sp-1",
        )
    )
    request = VerifyRoundRequest(
        binding_id="bind-1",
        verifier_session_id="vs-pp",
        seq_no=3,
        prefix_delta_token_ids=[4],
        draft_token_ids=[9],
        q_values=[0.75],
    )

    result = runner.dssd_verify_round(request)

    assert result == expected


def test_session_runner_verify_round_rejects_multiple_collective_replies():
    reply_a = VerifierForwardResult(
        verifier_session_id="vs-pp",
        seq_no=4,
        seq_probs=[[0.4, 0.6]],
        bonus_probs=[0.7, 0.3],
        finish_reason="gpu-replay-forward",
    )
    reply_b = VerifierForwardResult(
        verifier_session_id="vs-pp",
        seq_no=4,
        seq_probs=[[0.6, 0.4]],
        bonus_probs=[0.2, 0.8],
        finish_reason="gpu-replay-forward",
    )
    model_executor = types.SimpleNamespace(
        collective_rpc=Mock(return_value=[reply_a, None, reply_b]),
    )
    runner = DSSDSessionRunner(model_executor=model_executor)
    runner.create_verifier_session(
        VerifierSessionInitRequest(
            verifier_session_id="vs-pp",
            binding_id="bind-1",
            prompt_token_ids=[1, 2],
            sampling_params_digest="sp-1",
        )
    )
    request = VerifyRoundRequest(
        binding_id="bind-1",
        verifier_session_id="vs-pp",
        seq_no=4,
        prefix_delta_token_ids=[],
        draft_token_ids=[9],
        q_values=[0.75],
    )

    with pytest.raises(RuntimeError, match="exactly one verifier reply"):
        runner.dssd_verify_round(request)


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


def test_build_verifier_replay_request_view_preserves_prefix_and_spec_layout():
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-1",
        seq_no=3,
        committed_token_ids=[11, 12, 13],
        draft_token_ids=[21, 22],
        q_values=[0.6, 0.4],
    )

    view = build_verifier_replay_request_view(
        request,
        block_sizes=(4, 8),
    )

    assert view.request_id == "dssd-verify:vs-1:3"
    assert view.prompt_token_ids == [11, 12, 13]
    assert view.spec_token_ids == [21, 22]
    assert view.num_scheduled_tokens == 5
    assert view.block_ids == ([0, 1], [0])


def test_build_verifier_replay_scheduler_output_uses_committed_prompt_and_spec_tokens():
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-2",
        seq_no=4,
        committed_token_ids=[101, 102, 103],
        draft_token_ids=[201, 202],
        q_values=[0.3, 0.7],
    )

    view, scheduler_output = build_verifier_replay_scheduler_output(
        request,
        block_sizes=(4,),
    )

    assert scheduler_output.num_scheduled_tokens == {view.request_id: 5}
    assert scheduler_output.total_num_scheduled_tokens == 5
    assert scheduler_output.scheduled_spec_decode_tokens == {
        view.request_id: [201, 202],
    }

    new_req = scheduler_output.scheduled_new_reqs[0]
    assert new_req.req_id == "dssd-verify:vs-2:4"
    assert new_req.prompt_token_ids == [101, 102, 103]
    assert new_req.block_ids == ([0, 1],)
    assert new_req.num_computed_tokens == 0
    assert new_req.sampling_params.temperature == 0.0
    assert new_req.sampling_params.max_tokens == 1


def test_build_verifier_replay_request_view_rejects_empty_committed_prefix():
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-3",
        seq_no=0,
        committed_token_ids=[],
        draft_token_ids=[7],
        q_values=[0.9],
    )

    with pytest.raises(ValueError, match="committed_token_ids"):
        build_verifier_replay_request_view(request, block_sizes=(16,))


def test_build_verifier_replay_request_view_rejects_empty_draft_token_ids():
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-4",
        seq_no=1,
        committed_token_ids=[1],
        draft_token_ids=[],
        q_values=[],
    )

    with pytest.raises(ValueError, match="draft_token_ids"):
        build_verifier_replay_request_view(request, block_sizes=(16,))


def test_build_verifier_replay_request_view_rejects_mismatched_q_values():
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-5",
        seq_no=1,
        committed_token_ids=[1],
        draft_token_ids=[2, 3],
        q_values=[0.7],
    )

    with pytest.raises(ValueError, match="q_values"):
        build_verifier_replay_request_view(request, block_sizes=(16,))


def test_build_verifier_replay_request_view_rejects_empty_block_sizes():
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-6",
        seq_no=1,
        committed_token_ids=[1],
        draft_token_ids=[2],
        q_values=[0.5],
    )

    with pytest.raises(ValueError, match="block_sizes"):
        build_verifier_replay_request_view(request, block_sizes=())


def test_build_verifier_replay_scheduler_output_keeps_view_snapshot_stable():
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-7",
        seq_no=8,
        committed_token_ids=[11, 12],
        draft_token_ids=[21, 22],
        q_values=[0.1, 0.9],
    )

    view, scheduler_output = build_verifier_replay_scheduler_output(
        request,
        block_sizes=(4, 8),
    )

    scheduler_output.scheduled_new_reqs[0].prompt_token_ids.append(99)
    scheduler_output.scheduled_new_reqs[0].block_ids[0].append(7)
    scheduler_output.scheduled_spec_decode_tokens[view.request_id].append(33)

    assert view.prompt_token_ids == [11, 12]
    assert view.spec_token_ids == [21, 22]
    assert view.block_ids == ([0], [0])


def test_run_verifier_replay_forward_executes_model_and_cleans_up_batch_state():
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-4",
        seq_no=6,
        committed_token_ids=[1, 2, 3],
        draft_token_ids=[4, 5],
        q_values=[0.2, 0.8],
    )
    metadata = types.SimpleNamespace(
        target_logits_indices=torch.tensor([0, 1], dtype=torch.int32),
        bonus_logits_indices=torch.tensor([2], dtype=torch.int32),
    )
    fake_runner = types.SimpleNamespace(
        use_async_scheduling=False,
        is_pooling_model=False,
        supports_mm_inputs=False,
        kv_cache_config=types.SimpleNamespace(
            kv_cache_groups=[
                types.SimpleNamespace(
                    kv_cache_spec=types.SimpleNamespace(block_size=16)
                )
            ]
        ),
        execute_model=Mock(return_value=None),
        execute_model_state=types.SimpleNamespace(
            logits=torch.tensor(
                [
                    [4.0, 0.0],
                    [0.0, 4.0],
                    [2.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            spec_decode_metadata=metadata,
        ),
        input_batch=types.SimpleNamespace(
            prev_sampled_token_ids="stale",
            remove_request=Mock(return_value=0),
            condense=Mock(),
        ),
        requests={"dssd-verify:vs-4:6": object()},
        num_prompt_logprobs={"dssd-verify:vs-4:6": 1},
        late_interaction_runner=types.SimpleNamespace(
            on_requests_finished=Mock(),
        ),
        _draft_token_ids=[99],
        _draft_token_req_ids=["old-req"],
    )

    result = run_verifier_replay_forward(fake_runner, request)

    fake_runner.execute_model.assert_called_once()
    fake_runner.input_batch.remove_request.assert_called_once_with(
        "dssd-verify:vs-4:6"
    )
    fake_runner.input_batch.condense.assert_called_once_with()
    fake_runner.late_interaction_runner.on_requests_finished.assert_called_once_with(
        {"dssd-verify:vs-4:6"}
    )
    assert "dssd-verify:vs-4:6" not in fake_runner.requests
    assert "dssd-verify:vs-4:6" not in fake_runner.num_prompt_logprobs
    assert fake_runner.execute_model_state is None
    assert fake_runner.input_batch.prev_sampled_token_ids is None
    assert fake_runner._draft_token_ids is None
    assert fake_runner._draft_token_req_ids is None
    assert result.verifier_session_id == "vs-4"
    assert result.seq_no == 6
    assert len(result.seq_probs) == 2


def test_run_verifier_replay_forward_rejects_async_scheduling():
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-5",
        seq_no=0,
        committed_token_ids=[7],
        draft_token_ids=[8],
        q_values=[0.5],
    )
    fake_runner = types.SimpleNamespace(use_async_scheduling=True)

    with pytest.raises(NotImplementedError, match="async scheduling"):
        run_verifier_replay_forward(fake_runner, request)


def test_run_verifier_replay_forward_cleans_up_when_execute_model_returns_value():
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-6",
        seq_no=2,
        committed_token_ids=[1, 2],
        draft_token_ids=[3],
        q_values=[0.9],
    )
    fake_runner = types.SimpleNamespace(
        use_async_scheduling=False,
        is_pooling_model=False,
        supports_mm_inputs=False,
        kv_cache_config=types.SimpleNamespace(
            kv_cache_groups=[
                types.SimpleNamespace(
                    kv_cache_spec=types.SimpleNamespace(block_size=16)
                )
            ]
        ),
        execute_model=Mock(return_value=object()),
        execute_model_state="stale-state",
        input_batch=types.SimpleNamespace(
            prev_sampled_token_ids="stale",
            remove_request=Mock(return_value=0),
            condense=Mock(),
        ),
        requests={"dssd-verify:vs-6:2": object()},
        num_prompt_logprobs={"dssd-verify:vs-6:2": 1},
        late_interaction_runner=types.SimpleNamespace(
            on_requests_finished=Mock(),
        ),
        _draft_token_ids=[99],
        _draft_token_req_ids=["old-req"],
    )

    with pytest.raises(RuntimeError, match="cache logits state"):
        run_verifier_replay_forward(fake_runner, request)

    fake_runner.input_batch.remove_request.assert_called_once_with(
        "dssd-verify:vs-6:2"
    )
    fake_runner.input_batch.condense.assert_called_once_with()
    fake_runner.late_interaction_runner.on_requests_finished.assert_called_once_with(
        {"dssd-verify:vs-6:2"}
    )
    assert "dssd-verify:vs-6:2" not in fake_runner.requests
    assert "dssd-verify:vs-6:2" not in fake_runner.num_prompt_logprobs
    assert fake_runner.execute_model_state is None
    assert fake_runner.input_batch.prev_sampled_token_ids is None
    assert fake_runner._draft_token_ids is None
    assert fake_runner._draft_token_req_ids is None


def test_run_verifier_replay_forward_requires_cached_logits():
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-7",
        seq_no=1,
        committed_token_ids=[1, 2],
        draft_token_ids=[3],
        q_values=[0.6],
    )
    fake_runner = types.SimpleNamespace(
        use_async_scheduling=False,
        is_pooling_model=False,
        supports_mm_inputs=False,
        kv_cache_config=types.SimpleNamespace(
            kv_cache_groups=[
                types.SimpleNamespace(
                    kv_cache_spec=types.SimpleNamespace(block_size=16)
                )
            ]
        ),
        execute_model=Mock(return_value=None),
        execute_model_state=types.SimpleNamespace(
            logits=None,
            spec_decode_metadata=types.SimpleNamespace(
                target_logits_indices=torch.tensor([0], dtype=torch.int32),
                bonus_logits_indices=torch.tensor([1], dtype=torch.int32),
            ),
        ),
        input_batch=types.SimpleNamespace(
            prev_sampled_token_ids="stale",
            remove_request=Mock(return_value=0),
            condense=Mock(),
        ),
        requests={"dssd-verify:vs-7:1": object()},
        num_prompt_logprobs={"dssd-verify:vs-7:1": 1},
        late_interaction_runner=types.SimpleNamespace(
            on_requests_finished=Mock(),
        ),
        _draft_token_ids=[99],
        _draft_token_req_ids=["old-req"],
    )

    with pytest.raises(RuntimeError, match="cached logits"):
        run_verifier_replay_forward(fake_runner, request)


def test_run_verifier_replay_forward_non_output_pp_rank_returns_none_and_cleans_up(
    monkeypatch,
):
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-8",
        seq_no=2,
        committed_token_ids=[1, 2],
        draft_token_ids=[3],
        q_values=[0.5],
    )
    fake_runner = types.SimpleNamespace(
        use_async_scheduling=False,
        is_pooling_model=False,
        supports_mm_inputs=False,
        kv_cache_config=types.SimpleNamespace(
            kv_cache_groups=[
                types.SimpleNamespace(
                    kv_cache_spec=types.SimpleNamespace(block_size=16)
                )
            ]
        ),
        execute_model=Mock(return_value=object()),
        execute_model_state="stale-state",
        input_batch=types.SimpleNamespace(
            prev_sampled_token_ids="stale",
            remove_request=Mock(return_value=0),
            condense=Mock(),
        ),
        requests={"dssd-verify:vs-8:2": object()},
        num_prompt_logprobs={"dssd-verify:vs-8:2": 1},
        late_interaction_runner=types.SimpleNamespace(
            on_requests_finished=Mock(),
        ),
        _draft_token_ids=[99],
        _draft_token_req_ids=["old-req"],
    )
    monkeypatch.setattr(
        verifier_runner_module,
        "get_pp_group",
        lambda: types.SimpleNamespace(is_last_rank=False),
        raising=False,
    )

    result = run_verifier_replay_forward(fake_runner, request)

    assert result is None
    fake_runner.input_batch.remove_request.assert_called_once_with(
        "dssd-verify:vs-8:2"
    )
    fake_runner.input_batch.condense.assert_called_once_with()
    fake_runner.late_interaction_runner.on_requests_finished.assert_called_once_with(
        {"dssd-verify:vs-8:2"}
    )
    assert "dssd-verify:vs-8:2" not in fake_runner.requests
    assert "dssd-verify:vs-8:2" not in fake_runner.num_prompt_logprobs
    assert fake_runner.execute_model_state is None
    assert fake_runner.input_batch.prev_sampled_token_ids is None
    assert fake_runner._draft_token_ids is None
    assert fake_runner._draft_token_req_ids is None


def test_run_verifier_replay_forward_non_output_pp_rank_ignores_broadcast_logits_and_cleans_up(  # noqa: E501
    monkeypatch,
):
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-9",
        seq_no=3,
        committed_token_ids=[1, 2, 3],
        draft_token_ids=[4, 5],
        q_values=[0.2, 0.8],
    )
    metadata = types.SimpleNamespace(
        target_logits_indices=torch.tensor([0, 1], dtype=torch.int32),
        bonus_logits_indices=torch.tensor([2], dtype=torch.int32),
    )
    fake_runner = types.SimpleNamespace(
        use_async_scheduling=False,
        is_pooling_model=False,
        supports_mm_inputs=False,
        kv_cache_config=types.SimpleNamespace(
            kv_cache_groups=[
                types.SimpleNamespace(
                    kv_cache_spec=types.SimpleNamespace(block_size=16)
                )
            ]
        ),
        execute_model=Mock(return_value=None),
        execute_model_state=types.SimpleNamespace(
            logits=torch.tensor(
                [
                    [4.0, 0.0],
                    [0.0, 4.0],
                    [2.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            spec_decode_metadata=metadata,
        ),
        input_batch=types.SimpleNamespace(
            prev_sampled_token_ids="stale",
            remove_request=Mock(return_value=0),
            condense=Mock(),
        ),
        requests={"dssd-verify:vs-9:3": object()},
        num_prompt_logprobs={"dssd-verify:vs-9:3": 1},
        late_interaction_runner=types.SimpleNamespace(
            on_requests_finished=Mock(),
        ),
        _draft_token_ids=[99],
        _draft_token_req_ids=["old-req"],
    )
    monkeypatch.setattr(
        verifier_runner_module,
        "get_pp_group",
        lambda: types.SimpleNamespace(is_last_rank=False),
        raising=False,
    )

    result = run_verifier_replay_forward(fake_runner, request)

    assert result is None
    fake_runner.execute_model.assert_called_once()
    fake_runner.input_batch.remove_request.assert_called_once_with(
        "dssd-verify:vs-9:3"
    )
    fake_runner.input_batch.condense.assert_called_once_with()
    fake_runner.late_interaction_runner.on_requests_finished.assert_called_once_with(
        {"dssd-verify:vs-9:3"}
    )
    assert "dssd-verify:vs-9:3" not in fake_runner.requests
    assert "dssd-verify:vs-9:3" not in fake_runner.num_prompt_logprobs
    assert fake_runner.execute_model_state is None
    assert fake_runner.input_batch.prev_sampled_token_ids is None
    assert fake_runner._draft_token_ids is None
    assert fake_runner._draft_token_req_ids is None
