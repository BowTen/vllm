# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock, call

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
_install_package_stub("vllm.v1.sample", VLLM_DIR / "v1" / "sample")
_install_package_stub("vllm.v1.sample.ops", VLLM_DIR / "v1" / "sample" / "ops")
_install_package_stub("vllm.v1.dssd", DSSD_DIR)
_install_package_stub("vllm.v1.dssd.edge", DSSD_EDGE_DIR)
_install_package_stub("vllm.v1.dssd.engine", DSSD_ENGINE_DIR)
_install_package_stub("vllm.v1.dssd.worker", DSSD_DIR / "worker")

distributed_module = types.ModuleType("vllm.distributed")
distributed_module.get_pp_group = Mock(side_effect=AssertionError)
sys.modules["vllm.distributed"] = distributed_module

topk_topp_sampler_module = types.ModuleType("vllm.v1.sample.ops.topk_topp_sampler")
topk_topp_sampler_module.apply_top_k_top_p = lambda logits, *_args: logits
sys.modules["vllm.v1.sample.ops.topk_topp_sampler"] = topk_topp_sampler_module

sampler_module = types.ModuleType("vllm.v1.sample.sampler")


class _Sampler:
    def __init__(self, logprobs_mode="raw_logprobs") -> None:
        self.logprobs_mode = logprobs_mode

    def __call__(self, **_kwargs):
        raise NotImplementedError


sampler_module.Sampler = _Sampler
sys.modules["vllm.v1.sample.sampler"] = sampler_module

sampling_params_module = types.ModuleType("vllm.sampling_params")


class _SamplingParams:
    def __init__(
        self,
        *,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        seed=None,
        min_tokens=0,
        max_tokens=16,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.seed = seed
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def clone(self):
        return _SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            seed=self.seed,
            min_tokens=self.min_tokens,
            max_tokens=self.max_tokens,
        )


sampling_params_module.SamplingParams = _SamplingParams
sys.modules["vllm.sampling_params"] = sampling_params_module
sys.modules["vllm"].SamplingParams = _SamplingParams

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
DSSDVerifierReplayState = verifier_runner_module.DSSDVerifierReplayState
DSSDVerifierReplaySessionState = (
    verifier_runner_module.DSSDVerifierReplaySessionState
)
build_verifier_result = verifier_runner_module.build_verifier_result
build_verifier_result_from_logits = verifier_runner_module.build_verifier_result_from_logits
build_verifier_replay_request_view = (
    verifier_runner_module.build_verifier_replay_request_view
)
build_verifier_replay_state = verifier_runner_module.build_verifier_replay_state
build_verifier_replay_scheduler_output = (
    verifier_runner_module.build_verifier_replay_scheduler_output
)
extract_forward_probs = verifier_runner_module.extract_forward_probs
run_verifier_replay_forward = (
    verifier_runner_module.run_verifier_replay_forward
)
run_verifier_replay_forward_batch = (
    verifier_runner_module.run_verifier_replay_forward_batch
)


def _make_fake_replay_runner(*, block_size: int = 16):
    return types.SimpleNamespace(
        use_async_scheduling=False,
        is_pooling_model=False,
        supports_mm_inputs=False,
        kv_cache_config=types.SimpleNamespace(
            kv_cache_groups=[
                types.SimpleNamespace(
                    kv_cache_spec=types.SimpleNamespace(block_size=block_size)
                )
            ]
        ),
        execute_model_state=None,
        input_batch=types.SimpleNamespace(
            prev_sampled_token_ids="stale",
            remove_request=Mock(return_value=0),
            condense=Mock(),
        ),
        requests={},
        num_prompt_logprobs={},
        late_interaction_runner=types.SimpleNamespace(
            on_requests_finished=Mock(),
        ),
        kv_connector_output=object(),
        execute_verifier_replay_request=Mock(),
        _draft_token_ids=[99],
        _draft_token_req_ids=["old-req"],
    )


def _install_cached_request(
    fake_runner,
    request_id: str,
    *,
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    num_computed_tokens: int,
    block_ids: tuple[list[int], ...],
):
    fake_runner.requests[request_id] = types.SimpleNamespace(
        prompt_token_ids=list(prompt_token_ids),
        prompt_embeds=None,
        output_token_ids=list(output_token_ids),
        num_computed_tokens=num_computed_tokens,
        block_ids=tuple(list(ids) for ids in block_ids),
    )


def _cleanup_scratch_request(fake_runner, request_id: str) -> None:
    fake_runner.input_batch.remove_request(request_id)
    fake_runner.input_batch.condense()
    fake_runner.requests.pop(request_id, None)
    fake_runner.num_prompt_logprobs.pop(request_id, None)


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


def test_session_runner_verify_round_allows_empty_prompt_on_first_round():
    expected = VerifierForwardResult(
        verifier_session_id="vs-empty",
        seq_no=0,
        seq_probs=[[0.4, 0.6]],
        bonus_probs=[0.7, 0.3],
        finish_reason="gpu-replay-forward",
    )
    model_executor = types.SimpleNamespace(
        collective_rpc=Mock(return_value=[expected]),
    )
    runner = DSSDSessionRunner(model_executor=model_executor)
    runner.create_verifier_session(
        VerifierSessionInitRequest(
            verifier_session_id="vs-empty",
            binding_id="bind-1",
            prompt_token_ids=[],
            sampling_params_digest="sp-empty",
        )
    )
    request = VerifyRoundRequest(
        binding_id="bind-1",
        verifier_session_id="vs-empty",
        seq_no=0,
        prefix_delta_token_ids=[],
        draft_token_ids=[9],
        q_values=[0.75],
    )

    result = runner.dssd_verify_round(request)

    assert result == expected
    called_request = model_executor.collective_rpc.call_args.kwargs["args"][0]
    assert called_request.committed_token_ids == []


def test_session_runner_uses_collective_rpc_for_verify_round_batch():
    expected_a = VerifierForwardResult(
        verifier_session_id="vs-batch-a",
        seq_no=0,
        seq_probs=[[0.1, 0.9]],
        bonus_probs=[0.3, 0.7],
        finish_reason="gpu-replay-forward",
    )
    expected_b = VerifierForwardResult(
        verifier_session_id="vs-batch-b",
        seq_no=0,
        seq_probs=[[0.2, 0.8], [0.4, 0.6]],
        bonus_probs=[0.5, 0.5],
        finish_reason="gpu-replay-forward",
    )
    model_executor = types.SimpleNamespace(
        collective_rpc=Mock(return_value=[[expected_a, expected_b]]),
    )
    runner = DSSDSessionRunner(model_executor=model_executor)
    runner.create_verifier_session(
        VerifierSessionInitRequest(
            verifier_session_id="vs-batch-a",
            binding_id="bind-1",
            prompt_token_ids=[1, 2],
            sampling_params_digest="sp-shared",
        )
    )
    runner.create_verifier_session(
        VerifierSessionInitRequest(
            verifier_session_id="vs-batch-b",
            binding_id="bind-2",
            prompt_token_ids=[3],
            sampling_params_digest="sp-shared",
        )
    )

    results = runner.dssd_verify_round_batch([
        VerifyRoundRequest(
            binding_id="bind-1",
            verifier_session_id="vs-batch-a",
            seq_no=0,
            prefix_delta_token_ids=[4],
            draft_token_ids=[7],
            q_values=[0.6],
        ),
        VerifyRoundRequest(
            binding_id="bind-2",
            verifier_session_id="vs-batch-b",
            seq_no=0,
            prefix_delta_token_ids=[],
            draft_token_ids=[8, 9],
            q_values=[0.4, 0.5],
        ),
    ])

    assert results == [expected_a, expected_b]
    model_executor.collective_rpc.assert_called_once()
    called_requests = model_executor.collective_rpc.call_args.kwargs["args"][0]
    assert [request.verifier_session_id for request in called_requests] == [
        "vs-batch-a",
        "vs-batch-b",
    ]
    assert called_requests[0].committed_token_ids == [1, 2, 4]
    assert called_requests[1].committed_token_ids == [3]


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

    assert view.request_id == "dssd-verify:vs-1"
    assert view.prompt_token_ids == [11, 12, 13]
    assert view.spec_token_ids == [21, 22]
    assert view.num_scheduled_tokens == 5
    assert view.block_ids == ([0, 1], [0])


def test_build_verifier_replay_state_preserves_request_and_scheduler_semantics():
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-state",
        seq_no=5,
        committed_token_ids=[11, 12, 13],
        draft_token_ids=[21, 22],
        q_values=[0.6, 0.4],
    )

    state = build_verifier_replay_state(
        request,
        block_sizes=(4, 8),
    )

    assert isinstance(state, DSSDVerifierReplayState)
    assert state.request_id == "dssd-verify:vs-state"
    assert state.verifier_session_id == "vs-state"
    assert state.seq_no == 5
    assert state.committed_token_ids == (11, 12, 13)
    assert state.draft_token_ids == (21, 22)
    assert state.q_values == (0.6, 0.4)
    assert state.scheduled_spec_decode_tokens == (21, 22)
    assert state.num_scheduled_tokens == 5
    assert state.block_ids == ((0, 1), (0,))

    view = state.request_view
    assert view.request_id == state.request_id
    assert view.seq_no == 5
    assert view.prompt_token_ids == [11, 12, 13]
    assert view.spec_token_ids == [21, 22]
    assert view.scheduled_spec_decode_tokens == [21, 22]
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
    assert new_req.req_id == "dssd-verify:vs-2"
    assert new_req.prompt_token_ids == [101, 102, 103]
    assert new_req.block_ids == ([0, 1],)
    assert new_req.num_computed_tokens == 0
    assert new_req.sampling_params.temperature == 0.0
    assert new_req.sampling_params.max_tokens == 1


def test_build_verifier_replay_state_request_view_and_scheduler_output_are_copies():
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-copy",
        seq_no=9,
        committed_token_ids=[11, 12],
        draft_token_ids=[21, 22],
        q_values=[0.1, 0.9],
    )

    state = build_verifier_replay_state(
        request,
        block_sizes=(4, 8),
    )
    view = state.request_view
    scheduler_output = state.scheduler_output

    view.prompt_token_ids.append(99)
    view.spec_token_ids.append(33)
    view.scheduled_spec_decode_tokens.append(44)
    view.block_ids[0].append(7)
    scheduler_output.scheduled_new_reqs[0].prompt_token_ids.append(66)
    scheduler_output.scheduled_new_reqs[0].block_ids[0].append(8)
    scheduler_output.scheduled_spec_decode_tokens[state.request_id].append(55)

    assert state.committed_token_ids == (11, 12)
    assert state.draft_token_ids == (21, 22)
    assert state.scheduled_spec_decode_tokens == (21, 22)
    assert state.block_ids == ((0,), (0,))


def test_build_verifier_replay_state_supports_session_stable_request_identity():
    request_round_1 = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-stable",
        seq_no=1,
        committed_token_ids=[11, 12],
        draft_token_ids=[21],
        q_values=[0.6],
    )
    request_round_2 = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-stable",
        seq_no=2,
        committed_token_ids=[11, 12, 13],
        draft_token_ids=[22, 23],
        q_values=[0.3, 0.7],
    )

    state_round_1 = build_verifier_replay_state(
        request_round_1,
        block_sizes=(4,),
    )
    state_round_2 = build_verifier_replay_state(
        request_round_2,
        block_sizes=(4,),
    )

    assert state_round_1.request_id == "dssd-verify:vs-stable"
    assert state_round_2.request_id == "dssd-verify:vs-stable"
    assert state_round_1.seq_no == 1
    assert state_round_2.seq_no == 2
    assert state_round_1.committed_token_ids == (11, 12)
    assert state_round_2.committed_token_ids == (11, 12, 13)
    assert state_round_1.draft_token_ids == (21,)
    assert state_round_2.draft_token_ids == (22, 23)


def test_build_verifier_replay_request_view_allows_empty_committed_prefix():
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-3",
        seq_no=0,
        committed_token_ids=[],
        draft_token_ids=[7, 8],
        q_values=[0.9, 0.1],
    )

    view = build_verifier_replay_request_view(request, block_sizes=(16,))

    assert view.prompt_token_ids == []
    assert view.spec_token_ids == [7, 8]
    assert view.num_scheduled_tokens == 2
    assert view.block_ids == ([0],)


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
    fake_runner = _make_fake_replay_runner()
    fake_runner._dssd_verifier_replay_sessions = {
        "vs-4": DSSDVerifierReplaySessionState(
            request_id="dssd-verify:vs-4",
            verifier_session_id="vs-4",
            seq_no=5,
            committed_token_ids=(1, 2, 3),
        )
    }

    def _execute_verifier_replay_request(
        *,
        request_id,
        committed_token_ids,
        draft_token_ids,
    ):
        assert draft_token_ids == [4, 5]
        _install_cached_request(
            fake_runner,
            request_id,
            prompt_token_ids=committed_token_ids,
            output_token_ids=[],
            num_computed_tokens=0,
            block_ids=([0],),
        )
        fake_runner.execute_model_state = types.SimpleNamespace(
            logits=torch.tensor(
                [
                    [4.0, 0.0],
                    [0.0, 4.0],
                    [2.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            spec_decode_metadata=metadata,
        )
        _cleanup_scratch_request(fake_runner, request_id)
        return None

    fake_runner.execute_verifier_replay_request.side_effect = (
        _execute_verifier_replay_request
    )

    result = run_verifier_replay_forward(fake_runner, request)

    fake_runner.execute_verifier_replay_request.assert_called_once_with(
        request_id="dssd-verify:vs-4",
        committed_token_ids=[1, 2, 3],
        draft_token_ids=[4, 5],
    )
    fake_runner.input_batch.remove_request.assert_called_once_with(
        "dssd-verify:vs-4"
    )
    fake_runner.input_batch.condense.assert_called_once_with()
    fake_runner.late_interaction_runner.on_requests_finished.assert_not_called()
    assert "dssd-verify:vs-4" not in fake_runner.requests
    assert fake_runner.num_prompt_logprobs == {}
    assert fake_runner.execute_model_state is None
    assert fake_runner.input_batch.prev_sampled_token_ids is None
    assert fake_runner.kv_connector_output is None
    assert fake_runner._draft_token_ids is None
    assert fake_runner._draft_token_req_ids is None
    assert result.verifier_session_id == "vs-4"
    assert result.seq_no == 6
    assert len(result.seq_probs) == 2


def test_run_verifier_replay_forward_round_2_rebuilds_scratch_request_from_session_summary():
    metadata = types.SimpleNamespace(
        target_logits_indices=torch.tensor([0], dtype=torch.int32),
        bonus_logits_indices=torch.tensor([1], dtype=torch.int32),
    )
    logits_per_round = [
        torch.tensor(
            [
                [4.0, 0.0],
                [0.0, 4.0],
            ],
            dtype=torch.float32,
        ),
        torch.tensor(
            [
                [0.0, 4.0],
                [4.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    ]
    execute_inputs = []

    fake_runner = _make_fake_replay_runner()

    def _execute_verifier_replay_request(
        *,
        request_id,
        committed_token_ids,
        draft_token_ids,
    ):
        execute_inputs.append(
            {
                "request_id": request_id,
                "committed_token_ids": list(committed_token_ids),
                "draft_token_ids": list(draft_token_ids),
            }
        )
        _install_cached_request(
            fake_runner,
            request_id,
            prompt_token_ids=committed_token_ids,
            output_token_ids=[],
            num_computed_tokens=0,
            block_ids=([0],),
        )
        fake_runner.execute_model_state = types.SimpleNamespace(
            logits=logits_per_round.pop(0),
            spec_decode_metadata=metadata,
        )
        _cleanup_scratch_request(fake_runner, request_id)
        return None

    fake_runner.execute_verifier_replay_request.side_effect = (
        _execute_verifier_replay_request
    )

    request_round_1 = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-cache",
        seq_no=0,
        committed_token_ids=[1, 2],
        draft_token_ids=[3],
        q_values=[0.2],
    )
    request_round_2 = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-cache",
        seq_no=2,
        committed_token_ids=[1, 2, 3],
        draft_token_ids=[4],
        q_values=[0.8],
    )

    result_round_1 = run_verifier_replay_forward(fake_runner, request_round_1)

    session_state = fake_runner._dssd_verifier_replay_sessions["vs-cache"]
    assert isinstance(session_state, DSSDVerifierReplaySessionState)
    assert session_state.request_id == "dssd-verify:vs-cache"
    assert session_state.seq_no == 0
    assert session_state.committed_token_ids == (1, 2)
    assert not hasattr(session_state, "replay_state")
    assert "dssd-verify:vs-cache" not in fake_runner.requests

    result_round_2 = run_verifier_replay_forward(fake_runner, request_round_2)

    assert fake_runner.execute_verifier_replay_request.call_count == 2
    assert execute_inputs == [
        {
            "request_id": "dssd-verify:vs-cache",
            "committed_token_ids": [1, 2],
            "draft_token_ids": [3],
        },
        {
            "request_id": "dssd-verify:vs-cache",
            "committed_token_ids": [1, 2, 3],
            "draft_token_ids": [4],
        },
    ]
    assert fake_runner.input_batch.remove_request.call_args_list == [
        call("dssd-verify:vs-cache"),
        call("dssd-verify:vs-cache"),
    ]
    assert fake_runner.input_batch.condense.call_count == 2
    fake_runner.late_interaction_runner.on_requests_finished.assert_not_called()
    assert session_state is fake_runner._dssd_verifier_replay_sessions["vs-cache"]
    assert session_state.seq_no == 2
    assert session_state.committed_token_ids == (1, 2, 3)
    assert "dssd-verify:vs-cache" not in fake_runner.requests
    assert fake_runner.execute_model_state is None
    assert fake_runner.input_batch.prev_sampled_token_ids is None
    assert fake_runner.kv_connector_output is None
    assert fake_runner._draft_token_ids is None
    assert fake_runner._draft_token_req_ids is None
    assert result_round_1 is not None
    assert result_round_1.seq_no == 0
    assert result_round_2 is not None
    assert result_round_2.seq_no == 2


def test_run_verifier_replay_forward_batch_executes_two_requests_in_one_scratch_batch():
    metadata = types.SimpleNamespace(
        num_draft_tokens=[1, 2],
        target_logits_indices=torch.tensor([0, 2, 3], dtype=torch.int32),
        bonus_logits_indices=torch.tensor([1, 4], dtype=torch.int32),
    )
    fake_runner = _make_fake_replay_runner()
    fake_runner.execute_verifier_replay_requests = Mock(return_value=None)
    request_a = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-batch-a",
        seq_no=0,
        committed_token_ids=[1, 2],
        draft_token_ids=[3],
        q_values=[0.1],
    )
    request_b = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-batch-b",
        seq_no=0,
        committed_token_ids=[4],
        draft_token_ids=[5, 6],
        q_values=[0.2, 0.8],
    )
    fake_runner.execute_model_state = types.SimpleNamespace(
        logits=torch.tensor(
            [
                [4.0, 0.0],
                [0.0, 4.0],
                [0.0, 3.0],
                [5.0, 0.0],
                [0.0, 5.0],
            ],
            dtype=torch.float32,
        ),
        spec_decode_metadata=metadata,
    )

    results = run_verifier_replay_forward_batch(
        fake_runner,
        [request_a, request_b],
    )

    fake_runner.execute_verifier_replay_requests.assert_called_once_with(
        replay_requests=[
            {
                "request_id": "dssd-verify:vs-batch-a",
                "committed_token_ids": [1, 2],
                "draft_token_ids": [3],
            },
            {
                "request_id": "dssd-verify:vs-batch-b",
                "committed_token_ids": [4],
                "draft_token_ids": [5, 6],
            },
        ]
    )
    assert [result.verifier_session_id for result in results] == [
        "vs-batch-a",
        "vs-batch-b",
    ]
    assert [result.seq_no for result in results] == [0, 0]
    assert [len(result.seq_probs) for result in results] == [1, 2]
    assert results[0].bonus_probs[1] > results[0].bonus_probs[0]
    assert results[1].bonus_probs[1] > results[1].bonus_probs[0]
    assert fake_runner.execute_model_state is None
    assert fake_runner.input_batch.prev_sampled_token_ids is None
    assert fake_runner.kv_connector_output is None
    assert fake_runner._draft_token_ids is None
    assert fake_runner._draft_token_req_ids is None
    assert "dssd-verify:vs-batch-a" not in fake_runner.requests
    assert "dssd-verify:vs-batch-b" not in fake_runner.requests
    assert fake_runner._dssd_verifier_replay_sessions["vs-batch-a"].seq_no == 0
    assert fake_runner._dssd_verifier_replay_sessions["vs-batch-b"].seq_no == 0


def test_sync_cached_verifier_request_state_updates_worker_and_batch_mirrors():
    request_id = "dssd-verify:vs-sync"
    fake_runner = _make_fake_replay_runner()
    _install_cached_request(
        fake_runner,
        request_id,
        prompt_token_ids=[11, 12],
        output_token_ids=[99],
        num_computed_tokens=1,
        block_ids=([7],),
    )
    fake_runner.input_batch.req_id_to_index = {request_id: 0}
    fake_runner.input_batch.num_computed_tokens_cpu = torch.tensor(
        [0], dtype=torch.int32
    )
    fake_runner.input_batch.num_tokens_no_spec = torch.tensor([0], dtype=torch.int32)
    fake_runner.input_batch.req_output_token_ids = [
        fake_runner.requests[request_id].output_token_ids
    ]
    fake_runner.input_batch.token_ids_cpu = torch.tensor(
        [[11, 12, 0, 0]], dtype=torch.int32
    )
    fake_runner.input_batch.block_table = types.SimpleNamespace(
        add_row=Mock(),
    )

    cached_block_ids, num_output_tokens = (
        verifier_runner_module._sync_cached_verifier_request_state(
            fake_runner,
            request_id=request_id,
            committed_token_ids=[11, 12, 13],
            block_sizes=(4,),
        )
    )

    assert cached_block_ids == ([0],)
    assert num_output_tokens == 1
    cached_request = fake_runner.requests[request_id]
    assert cached_request.output_token_ids == [13]
    assert cached_request.num_computed_tokens == 2
    assert fake_runner.input_batch.num_computed_tokens_cpu[0].item() == 2
    assert fake_runner.input_batch.num_tokens_no_spec[0].item() == 3
    assert fake_runner.input_batch.req_output_token_ids[0] == [13]
    assert fake_runner.input_batch.token_ids_cpu.tolist()[0][:3] == [11, 12, 13]
    fake_runner.input_batch.block_table.add_row.assert_called_once_with(
        ([0],),
        0,
    )


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
    fake_runner = _make_fake_replay_runner()
    _install_cached_request(
        fake_runner,
        "dssd-verify:vs-6",
        prompt_token_ids=[1, 2],
        output_token_ids=[],
        num_computed_tokens=2,
        block_ids=([0],),
    )
    fake_runner.num_prompt_logprobs["dssd-verify:vs-6"] = 1
    fake_runner._dssd_verifier_replay_sessions = {
        "vs-6": DSSDVerifierReplaySessionState(
            request_id="dssd-verify:vs-6",
            verifier_session_id="vs-6",
            seq_no=1,
            committed_token_ids=(1, 2),
        )
    }
    fake_runner.execute_verifier_replay_request = Mock(
        side_effect=lambda **kwargs: (
            _cleanup_scratch_request(fake_runner, kwargs["request_id"]),
            object(),
        )[1]
    )
    fake_runner.execute_model_state = "stale-state"

    with pytest.raises(RuntimeError, match="cache logits state"):
        run_verifier_replay_forward(fake_runner, request)

    assert fake_runner.input_batch.remove_request.call_args_list == [
        call("dssd-verify:vs-6"),
        call("dssd-verify:vs-6"),
    ]
    assert fake_runner.input_batch.condense.call_count == 2
    fake_runner.late_interaction_runner.on_requests_finished.assert_called_once_with(
        {"dssd-verify:vs-6"}
    )
    assert "dssd-verify:vs-6" not in fake_runner.requests
    assert "dssd-verify:vs-6" not in fake_runner.num_prompt_logprobs
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
    fake_runner = _make_fake_replay_runner()
    _install_cached_request(
        fake_runner,
        "dssd-verify:vs-7",
        prompt_token_ids=[1, 2],
        output_token_ids=[],
        num_computed_tokens=2,
        block_ids=([0],),
    )
    fake_runner.num_prompt_logprobs["dssd-verify:vs-7"] = 1
    fake_runner._dssd_verifier_replay_sessions = {
        "vs-7": DSSDVerifierReplaySessionState(
            request_id="dssd-verify:vs-7",
            verifier_session_id="vs-7",
            seq_no=0,
            committed_token_ids=(1, 2),
        )
    }
    fake_runner.execute_verifier_replay_request = Mock(return_value=None)
    fake_runner.execute_model_state = types.SimpleNamespace(
        logits=None,
        spec_decode_metadata=types.SimpleNamespace(
            target_logits_indices=torch.tensor([0], dtype=torch.int32),
            bonus_logits_indices=torch.tensor([1], dtype=torch.int32),
        ),
    )

    with pytest.raises(RuntimeError, match="cached logits"):
        run_verifier_replay_forward(fake_runner, request)


def test_run_verifier_replay_forward_without_session_summary_does_not_bootstrap_cached_request():
    fake_runner = _make_fake_replay_runner()
    _install_cached_request(
        fake_runner,
        "dssd-verify:vs-bootstrap",
        prompt_token_ids=[1, 2],
        output_token_ids=[3],
        num_computed_tokens=2,
        block_ids=([0],),
    )
    fake_runner.execute_verifier_replay_request = Mock()
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-bootstrap",
        seq_no=4,
        committed_token_ids=[1, 2, 3, 4],
        draft_token_ids=[5],
        q_values=[0.5],
    )

    with pytest.raises(RuntimeError, match="seq_no == 0"):
        run_verifier_replay_forward(fake_runner, request)

    fake_runner.execute_verifier_replay_request.assert_not_called()
    assert "vs-bootstrap" not in fake_runner._dssd_verifier_replay_sessions
    assert "dssd-verify:vs-bootstrap" not in fake_runner.requests


def test_run_verifier_replay_forward_seq_no_zero_resets_stale_session():
    metadata = types.SimpleNamespace(
        target_logits_indices=torch.tensor([0], dtype=torch.int32),
        bonus_logits_indices=torch.tensor([1], dtype=torch.int32),
    )
    fake_runner = _make_fake_replay_runner()
    _install_cached_request(
        fake_runner,
        "dssd-verify:vs-reset",
        prompt_token_ids=[9, 9],
        output_token_ids=[9],
        num_computed_tokens=3,
        block_ids=([0],),
    )
    fake_runner.num_prompt_logprobs["dssd-verify:vs-reset"] = 1
    fake_runner._dssd_verifier_replay_sessions = {
        "vs-reset": DSSDVerifierReplaySessionState(
            request_id="dssd-verify:vs-reset",
            verifier_session_id="vs-reset",
            seq_no=5,
            committed_token_ids=(9, 9, 9),
        )
    }
    old_session_state = fake_runner._dssd_verifier_replay_sessions["vs-reset"]

    def _execute_verifier_replay_request(
        *,
        request_id,
        committed_token_ids,
        draft_token_ids,
    ):
        assert draft_token_ids == [2]
        _install_cached_request(
            fake_runner,
            request_id,
            prompt_token_ids=committed_token_ids,
            output_token_ids=[],
            num_computed_tokens=0,
            block_ids=([0],),
        )
        fake_runner.execute_model_state = types.SimpleNamespace(
            logits=torch.tensor(
                [
                    [4.0, 0.0],
                    [0.0, 4.0],
                ],
                dtype=torch.float32,
            ),
            spec_decode_metadata=metadata,
        )
        _cleanup_scratch_request(fake_runner, request_id)
        return None

    fake_runner.execute_verifier_replay_request.side_effect = (
        _execute_verifier_replay_request
    )
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-reset",
        seq_no=0,
        committed_token_ids=[1],
        draft_token_ids=[2],
        q_values=[0.5],
    )

    result = run_verifier_replay_forward(fake_runner, request)

    assert result is not None
    fake_runner.execute_verifier_replay_request.assert_called_once_with(
        request_id="dssd-verify:vs-reset",
        committed_token_ids=[1],
        draft_token_ids=[2],
    )
    assert fake_runner.input_batch.remove_request.call_args_list == [
        call("dssd-verify:vs-reset"),
        call("dssd-verify:vs-reset"),
    ]
    fake_runner.late_interaction_runner.on_requests_finished.assert_called_once_with(
        {"dssd-verify:vs-reset"}
    )
    session_state = fake_runner._dssd_verifier_replay_sessions["vs-reset"]
    assert session_state is not old_session_state
    assert session_state.seq_no == 0
    assert session_state.committed_token_ids == (1,)
    assert "dssd-verify:vs-reset" not in fake_runner.requests


def test_run_verifier_replay_forward_non_extension_fails_closed():
    fake_runner = _make_fake_replay_runner()
    _install_cached_request(
        fake_runner,
        "dssd-verify:vs-branch",
        prompt_token_ids=[1, 2],
        output_token_ids=[3],
        num_computed_tokens=3,
        block_ids=([0],),
    )
    fake_runner.num_prompt_logprobs["dssd-verify:vs-branch"] = 1
    fake_runner._dssd_verifier_replay_sessions = {
        "vs-branch": DSSDVerifierReplaySessionState(
            request_id="dssd-verify:vs-branch",
            verifier_session_id="vs-branch",
            seq_no=2,
            committed_token_ids=(1, 2, 3),
        )
    }
    fake_runner.execute_verifier_replay_request = Mock()
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-branch",
        seq_no=3,
        committed_token_ids=[1, 4],
        draft_token_ids=[5],
        q_values=[0.5],
    )

    with pytest.raises(RuntimeError, match="committed prefix"):
        run_verifier_replay_forward(fake_runner, request)

    fake_runner.execute_verifier_replay_request.assert_not_called()
    fake_runner.input_batch.remove_request.assert_called_once_with(
        "dssd-verify:vs-branch"
    )
    fake_runner.input_batch.condense.assert_called_once_with()
    fake_runner.late_interaction_runner.on_requests_finished.assert_called_once_with(
        {"dssd-verify:vs-branch"}
    )
    assert "dssd-verify:vs-branch" not in fake_runner.requests
    assert "dssd-verify:vs-branch" not in fake_runner.num_prompt_logprobs
    assert "vs-branch" not in fake_runner._dssd_verifier_replay_sessions


def test_run_verifier_replay_forward_non_output_pp_rank_returns_none_and_cleans_scratch_request(
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
    fake_runner = _make_fake_replay_runner()
    _install_cached_request(
        fake_runner,
        "dssd-verify:vs-8",
        prompt_token_ids=[1, 2],
        output_token_ids=[],
        num_computed_tokens=2,
        block_ids=([0],),
    )
    fake_runner._dssd_verifier_replay_sessions = {
        "vs-8": DSSDVerifierReplaySessionState(
            request_id="dssd-verify:vs-8",
            verifier_session_id="vs-8",
            seq_no=1,
            committed_token_ids=(1, 2),
        )
    }
    fake_runner.execute_verifier_replay_request = Mock(
        side_effect=lambda **kwargs: (
            _cleanup_scratch_request(fake_runner, kwargs["request_id"]),
            object(),
        )[1]
    )
    fake_runner.execute_model_state = "stale-state"
    monkeypatch.setattr(
        verifier_runner_module,
        "get_pp_group",
        lambda: types.SimpleNamespace(is_last_rank=False),
        raising=False,
    )

    result = run_verifier_replay_forward(fake_runner, request)

    assert result is None
    fake_runner.input_batch.remove_request.assert_called_once_with(
        "dssd-verify:vs-8"
    )
    fake_runner.input_batch.condense.assert_called_once_with()
    fake_runner.late_interaction_runner.on_requests_finished.assert_not_called()
    assert "dssd-verify:vs-8" not in fake_runner.requests
    assert "dssd-verify:vs-8" not in fake_runner.num_prompt_logprobs
    assert fake_runner.execute_model_state is None
    assert fake_runner.input_batch.prev_sampled_token_ids is None
    assert fake_runner._draft_token_ids is None
    assert fake_runner._draft_token_req_ids is None


def test_run_verifier_replay_forward_non_output_pp_rank_ignores_broadcast_logits_and_cleans_scratch_request(  # noqa: E501
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
    fake_runner = _make_fake_replay_runner()
    _install_cached_request(
        fake_runner,
        "dssd-verify:vs-9",
        prompt_token_ids=[1, 2, 3],
        output_token_ids=[],
        num_computed_tokens=3,
        block_ids=([0],),
    )
    fake_runner._dssd_verifier_replay_sessions = {
        "vs-9": DSSDVerifierReplaySessionState(
            request_id="dssd-verify:vs-9",
            verifier_session_id="vs-9",
            seq_no=2,
            committed_token_ids=(1, 2, 3),
        )
    }
    fake_runner.execute_verifier_replay_request = Mock(
        side_effect=lambda **kwargs: (
            _cleanup_scratch_request(fake_runner, kwargs["request_id"]),
            None,
        )[1]
    )
    fake_runner.execute_model_state = types.SimpleNamespace(
        logits=torch.tensor(
            [
                [4.0, 0.0],
                [0.0, 4.0],
                [2.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        spec_decode_metadata=metadata,
    )
    monkeypatch.setattr(
        verifier_runner_module,
        "get_pp_group",
        lambda: types.SimpleNamespace(is_last_rank=False),
        raising=False,
    )

    result = run_verifier_replay_forward(fake_runner, request)

    assert result is None
    fake_runner.execute_verifier_replay_request.assert_called_once_with(
        request_id="dssd-verify:vs-9",
        committed_token_ids=[1, 2, 3],
        draft_token_ids=[4, 5],
    )
    fake_runner.input_batch.remove_request.assert_called_once_with(
        "dssd-verify:vs-9"
    )
    fake_runner.input_batch.condense.assert_called_once_with()
    fake_runner.late_interaction_runner.on_requests_finished.assert_not_called()
    assert "dssd-verify:vs-9" not in fake_runner.requests
    assert "dssd-verify:vs-9" not in fake_runner.num_prompt_logprobs
    assert fake_runner.execute_model_state is None
    assert fake_runner.input_batch.prev_sampled_token_ids is None
    assert fake_runner._draft_token_ids is None
    assert fake_runner._draft_token_req_ids is None


def test_close_verifier_replay_session_resets_summary_and_residue():
    request = sys.modules["vllm.v1.dssd.protocol"].CloseSessionRequest(
        verifier_session_id="vs-close",
        reason="done",
    )
    fake_runner = _make_fake_replay_runner()
    _install_cached_request(
        fake_runner,
        "dssd-verify:vs-close",
        prompt_token_ids=[1, 2],
        output_token_ids=[3],
        num_computed_tokens=2,
        block_ids=([0],),
    )
    fake_runner.num_prompt_logprobs["dssd-verify:vs-close"] = 1
    fake_runner._dssd_verifier_replay_sessions = {
        "vs-close": DSSDVerifierReplaySessionState(
            request_id="dssd-verify:vs-close",
            verifier_session_id="vs-close",
            seq_no=2,
            committed_token_ids=(1, 2, 3),
        )
    }
    fake_runner.execute_model_state = "stale-state"

    closed = verifier_runner_module.close_verifier_replay_session(
        fake_runner, request
    )

    assert closed is True
    assert "vs-close" not in fake_runner._dssd_verifier_replay_sessions
    assert "dssd-verify:vs-close" not in fake_runner.requests
    assert "dssd-verify:vs-close" not in fake_runner.num_prompt_logprobs
    assert fake_runner.execute_model_state is None
