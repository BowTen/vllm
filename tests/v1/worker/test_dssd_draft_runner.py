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
VerifierSessionInitRequest = protocol_module.VerifierSessionInitRequest
VerifierCommitRequest = protocol_module.VerifierCommitRequest
CloseSessionRequest = protocol_module.CloseSessionRequest
VerifyRoundRequest = protocol_module.VerifyRoundRequest
VerifierForwardResult = protocol_module.VerifierForwardResult
SamplingParams = draft_runner_module.SamplingParams
build_draft_result_from_distribution = (
    draft_runner_module.build_draft_result_from_distribution
)
build_draft_replay_scheduler_output = (
    draft_runner_module.build_draft_replay_scheduler_output
)
run_draft_replay_forward = draft_runner_module.run_draft_replay_forward


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


def test_session_runner_draft_round_selects_non_none_collective_reply():
    expected = DraftRoundResult(
        draft_token_ids=[5, 6],
        q_values=[0.5, 0.4],
        q_dists_handle="rpc-handle",
        q_distributions=[[0.5, 0.5], [0.6, 0.4]],
    )
    model_executor = types.SimpleNamespace(
        collective_rpc=Mock(return_value=[None, expected, None]),
    )
    runner = DSSDSessionRunner(model_executor=model_executor)
    request = DraftRoundRequest(
        local_session_id="edge-pp",
        prompt_token_ids=[1, 2],
        committed_token_ids=[3],
        seq_no=4,
        gamma=2,
    )

    result = runner.dssd_draft_round(request)

    assert result == expected


def test_build_draft_result_from_distribution_uses_sampled_token_probability():
    q_distribution = torch.tensor([0.1, 0.3, 0.6], dtype=torch.float32)

    result = build_draft_result_from_distribution(
        q_distribution=q_distribution,
        sampled_token_id=2,
        q_dists_handle="edge-1:0",
    )

    assert result.draft_token_ids == [2]
    assert result.q_dists_handle == "edge-1:0"
    assert result.q_distributions is not None
    assert len(result.q_distributions) == 1
    assert pytest.approx(result.q_values[0]) == result.q_distributions[0][2]
    assert sum(result.q_distributions[0]) == pytest.approx(1.0)


def test_run_draft_replay_forward_uses_replay_sampler_public_interface(
    monkeypatch,
):
    request = DraftRoundRequest(
        local_session_id="edge-processed",
        prompt_token_ids=[11, 12],
        committed_token_ids=[13],
        seq_no=6,
        gamma=1,
        sampling_params=SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=17,
            min_p=0.05,
            seed=321,
            max_tokens=64,
        ),
    )
    raw_logits = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)
    processed_distribution = torch.tensor([[0.0, 0.25, 0.75]], dtype=torch.float32)
    sampler_inits = []
    sampler_calls = []
    sampling_metadata = types.SimpleNamespace(
        temperature=torch.tensor([0.7], dtype=torch.float32),
        all_greedy=False,
        all_random=False,
        top_p=torch.tensor([1.0], dtype=torch.float32),
        top_k=torch.tensor([3], dtype=torch.int32),
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.tensor([0.0], dtype=torch.float32),
        presence_penalties=torch.tensor([0.0], dtype=torch.float32),
        repetition_penalties=torch.tensor([1.0], dtype=torch.float32),
        output_token_ids=[[]],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=types.SimpleNamespace(argmax_invariant=[]),
    )

    def _execute_model(_scheduler_output):
        fake_runner.execute_model_state = types.SimpleNamespace(logits=raw_logits)
        return None

    class _FakeReplaySampler:
        def __init__(self, logprobs_mode="raw_logprobs") -> None:
            sampler_inits.append(logprobs_mode)

        def __call__(
            self,
            *,
            logits,
            sampling_metadata,
            logprobs_mode_override=None,
        ):
            sampler_calls.append(
                types.SimpleNamespace(
                    logits=logits.clone(),
                    sampling_metadata=sampling_metadata,
                    logprobs_mode_override=logprobs_mode_override,
                )
            )
            return types.SimpleNamespace(
                sampled_token_ids=torch.tensor([[2]], dtype=torch.int32),
                logprobs_tensors=types.SimpleNamespace(
                    logprobs=processed_distribution.log()
                ),
            )

    monkeypatch.setattr(
        draft_runner_module,
        "Sampler",
        _FakeReplaySampler,
        raising=False,
    )

    fake_runner = types.SimpleNamespace(
        use_async_scheduling=False,
        is_pooling_model=False,
        supports_mm_inputs=False,
        kv_cache_config=types.SimpleNamespace(
            kv_cache_groups=[
                types.SimpleNamespace(
                    kv_cache_spec=types.SimpleNamespace(block_size=4)
                )
            ]
        ),
        execute_model=_execute_model,
        execute_model_state=None,
        input_batch=types.SimpleNamespace(
            prev_sampled_token_ids=None,
            remove_request=Mock(return_value=0),
            condense=Mock(),
            sampling_metadata=sampling_metadata,
        ),
        sample_tokens=Mock(
            side_effect=AssertionError("draft replay must not call sample_tokens")
        ),
        requests={},
        num_prompt_logprobs={},
        late_interaction_runner=types.SimpleNamespace(
            on_requests_finished=Mock(),
        ),
        kv_connector_output=None,
        _draft_token_ids=None,
        _draft_token_req_ids=None,
    )

    result = run_draft_replay_forward(fake_runner, request)

    assert result is not None
    assert result.draft_token_ids == [2]
    assert sampler_inits == ["processed_logprobs"]
    assert len(sampler_calls) == 1
    sampler_call = sampler_calls[0]
    assert sampler_call.logprobs_mode_override is None
    assert sampler_call.logits.equal(raw_logits)
    assert sampler_call.sampling_metadata.max_num_logprobs == -1
    assert sampler_call.sampling_metadata.prompt_token_ids.tolist() == [[11, 12]]
    assert sampler_call.sampling_metadata.output_token_ids == [[13]]
    assert result.q_distributions == [pytest.approx(processed_distribution[0].tolist())]
    assert result.q_values == [pytest.approx(0.75)]


def test_run_draft_replay_forward_uses_deterministic_q_distribution_for_greedy(
    monkeypatch,
):
    request = DraftRoundRequest(
        local_session_id="edge-greedy",
        prompt_token_ids=[11, 12],
        committed_token_ids=[13],
        seq_no=7,
        gamma=1,
        sampling_params=SamplingParams(
            temperature=0.0,
            max_tokens=32,
        ),
    )
    raw_logits = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)
    sampler_inits = []
    sampling_metadata = types.SimpleNamespace(
        temperature=None,
        all_greedy=True,
        all_random=False,
        top_p=torch.tensor([1.0], dtype=torch.float32),
        top_k=torch.tensor([3], dtype=torch.int32),
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.tensor([0.0], dtype=torch.float32),
        presence_penalties=torch.tensor([0.0], dtype=torch.float32),
        repetition_penalties=torch.tensor([1.0], dtype=torch.float32),
        output_token_ids=[[]],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=types.SimpleNamespace(argmax_invariant=[]),
    )

    def _execute_model(_scheduler_output):
        fake_runner.execute_model_state = types.SimpleNamespace(logits=raw_logits)
        return None

    sampler_calls = []

    class _FakeReplaySampler:
        def __init__(self, logprobs_mode="raw_logprobs") -> None:
            sampler_inits.append(logprobs_mode)

        def __call__(
            self,
            *,
            logits,
            sampling_metadata,
            logprobs_mode_override=None,
        ):
            sampler_calls.append(
                types.SimpleNamespace(
                    logits=logits.clone(),
                    sampling_metadata=sampling_metadata,
                    logprobs_mode_override=logprobs_mode_override,
                )
            )
            return types.SimpleNamespace(
                sampled_token_ids=torch.tensor([[2]], dtype=torch.int32),
                logprobs_tensors=types.SimpleNamespace(
                    logprobs=torch.tensor(
                        [[float("-inf"), float("-inf"), 0.0]],
                        dtype=torch.float32,
                    )
                ),
            )

    monkeypatch.setattr(
        draft_runner_module,
        "Sampler",
        _FakeReplaySampler,
        raising=False,
    )

    fake_runner = types.SimpleNamespace(
        use_async_scheduling=False,
        is_pooling_model=False,
        supports_mm_inputs=False,
        kv_cache_config=types.SimpleNamespace(
            kv_cache_groups=[
                types.SimpleNamespace(
                    kv_cache_spec=types.SimpleNamespace(block_size=4)
                )
            ]
        ),
        execute_model=_execute_model,
        execute_model_state=None,
        input_batch=types.SimpleNamespace(
            prev_sampled_token_ids=None,
            remove_request=Mock(return_value=0),
            condense=Mock(),
            sampling_metadata=sampling_metadata,
        ),
        sample_tokens=Mock(
            side_effect=AssertionError("draft replay must not call sample_tokens")
        ),
        requests={},
        num_prompt_logprobs={},
        late_interaction_runner=types.SimpleNamespace(
            on_requests_finished=Mock(),
        ),
        kv_connector_output=None,
        _draft_token_ids=None,
        _draft_token_req_ids=None,
    )

    result = run_draft_replay_forward(fake_runner, request)

    assert result is not None
    assert result.draft_token_ids == [2]
    assert sampler_inits == ["processed_logprobs"]
    assert len(sampler_calls) == 1
    sampler_call = sampler_calls[0]
    assert sampler_call.logprobs_mode_override is None
    assert sampler_call.sampling_metadata.max_num_logprobs == -1
    assert sampler_call.sampling_metadata.prompt_token_ids.tolist() == [[11, 12]]
    assert sampler_call.sampling_metadata.output_token_ids == [[13]]
    assert result.q_distributions == [pytest.approx([0.0, 0.0, 1.0])]
    assert result.q_values == [pytest.approx(1.0)]


def test_build_draft_replay_scheduler_output_uses_prefix_tokens():
    request = DraftRoundRequest(
        local_session_id="edge-1",
        prompt_token_ids=[1, 2, 3],
        committed_token_ids=[4],
        seq_no=5,
        gamma=3,
    )

    view, scheduler_output = build_draft_replay_scheduler_output(
        request,
        draft_token_ids=[9, 8],
        step_index=2,
        block_sizes=(4,),
    )

    assert view.request_id == "dssd-draft:edge-1:5:2"
    assert view.prompt_token_ids == [1, 2, 3, 4, 9, 8]
    assert view.num_scheduled_tokens == 6
    assert view.block_ids == ([0, 1],)
    assert scheduler_output.total_num_scheduled_tokens == 6
    assert scheduler_output.num_scheduled_tokens == {view.request_id: 6}
    assert scheduler_output.scheduled_spec_decode_tokens == {}
    assert scheduler_output.scheduled_new_reqs[0].prompt_token_ids == [
        1,
        2,
        3,
        4,
        9,
        8,
    ]
    assert scheduler_output.scheduled_new_reqs[0].sampling_params.temperature == 0.0
    assert scheduler_output.scheduled_new_reqs[0].sampling_params.max_tokens == 1


def test_build_draft_replay_scheduler_output_uses_request_sampling_params():
    request = DraftRoundRequest(
        local_session_id="edge-1",
        prompt_token_ids=[1, 2, 3],
        committed_token_ids=[4],
        seq_no=5,
        gamma=3,
        sampling_params=SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=17,
            min_p=0.05,
            seed=321,
            max_tokens=64,
        ),
    )

    _, scheduler_output = build_draft_replay_scheduler_output(
        request,
        draft_token_ids=[9],
        step_index=1,
        block_sizes=(4,),
    )

    replay_sampling_params = scheduler_output.scheduled_new_reqs[0].sampling_params
    assert replay_sampling_params is not request.sampling_params
    assert replay_sampling_params.temperature == 0.7
    assert replay_sampling_params.top_p == 0.8
    assert replay_sampling_params.top_k == 17
    assert replay_sampling_params.min_p == 0.05
    assert replay_sampling_params.seed == 321
    assert replay_sampling_params.max_tokens == 1
    assert request.sampling_params.max_tokens == 64


def test_build_draft_replay_scheduler_output_recomputes_min_tokens_per_step():
    consumed_request = DraftRoundRequest(
        local_session_id="edge-1",
        prompt_token_ids=[1, 2, 3],
        committed_token_ids=[4, 5],
        seq_no=5,
        gamma=3,
        sampling_params=SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=17,
            min_p=0.05,
            seed=321,
            min_tokens=2,
            max_tokens=64,
        ),
    )
    unmet_request = DraftRoundRequest(
        local_session_id="edge-2",
        prompt_token_ids=[1, 2, 3],
        committed_token_ids=[4],
        seq_no=6,
        gamma=3,
        sampling_params=SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=17,
            min_p=0.05,
            seed=321,
            min_tokens=4,
            max_tokens=64,
        ),
    )

    _, consumed_scheduler_output = build_draft_replay_scheduler_output(
        consumed_request,
        draft_token_ids=[],
        step_index=0,
        block_sizes=(4,),
    )
    _, unmet_scheduler_output = build_draft_replay_scheduler_output(
        unmet_request,
        draft_token_ids=[9, 8],
        step_index=2,
        block_sizes=(4,),
    )

    consumed_sampling_params = (
        consumed_scheduler_output.scheduled_new_reqs[0].sampling_params
    )
    unmet_sampling_params = (
        unmet_scheduler_output.scheduled_new_reqs[0].sampling_params
    )

    assert consumed_sampling_params.min_tokens == 0
    assert unmet_sampling_params.min_tokens == 1
    assert consumed_request.sampling_params.min_tokens == 2
    assert unmet_request.sampling_params.min_tokens == 4


def test_run_draft_replay_forward_rolls_out_gamma_steps_with_real_output_history_and_cleans_up(
    monkeypatch,
):
    request = DraftRoundRequest(
        local_session_id="edge-2",
        prompt_token_ids=[11, 12],
        committed_token_ids=[13],
        seq_no=6,
        gamma=2,
    )
    logits = [
        torch.tensor([[0.0, 1.0, 3.0]], dtype=torch.float32),
        torch.tensor([[4.0, 0.0, 0.0]], dtype=torch.float32),
    ]
    sampling_metadata = types.SimpleNamespace(
        temperature=torch.tensor([0.7], dtype=torch.float32),
        all_greedy=False,
        all_random=False,
        top_p=torch.tensor([1.0], dtype=torch.float32),
        top_k=torch.tensor([3], dtype=torch.int32),
        generators={},
        max_num_logprobs=None,
        no_penalties=False,
        prompt_token_ids=None,
        frequency_penalties=torch.tensor([0.4], dtype=torch.float32),
        presence_penalties=torch.tensor([0.3], dtype=torch.float32),
        repetition_penalties=torch.tensor([1.0], dtype=torch.float32),
        output_token_ids=[[]],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=types.SimpleNamespace(argmax_invariant=[]),
    )
    sampler_outputs = [
        types.SimpleNamespace(
            sampled_token_ids=torch.tensor([[2]], dtype=torch.int32),
            logprobs_tensors=types.SimpleNamespace(
                logprobs=torch.tensor([[float("-inf"), float("-inf"), 0.0]])
            ),
        ),
        types.SimpleNamespace(
            sampled_token_ids=torch.tensor([[0]], dtype=torch.int32),
            logprobs_tensors=types.SimpleNamespace(
                logprobs=torch.tensor([[0.0, float("-inf"), float("-inf")]])
            ),
        ),
    ]
    execute_calls = []
    sampler_inits = []
    sampler_calls = []

    def _execute_model(scheduler_output):
        execute_calls.append(
            scheduler_output.scheduled_new_reqs[0].prompt_token_ids
        )
        fake_runner.execute_model_state = types.SimpleNamespace(
            logits=logits[len(execute_calls) - 1]
        )
        return None

    class _FakeReplaySampler:
        def __init__(self, logprobs_mode="raw_logprobs") -> None:
            sampler_inits.append(logprobs_mode)

        def __call__(
            self,
            *,
            logits,
            sampling_metadata,
            logprobs_mode_override=None,
        ):
            sampler_calls.append(
                types.SimpleNamespace(
                    logits=logits.clone(),
                    sampling_metadata=sampling_metadata,
                    logprobs_mode_override=logprobs_mode_override,
                )
            )
            return sampler_outputs.pop(0)

    monkeypatch.setattr(
        draft_runner_module,
        "Sampler",
        _FakeReplaySampler,
        raising=False,
    )

    fake_runner = types.SimpleNamespace(
        use_async_scheduling=False,
        is_pooling_model=False,
        supports_mm_inputs=False,
        kv_cache_config=types.SimpleNamespace(
            kv_cache_groups=[
                types.SimpleNamespace(
                    kv_cache_spec=types.SimpleNamespace(block_size=4)
                )
            ]
        ),
        execute_model=_execute_model,
        execute_model_state=None,
        input_batch=types.SimpleNamespace(
            prev_sampled_token_ids="stale",
            sampling_metadata=sampling_metadata,
            remove_request=Mock(return_value=0),
            condense=Mock(),
        ),
        sample_tokens=Mock(
            side_effect=AssertionError("draft replay must not call sample_tokens")
        ),
        requests={},
        num_prompt_logprobs={},
        late_interaction_runner=types.SimpleNamespace(
            on_requests_finished=Mock(),
        ),
        kv_connector_output="stale-kv",
        _draft_token_ids=[99],
        _draft_token_req_ids=["old-req"],
    )

    result = run_draft_replay_forward(fake_runner, request)

    assert execute_calls == [
        [11, 12, 13],
        [11, 12, 13, 2],
    ]
    assert sampler_inits == ["processed_logprobs"]
    assert [call.sampling_metadata.prompt_token_ids.tolist()
            for call in sampler_calls] == [[[11, 12]], [[11, 12]]]
    assert [call.sampling_metadata.output_token_ids
            for call in sampler_calls] == [[[13]], [[13, 2]]]
    assert result.draft_token_ids == [2, 0]
    assert len(result.q_values) == 2
    assert result.q_distributions is not None
    assert len(result.q_distributions) == 2
    assert fake_runner.input_batch.remove_request.call_args_list == [
        (( "dssd-draft:edge-2:6:0",),),
        (( "dssd-draft:edge-2:6:1",),),
    ]
    assert fake_runner.input_batch.condense.call_count == 2
    assert fake_runner.late_interaction_runner.on_requests_finished.call_args_list == [
        (({"dssd-draft:edge-2:6:0"},),),
        (({"dssd-draft:edge-2:6:1"},),),
    ]
    assert fake_runner.input_batch.prev_sampled_token_ids is None
    assert fake_runner.kv_connector_output is None
    assert fake_runner._draft_token_ids is None
    assert fake_runner._draft_token_req_ids is None


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

    assert result.verifier_session_id == "vs-1"
    assert result.seq_no == 2
    assert len(result.seq_probs) == 2
    assert len(result.bonus_probs) == 9


def test_session_runner_tracks_verifier_sessions():
    runner = DSSDSessionRunner()

    created = runner.create_verifier_session(
        VerifierSessionInitRequest(
            verifier_session_id="vs-1",
            binding_id="bind-1",
            prompt_token_ids=[1, 2, 3],
            sampling_params_digest="sp-1",
        )
    )

    assert created is True
    state = runner.verifier_sessions.get("vs-1")
    assert state is not None
    assert state.verifier_session_id == "vs-1"
    assert state.committed_token_ids == [1, 2, 3]

    closed = runner.close_verifier_session(
        CloseSessionRequest(
            verifier_session_id="vs-1",
            reason="done",
        )
    )

    assert closed is True
    assert runner.verifier_sessions.get("vs-1") is None


def test_session_runner_close_verifier_session_dispatches_to_workers():
    class _Executor:
        def __init__(self) -> None:
            self.calls = []

        def collective_rpc(self, method, args=(), **kwargs):
            self.calls.append((method, args, kwargs))
            return [True, True]

    executor = _Executor()
    runner = DSSDSessionRunner(model_executor=executor)
    runner.create_verifier_session(
        VerifierSessionInitRequest(
            verifier_session_id="vs-1",
            binding_id="bind-1",
            prompt_token_ids=[1, 2, 3],
            sampling_params_digest="sp-1",
        )
    )

    closed = runner.close_verifier_session(
        CloseSessionRequest(
            verifier_session_id="vs-1",
            reason="done",
        )
    )

    assert closed is True
    assert runner.verifier_sessions.get("vs-1") is None
    assert executor.calls == [
        (
            "dssd_close_verifier_session",
            (CloseSessionRequest(verifier_session_id="vs-1", reason="done"),),
            {},
        )
    ]


def test_session_runner_close_verifier_session_keeps_engine_state_on_worker_failure():
    class _Executor:
        def __init__(self) -> None:
            self.calls = []

        def collective_rpc(self, method, args=(), **kwargs):
            self.calls.append((method, args, kwargs))
            raise RuntimeError("worker cleanup failed")

    executor = _Executor()
    runner = DSSDSessionRunner(model_executor=executor)
    runner.create_verifier_session(
        VerifierSessionInitRequest(
            verifier_session_id="vs-1",
            binding_id="bind-1",
            prompt_token_ids=[1, 2, 3],
            sampling_params_digest="sp-1",
        )
    )
    request = CloseSessionRequest(
        verifier_session_id="vs-1",
        reason="done",
    )

    with pytest.raises(RuntimeError, match="worker cleanup failed"):
        runner.close_verifier_session(request)

    state = runner.verifier_sessions.get("vs-1")
    assert state is not None
    assert state.verifier_session_id == "vs-1"
    assert executor.calls == [
        ("dssd_close_verifier_session", (request,), {})
    ]


def test_session_runner_commits_verifier_tokens_after_round():
    runner = DSSDSessionRunner()
    runner.create_verifier_session(
        VerifierSessionInitRequest(
            verifier_session_id="vs-1",
            binding_id="bind-1",
            prompt_token_ids=[1, 2, 3],
            sampling_params_digest="sp-1",
        )
    )

    committed = runner.commit_verifier_tokens(
        VerifierCommitRequest(
            verifier_session_id="vs-1",
            token_ids=[7, 8],
        )
    )

    assert committed is True
    state = runner.verifier_sessions.get("vs-1")
    assert state is not None
    assert state.committed_token_ids == [1, 2, 3, 7, 8]


def test_session_runner_applies_prefix_delta_once_after_successful_verify():
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

    try:
        runner.dssd_verify_round(request)
    except RuntimeError as exc:
        assert "temporary failure" in str(exc)

    state = runner.verifier_sessions.get("vs-1")
    assert state is not None
    assert state.committed_token_ids == [1]

    runner.dssd_verify_round(request)
    assert state.committed_token_ids == [1, 9]
    assert [req.committed_token_ids for req in runner.model_executor.requests] == [
        [1, 9],
        [1, 9],
    ]
