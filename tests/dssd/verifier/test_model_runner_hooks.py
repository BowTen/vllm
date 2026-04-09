# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_runner import ExecuteModelState
from vllm.v1.worker.gpu.sample.output import SamplerOutput


def _seed_request_state(dummy_model_runner) -> int:
    runner = dummy_model_runner
    req_idx = 0
    runner.req_states.req_id_to_index["req-0"] = req_idx
    runner.req_states.all_token_ids.gpu[req_idx, :3] = torch.tensor(
        [1, 2, 3], dtype=torch.int32
    )
    runner.req_states.total_len.gpu[req_idx] = 3
    return req_idx


def test_sample_without_postprocess_does_not_advance_total_len(
    dummy_model_runner,
) -> None:
    runner = dummy_model_runner
    req_idx = _seed_request_state(runner)
    original_total_len = int(runner.req_states.total_len.gpu[req_idx].item())
    grad_enabled = None

    fake_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[7]], device=runner.device),
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=torch.tensor([1], device=runner.device),
    )
    hidden_states = torch.zeros((1, 8), device=runner.device)
    input_batch = mock.MagicMock()

    def _fake_sample(*args, **kwargs):
        nonlocal grad_enabled
        grad_enabled = torch.is_grad_enabled()
        return (
            fake_output,
            torch.tensor([1], device=runner.device),
            torch.tensor([0], device=runner.device),
        )

    with mock.patch.object(
        runner,
        "sample",
        side_effect=_fake_sample,
    ) as sample_mock:
        result = runner.sample_without_postprocess(
            hidden_states,
            input_batch,
            grammar_output=None,
        )

    assert result is fake_output
    sample_mock.assert_called_once_with(hidden_states, input_batch, None)
    assert grad_enabled is False
    assert int(runner.req_states.total_len.gpu[req_idx].item()) == original_total_len


def test_commit_input_token_updates_req_state(dummy_model_runner) -> None:
    runner = dummy_model_runner
    req_idx = _seed_request_state(runner)
    output_bin_counts = torch.zeros((4, 16), dtype=torch.int32)
    runner.sampler = SimpleNamespace(
        penalties_state=SimpleNamespace(output_bin_counts=output_bin_counts)
    )

    runner.commit_input_token(req_idx, 9)

    assert int(runner.req_states.last_sampled_tokens[req_idx, 0].item()) == 9
    assert int(runner.req_states.all_token_ids.gpu[req_idx, 3].item()) == 9
    assert int(runner.req_states.total_len.gpu[req_idx].item()) == 4
    assert int(output_bin_counts[req_idx, 9].item()) == 1


def test_take_execute_model_state_returns_then_clears_state(
    dummy_model_runner,
) -> None:
    runner = dummy_model_runner
    input_batch = InputBatch.make_dummy(
        num_reqs=1,
        num_tokens=1,
        input_buffers=runner.input_buffers,
    )
    execute_model_state = ExecuteModelState(
        input_batch=input_batch,
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=torch.zeros((1, 8), device=runner.device),
        aux_hidden_states=None,
        kv_connector_output=None,
        num_tokens_across_dp=None,
    )
    runner.execute_model_state = execute_model_state

    result = runner.take_execute_model_state()

    assert result is execute_model_state
    assert runner.execute_model_state is None

    with pytest.raises(RuntimeError, match="execute_model_state is empty"):
        runner.take_execute_model_state()
