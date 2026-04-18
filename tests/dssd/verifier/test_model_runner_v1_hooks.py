from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu_model_runner import ExecuteModelState, GPUModelRunner


def test_take_execute_model_state_returns_then_clears_old_runner_state() -> None:
    runner = object.__new__(GPUModelRunner)
    execute_model_state = ExecuteModelState(
        scheduler_output=SimpleNamespace(),
        logits=torch.zeros((1, 3), dtype=torch.float32),
        spec_decode_metadata=None,
        spec_decode_common_attn_metadata=None,
        hidden_states=torch.zeros((1, 4), dtype=torch.float32),
        sample_hidden_states=torch.zeros((1, 4), dtype=torch.float32),
        aux_hidden_states=None,
        ec_connector_output=None,
        cudagraph_stats=None,
        slot_mappings=None,
    )
    runner.execute_model_state = execute_model_state

    state = runner.take_execute_model_state()

    assert state is execute_model_state
    assert state.logits.shape == (1, 3)
    assert runner.execute_model_state is None


def test_take_execute_model_state_raises_when_empty() -> None:
    runner = object.__new__(GPUModelRunner)
    runner.execute_model_state = None

    with pytest.raises(RuntimeError, match="execute_model_state is empty"):
        runner.take_execute_model_state()
