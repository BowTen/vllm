# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


@pytest.fixture
def dummy_model_runner() -> GPUModelRunner:
    runner = object.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.execute_model_state = None
    runner.is_last_pp_rank = True
    runner.sampler = None
    runner.input_buffers = InputBuffers(
        max_num_reqs=4,
        max_num_tokens=16,
        device=runner.device,
    )
    runner.req_states = SimpleNamespace(
        req_id_to_index={},
        last_sampled_tokens=torch.zeros((4, 1), dtype=torch.int64),
        total_len=SimpleNamespace(gpu=torch.zeros(4, dtype=torch.int32)),
        all_token_ids=SimpleNamespace(gpu=torch.zeros((4, 16), dtype=torch.int32)),
    )
    return runner
