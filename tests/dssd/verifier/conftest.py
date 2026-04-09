# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import tempfile
from types import SimpleNamespace

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
)
from vllm.v1.worker.gpu_worker import Worker
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


@pytest.fixture
def real_worker():
    if not torch.cuda.is_available() or not current_platform.is_cuda():
        pytest.skip("requires a CUDA-resolved vLLM runtime")

    engine_args = EngineArgs(
        model="hmellor/tiny-random-LlamaForCausalLM",
        enforce_eager=True,
        gpu_memory_utilization=0.4,
        max_model_len=64,
    )
    vllm_config = engine_args.create_engine_config()

    with tempfile.NamedTemporaryFile() as tmp_file, set_current_vllm_config(
        vllm_config
    ):
        worker = Worker(
            vllm_config=vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method=f"file://{tmp_file.name}",
            is_driver_worker=True,
        )
        worker.init_device()
        worker.load_model()

        available_memory = [worker.determine_available_memory()]
        kv_cache_configs = get_kv_cache_configs(
            vllm_config,
            [worker.get_kv_cache_spec()],
            available_memory,
        )
        scheduler_kv_cache_config = generate_scheduler_kv_cache_config(
            kv_cache_configs
        )
        worker.initialize_from_config(kv_cache_configs[0])

        yield (
            worker,
            vllm_config,
            KVCacheManager(
                kv_cache_config=scheduler_kv_cache_config,
                max_model_len=vllm_config.model_config.max_model_len,
                hash_block_size=vllm_config.cache_config.block_size,
            ),
        )
