# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm import envs
from vllm.config import set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory
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


_REMOTE_SMOKE_MODEL = "hmellor/tiny-random-LlamaForCausalLM"
_LOCAL_SMOKE_MODEL_CANDIDATES = (
    _REMOTE_SMOKE_MODEL,
    "facebook/opt-125m",
)


def _snapshot_has_required_files(snapshot_dir: Path) -> bool:
    if not snapshot_dir.is_dir() or not (snapshot_dir / "config.json").is_file():
        return False

    has_weights = any(snapshot_dir.glob("*.safetensors")) or any(
        snapshot_dir.glob("*.bin")
    )
    has_tokenizer = (
        (snapshot_dir / "tokenizer.json").is_file()
        or (snapshot_dir / "tokenizer.model").is_file()
        or (
            (snapshot_dir / "vocab.json").is_file()
            and (snapshot_dir / "merges.txt").is_file()
        )
    )
    return has_weights and has_tokenizer


def _find_local_snapshot(model_id: str) -> str | None:
    repo_cache = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{model_id.replace('/', '--')}"
    )
    snapshots_dir = repo_cache / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    ref_path = repo_cache / "refs" / "main"
    if ref_path.is_file():
        preferred_snapshot = snapshots_dir / ref_path.read_text().strip()
        if _snapshot_has_required_files(preferred_snapshot):
            return str(preferred_snapshot)

    for snapshot_dir in sorted(snapshots_dir.iterdir()):
        if _snapshot_has_required_files(snapshot_dir):
            return str(snapshot_dir)

    return None


def _resolve_smoke_model() -> str:
    for model_id in _LOCAL_SMOKE_MODEL_CANDIDATES:
        local_snapshot = _find_local_snapshot(model_id)
        if local_snapshot is not None:
            return local_snapshot

    return _REMOTE_SMOKE_MODEL


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
def dummy_model_runner_v1():
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    runner = object.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.execute_model_state = None
    runner.requests = {}
    runner.input_batch = SimpleNamespace(
        req_id_to_index={},
        token_ids_cpu=torch.zeros((4, 16), dtype=torch.int32).numpy(),
        num_tokens_no_spec=torch.zeros(4, dtype=torch.int32).numpy(),
        num_computed_tokens_cpu=torch.zeros(4, dtype=torch.int32).numpy(),
        sampling_metadata=SimpleNamespace(generators={}),
    )
    return runner


def _build_real_worker(*, use_v2_model_runner: bool):
    if not torch.cuda.is_available() or not current_platform.is_cuda():
        pytest.skip("requires a CUDA-resolved vLLM runtime")

    old_use_v2_model_runner = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if use_v2_model_runner else "0"
    envs.disable_envs_cache()
    try:
        engine_args = EngineArgs(
            model=_resolve_smoke_model(),
            enforce_eager=True,
            gpu_memory_utilization=0.4,
            max_model_len=64,
        )
        vllm_config = engine_args.create_engine_config()

        with tempfile.NamedTemporaryFile() as tmp_file, set_current_vllm_config(
            vllm_config
        ):
            worker = None
            worker = Worker(
                vllm_config=vllm_config,
                local_rank=0,
                rank=0,
                distributed_init_method=f"file://{tmp_file.name}",
                is_driver_worker=True,
            )
            try:
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
            finally:
                if worker is not None:
                    worker.shutdown()
                cleanup_dist_env_and_memory()
    finally:
        if old_use_v2_model_runner is None:
            os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
        else:
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = old_use_v2_model_runner
        envs.disable_envs_cache()


@pytest.fixture
def real_worker():
    yield from _build_real_worker(use_v2_model_runner=True)


@pytest.fixture
def real_worker_v1():
    yield from _build_real_worker(use_v2_model_runner=False)
