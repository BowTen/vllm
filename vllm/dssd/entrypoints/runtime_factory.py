from __future__ import annotations

import argparse
import contextlib
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

from vllm import envs
from vllm.config import set_current_vllm_config
from vllm.dssd.edge import (
    DSSDEdgeDraftSampler,
    DSSDEdgeDraftSamplerV1,
    EdgeDecodeEngine,
    EdgeDecodeEngineV1,
    EdgeSchedulerAdapter,
    EdgeStateBridge,
    EdgeStateBridgeV1,
)
from vllm.dssd.service import DSSDEdgeService, DSSDVerifierService
from vllm.dssd.transport import HTTPVerifierTransport
from vllm.dssd.verifier import (
    DSSDVerifierSampler,
    VerifierDecodeEngine,
    VerifierSchedulerAdapter,
    VerifierStateBridge,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
)
from vllm.v1.worker.gpu_worker import Worker

_DEFAULT_MAX_MODEL_LEN = 64
_DEFAULT_GPU_MEMORY_UTILIZATION = 0.01
_DEFAULT_KV_CACHE_MEMORY_BYTES = 128 * 1024 * 1024
_DEFAULT_MAX_NUM_BATCHED_TOKENS = 64
_DEFAULT_MAX_NUM_SEQS = 2
_REMOTE_SMOKE_MODEL = "hmellor/tiny-random-LlamaForCausalLM"
_LOCAL_SMOKE_MODEL_CANDIDATES = (
    _REMOTE_SMOKE_MODEL,
    "facebook/opt-125m",
)


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-model-len", type=int, default=_DEFAULT_MAX_MODEL_LEN)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=_DEFAULT_GPU_MEMORY_UTILIZATION,
    )
    parser.add_argument(
        "--kv-cache-memory-bytes",
        type=int,
        default=_DEFAULT_KV_CACHE_MEMORY_BYTES,
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=_DEFAULT_MAX_NUM_BATCHED_TOKENS,
    )
    parser.add_argument("--max-num-seqs", type=int, default=_DEFAULT_MAX_NUM_SEQS)
    parser.add_argument(
        "--enforce-eager",
        dest="enforce_eager",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-enforce-eager",
        dest="enforce_eager",
        action="store_false",
    )
    parser.add_argument(
        "--async-scheduling",
        action="store_true",
        default=False,
    )


def build_real_verifier_service(args):
    runtime, cleanup = _init_real_runtime(args, use_v2_model_runner=True)
    try:
        sampler = runtime.worker.model_runner.sampler
        if sampler is None:
            raise RuntimeError("real verifier runtime requires a sampler")
        block_hasher = _make_request_block_hasher(runtime.vllm_config)

        verifier_engine = VerifierDecodeEngine(
            vllm_config=runtime.vllm_config,
            worker=runtime.worker,
            scheduler=VerifierSchedulerAdapter(
                kv_cache_manager=runtime.kv_cache_manager,
                request_block_hasher=block_hasher,
            ),
            state_bridge=VerifierStateBridge(),
            verifier_sampler=DSSDVerifierSampler(
                sampler=sampler,
                num_speculative_steps=runtime.worker.model_runner.num_speculative_steps,
            ),
        )
        return DSSDVerifierService(decode_engine=verifier_engine), cleanup
    except Exception:
        with contextlib.suppress(Exception):
            cleanup()
        raise


def build_real_edge_service(args):
    model_runner_version = getattr(args, "model_runner_version", "v2")
    if model_runner_version not in {"v1", "v2"}:
        raise ValueError(
            "model_runner_version must be one of {'v1', 'v2'}"
        )
    async_scheduling = getattr(args, "async_scheduling", False)
    if model_runner_version == "v1" and async_scheduling:
        raise ValueError(
            "model_runner_version='v1' requires async_scheduling=False"
        )

    runtime, cleanup = _init_real_runtime(
        args,
        use_v2_model_runner=(model_runner_version != "v1"),
    )
    try:
        sampler = runtime.worker.model_runner.sampler
        if sampler is None:
            raise RuntimeError("real edge runtime requires a sampler")
        block_hasher = _make_request_block_hasher(runtime.vllm_config)

        if model_runner_version == "v1":
            edge_engine_cls = EdgeDecodeEngineV1
            state_bridge = EdgeStateBridgeV1()
            draft_sampler = DSSDEdgeDraftSamplerV1(sampler)
        else:
            edge_engine_cls = EdgeDecodeEngine
            state_bridge = EdgeStateBridge()
            draft_sampler = DSSDEdgeDraftSampler(sampler)

        edge_engine = edge_engine_cls(
            vllm_config=runtime.vllm_config,
            worker=runtime.worker,
            scheduler=EdgeSchedulerAdapter(
                kv_cache_manager=runtime.kv_cache_manager,
                request_block_hasher=block_hasher,
            ),
            state_bridge=state_bridge,
            draft_sampler=draft_sampler,
        )
        return (
            DSSDEdgeService(
                decode_engine=edge_engine,
                verifier=HTTPVerifierTransport(server_url=args.verifier_url),
                eos_token_id=args.eos_token_id,
                gamma=args.gamma,
            ),
            cleanup,
        )
    except Exception:
        with contextlib.suppress(Exception):
            cleanup()
        raise


def _init_real_runtime(
    args,
    *,
    use_v2_model_runner: bool = True,
) -> tuple[SimpleNamespace, Callable[[], None]]:
    old_use_v2_model_runner = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1" if use_v2_model_runner else "0"
    envs.disable_envs_cache()

    worker = None
    try:
        engine_args = EngineArgs(
            model=args.model or _resolve_default_model(),
            enforce_eager=args.enforce_eager,
            async_scheduling=args.async_scheduling,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            kv_cache_memory_bytes=args.kv_cache_memory_bytes,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
            enable_prefix_caching=False,
            disable_log_stats=True,
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
            worker.compile_or_warm_up_model()

            runtime = SimpleNamespace(
                worker=worker,
                vllm_config=vllm_config,
                kv_cache_manager=KVCacheManager(
                    kv_cache_config=scheduler_kv_cache_config,
                    max_model_len=vllm_config.model_config.max_model_len,
                    hash_block_size=vllm_config.cache_config.block_size,
                ),
            )
    except Exception:
        if worker is not None:
            with contextlib.suppress(Exception):
                worker.shutdown()
        _cleanup_dist_runtime(old_use_v2_model_runner)
        raise

    cleanup_ran = False

    def cleanup() -> None:
        nonlocal cleanup_ran
        if cleanup_ran:
            return
        cleanup_ran = True
        if worker is not None:
            with contextlib.suppress(Exception):
                worker.shutdown()
        _cleanup_dist_runtime(old_use_v2_model_runner)

    return runtime, cleanup


def _cleanup_dist_runtime(old_use_v2_model_runner: str | None) -> None:
    from vllm.distributed import cleanup_dist_env_and_memory

    cleanup_dist_env_and_memory()
    if old_use_v2_model_runner is None:
        os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
    else:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = old_use_v2_model_runner
    envs.disable_envs_cache()


def _make_standalone_cleanup(worker) -> Callable[[], None]:
    cleanup_ran = False

    def cleanup() -> None:
        nonlocal cleanup_ran
        if cleanup_ran:
            return
        cleanup_ran = True
        with contextlib.suppress(Exception):
            worker.shutdown()
        envs.disable_envs_cache()

    return cleanup


def _make_request_block_hasher(vllm_config):
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core.kv_cache_utils import (
        get_request_block_hasher,
        init_none_hash,
    )

    hash_fn = get_hash_fn_by_name(vllm_config.cache_config.prefix_caching_hash_algo)
    init_none_hash(hash_fn)
    return get_request_block_hasher(
        vllm_config.cache_config.block_size,
        hash_fn,
    )


def _resolve_default_model() -> str:
    for model_id in _LOCAL_SMOKE_MODEL_CANDIDATES:
        local_snapshot = _find_local_snapshot(model_id)
        if local_snapshot is not None:
            return local_snapshot
    return _REMOTE_SMOKE_MODEL


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
