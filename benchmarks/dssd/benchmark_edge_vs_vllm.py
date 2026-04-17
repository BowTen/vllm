#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from vllm import EngineArgs, LLM, SamplingParams, TokensPrompt
from vllm import envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.dssd.edge import (
    DSSDEdgeDraftSampler,
    EdgeDecodeEngine,
    EdgeSchedulerAdapter,
    EdgeStateBridge,
)
from vllm.distributed import cleanup_dist_env_and_memory
import vllm.platforms as platforms
from vllm.platforms.cuda import CudaPlatform
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
)
from vllm.v1.worker.gpu_worker import Worker
from vllm.usage.usage_lib import UsageContext


@dataclass
class BenchmarkResult:
    engine: str
    model: str
    prompt_tokens: int
    requested_output_tokens: int
    generated_tokens: int
    warmup_iters: int
    benchmark_iters: int
    total_seconds: float
    avg_seconds: float
    tokens_per_second: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark local decode speed for vLLM native vs DSSD edge",
    )
    parser.add_argument(
        "--engine",
        choices=("native", "edge", "both"),
        default="both",
        help="Which engine to benchmark.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model path or HF model name.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Explain speculative decoding in one paragraph and mention latency, "
            "acceptance rate, and KV cache."
        ),
        help="Prompt text used for benchmarking.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of generated tokens to benchmark.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=1,
        help="Number of warmup iterations before timing.",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=5,
        help="Number of timed iterations.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Model dtype passed to vLLM.",
    )
    parser.add_argument(
        "--native-model-runner",
        choices=("v1", "v2"),
        default="v1",
        help="Model runner used by the native engine benchmark.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for model/KV cache.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Max model length.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0.0 means greedy.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Sampling top-p.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Sampling top-k.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to tokenizer/model loading.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs for both engines.",
    )
    parser.add_argument(
        "--honor-eos",
        action="store_true",
        help="Stop on EOS instead of forcing max-new-tokens.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON only.",
    )
    return parser.parse_args()


def build_sampling_params(args: argparse.Namespace) -> SamplingParams:
    return SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        ignore_eos=not args.honor_eos,
        detokenize=False,
    )


def build_vllm_config(args: argparse.Namespace) -> VllmConfig:
    ensure_cuda_platform_if_needed()
    engine_args = EngineArgs(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,
    )
    vllm_config = engine_args.create_engine_config(UsageContext.LLM_CLASS)
    if args.enforce_eager:
        vllm_config.compilation_config.custom_ops = ["none"]
        vllm_config.compilation_config.pass_config.fuse_norm_quant = False
        vllm_config.compilation_config.pass_config.fuse_act_quant = False
    return vllm_config


def synchronize_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def ensure_cuda_platform_if_needed() -> None:
    platform = platforms.current_platform
    if torch.cuda.is_available() and platform.device_type == "cpu":
        platforms.current_platform = CudaPlatform()
        for module in tuple(sys.modules.values()):
            if module is None or module is platforms:
                continue
            if hasattr(module, "current_platform"):
                try:
                    setattr(module, "current_platform", platforms.current_platform)
                except Exception:
                    pass
    if torch.cuda.is_available():
        try:
            setattr(
                platforms.current_platform,
                "opaque_attention_op",
                lambda *args, **kwargs: False,
            )
        except Exception:
            pass


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in the current Python environment. "
            "Install a CUDA-enabled PyTorch build before running this benchmark."
        )


class EdgeRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.vllm_config = build_vllm_config(args)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
        )
        self._tempfile: tempfile.NamedTemporaryFile[str] | None = None
        self.worker: Worker | None = None
        self.engine: EdgeDecodeEngine | None = None
        self._old_use_v2_model_runner = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")

    def __enter__(self) -> "EdgeRuntime":
        require_cuda()
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        envs.disable_envs_cache()
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        self._tempfile = tempfile.NamedTemporaryFile(delete=False)
        distributed_init_method = f"file://{self._tempfile.name}"
        self.worker = Worker(
            vllm_config=self.vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method=distributed_init_method,
        )
        with set_current_vllm_config(self.vllm_config):
            self.worker.init_device()
            self.worker.load_model()
            platforms.current_platform.update_block_size_for_backend(
                self.vllm_config
            )
            kv_cache_specs = [self.worker.get_kv_cache_spec()]
            available_memory = [self.worker.determine_available_memory()]
            kv_cache_configs = get_kv_cache_configs(
                self.vllm_config,
                kv_cache_specs,
                available_memory,
            )
            scheduler_kv_cache_config = generate_scheduler_kv_cache_config(
                kv_cache_configs
            )
            self.vllm_config.cache_config.num_gpu_blocks = (
                scheduler_kv_cache_config.num_blocks
            )
            if scheduler_kv_cache_config.kv_cache_groups:
                self.vllm_config.cache_config.block_size = min(
                    group.kv_cache_spec.block_size
                    for group in scheduler_kv_cache_config.kv_cache_groups
                )
            self.vllm_config.validate_block_size()
            self.worker.initialize_from_config(kv_cache_configs[0])
            self.worker.compile_or_warm_up_model()

        kv_cache_manager = KVCacheManager(
            kv_cache_config=scheduler_kv_cache_config,
            max_model_len=self.vllm_config.model_config.max_model_len,
            hash_block_size=self.vllm_config.cache_config.block_size,
            enable_caching=self.vllm_config.cache_config.enable_prefix_caching,
            use_eagle=False,
            log_stats=False,
            enable_kv_cache_events=False,
            dcp_world_size=self.vllm_config.parallel_config.decode_context_parallel_size,
            pcp_world_size=self.vllm_config.parallel_config.prefill_context_parallel_size,
        )
        sampler = self.worker.model_runner.sampler
        if sampler is None:
            raise RuntimeError("worker model runner sampler is not initialized")
        self.engine = EdgeDecodeEngine(
            vllm_config=self.vllm_config,
            worker=self.worker,
            scheduler=EdgeSchedulerAdapter(kv_cache_manager=kv_cache_manager),
            state_bridge=EdgeStateBridge(),
            draft_sampler=DSSDEdgeDraftSampler(sampler),
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.worker is not None:
            self.worker.shutdown()
        cleanup_dist_env_and_memory()
        if self._tempfile is not None:
            try:
                Path(self._tempfile.name).unlink(missing_ok=True)
            except OSError:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self._old_use_v2_model_runner is None:
            os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
        else:
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = self._old_use_v2_model_runner
        envs.disable_envs_cache()


def benchmark_edge(
    args: argparse.Namespace,
    sampling_params: SamplingParams,
) -> BenchmarkResult:
    with EdgeRuntime(args) as runtime:
        prompt_token_ids = runtime.tokenizer(args.prompt).input_ids
        for warmup_idx in range(args.warmup_iters):
            runtime.engine.generate_local(
                req_id=f"warmup-{warmup_idx}",
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
            )
        synchronize_if_needed()

        total_tokens = 0
        total_seconds = 0.0
        for iter_idx in range(args.benchmark_iters):
            synchronize_if_needed()
            started = time.perf_counter()
            output_token_ids = runtime.engine.generate_local(
                req_id=f"bench-{iter_idx}",
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
            )
            synchronize_if_needed()
            elapsed = time.perf_counter() - started
            total_seconds += elapsed
            total_tokens += len(output_token_ids)

        return BenchmarkResult(
            engine="edge",
            model=args.model,
            prompt_tokens=len(prompt_token_ids),
            requested_output_tokens=args.max_new_tokens,
            generated_tokens=total_tokens,
            warmup_iters=args.warmup_iters,
            benchmark_iters=args.benchmark_iters,
            total_seconds=total_seconds,
            avg_seconds=total_seconds / max(args.benchmark_iters, 1),
            tokens_per_second=total_tokens / total_seconds,
        )


def benchmark_native(
    args: argparse.Namespace,
    sampling_params: SamplingParams,
) -> BenchmarkResult:
    require_cuda()
    old_use_v2_model_runner = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    if args.native_model_runner == "v2":
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    else:
        os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
    envs.disable_envs_cache()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    prompt_token_ids = tokenizer(args.prompt).input_ids
    prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
    llm = None
    try:
        llm = LLM(
            model=args.model,
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
            tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            enforce_eager=args.enforce_eager,
            enable_prefix_caching=False,
        )
        for _ in range(args.warmup_iters):
            llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
        synchronize_if_needed()

        total_tokens = 0
        total_seconds = 0.0
        for _ in range(args.benchmark_iters):
            synchronize_if_needed()
            started = time.perf_counter()
            outputs = llm.generate(
                [prompt],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            synchronize_if_needed()
            elapsed = time.perf_counter() - started
            total_seconds += elapsed
            total_tokens += len(outputs[0].outputs[0].token_ids)

        return BenchmarkResult(
            engine="native",
            model=args.model,
            prompt_tokens=len(prompt_token_ids),
            requested_output_tokens=args.max_new_tokens,
            generated_tokens=total_tokens,
            warmup_iters=args.warmup_iters,
            benchmark_iters=args.benchmark_iters,
            total_seconds=total_seconds,
            avg_seconds=total_seconds / max(args.benchmark_iters, 1),
            tokens_per_second=total_tokens / total_seconds,
        )
    finally:
        if llm is not None:
            del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if old_use_v2_model_runner is None:
            os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
        else:
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = old_use_v2_model_runner
        envs.disable_envs_cache()


def run_subprocess_for_engine(
    args: argparse.Namespace,
    engine: str,
) -> BenchmarkResult:
    cmd = [sys.executable, __file__]
    passthrough = [
        "--engine",
        engine,
        "--model",
        args.model,
        "--prompt",
        args.prompt,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--warmup-iters",
        str(args.warmup_iters),
        "--benchmark-iters",
        str(args.benchmark_iters),
        "--dtype",
        args.dtype,
        "--native-model-runner",
        args.native_model_runner,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--json",
    ]
    if args.trust_remote_code:
        passthrough.append("--trust-remote-code")
    if args.enforce_eager:
        passthrough.append("--enforce-eager")
    if args.honor_eos:
        passthrough.append("--honor-eos")
    completed = subprocess.run(
        cmd + passthrough,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip())
    return BenchmarkResult(**payload)


def print_results(results: list[BenchmarkResult], *, json_only: bool) -> None:
    if json_only:
        if len(results) == 1:
            print(json.dumps(asdict(results[0]), ensure_ascii=False))
        else:
            print(json.dumps([asdict(result) for result in results], ensure_ascii=False))
        return

    print(f"model: {results[0].model}")
    print(
        f"prompt_tokens: {results[0].prompt_tokens}, "
        f"max_new_tokens: {results[0].requested_output_tokens}, "
        f"warmup: {results[0].warmup_iters}, "
        f"iters: {results[0].benchmark_iters}"
    )
    print()
    header = (
        f"{'engine':<10} {'avg_s':>10} {'total_s':>10} "
        f"{'gen_toks':>10} {'tok/s':>12}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.engine:<10} {result.avg_seconds:>10.4f} "
            f"{result.total_seconds:>10.4f} {result.generated_tokens:>10d} "
            f"{result.tokens_per_second:>12.2f}"
        )


def main() -> None:
    args = parse_args()
    sampling_params = build_sampling_params(args)

    if args.engine == "both":
        results = [
            run_subprocess_for_engine(args, "native"),
            run_subprocess_for_engine(args, "edge"),
        ]
    elif args.engine == "native":
        results = [benchmark_native(args, sampling_params)]
    else:
        results = [benchmark_edge(args, sampling_params)]

    print_results(results, json_only=args.json)


if __name__ == "__main__":
    main()
