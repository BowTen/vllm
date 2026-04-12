#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark decode-only speed for native vLLM vs DSSD verifier.

This benchmark is intentionally narrow:
- batch size is fixed to 1
- prompt prefill is excluded from timing
- the bootstrap token after prefill is excluded from timing
- an optional decode warmup window is excluded from timing

Example:
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD \\
      python benchmarks/dssd/benchmark_verifier_vs_vllm.py \\
      --model /data/zz/hf/Qwen/Qwen3-0.6B-msfull --json
"""

import argparse
import gc
import json
import os
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkConfig:
    model: str
    prompt_len: int
    decode_tokens: int
    warmup_tokens: int
    repeats: int
    gpu_memory_utilization: float
    max_model_len: int
    dtype: str
    enforce_eager: bool
    trust_remote_code: bool


@dataclass
class RepeatResult:
    total_s: float
    tokens_per_s: float
    per_token_ms: float
    token_ids: list[int]


class DecodeMeasurementCounter:
    """Track decode tokens while excluding prefill-adjacent bootstrap/warmup."""

    def __init__(self, warmup_tokens: int, target_tokens: int) -> None:
        self.warmup_tokens = warmup_tokens
        self.target_tokens = target_tokens
        self._bootstrap_consumed = False
        self._measured_count = 0

    def consume(self, token_ids: list[int]) -> list[int]:
        measured: list[int] = []
        for token_id in token_ids:
            if not self._bootstrap_consumed:
                self._bootstrap_consumed = True
                continue
            if self.warmup_tokens > 0:
                self.warmup_tokens -= 1
                continue
            if self._measured_count >= self.target_tokens:
                break
            measured.append(int(token_id))
            self._measured_count += 1
        return measured

    def is_complete(self) -> bool:
        return self._measured_count >= self.target_tokens


class BenchmarkVerifierSchedulerAdapter:
    """Verifier scheduler wrapper with explicit request block hashing."""

    def __init__(self, kv_cache_manager, block_hasher) -> None:
        from vllm.dssd.verifier.scheduler import VerifierSchedulerAdapter

        self._delegate = VerifierSchedulerAdapter(kv_cache_manager=kv_cache_manager)
        self.kv_cache_manager = kv_cache_manager
        self._block_hasher = block_hasher

    def allocate_blocks(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params,
        lora_request=None,
    ) -> tuple[list[int], ...]:
        from vllm.v1.request import Request

        request = Request(
            request_id=req_id,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=sampling_params,
            pooling_params=None,
            lora_request=lora_request,
            block_hasher=self._block_hasher,
        )
        blocks = self.kv_cache_manager.allocate_slots(
            request=request,
            num_new_tokens=request.num_tokens - request.num_computed_tokens,
        )
        if blocks is None:
            raise RuntimeError("prefill cannot allocate KV blocks for verifier")
        self._delegate._pending_new_block_ids_to_zero = (  # noqa: SLF001
            self._delegate._drain_new_block_ids_to_zero(blocks)  # noqa: SLF001
        )
        block_ids = blocks.get_block_ids(allow_none=True)
        return ([],) if block_ids is None else block_ids

    def _make_request(self, session):
        from vllm.v1.request import Request

        request = Request(
            request_id=session.req_id,
            prompt_token_ids=list(session.prompt_token_ids),
            sampling_params=session.sampling_params,
            pooling_params=None,
            lora_request=session.lora_request,
            block_hasher=self._block_hasher,
        )
        finalized_output = session.token_ids[session.prompt_len:]
        if finalized_output:
            request.append_output_token_ids(finalized_output)
        request.num_computed_tokens = session.num_computed_tokens
        return request

    def __getattr__(self, name: str):
        return getattr(self._delegate, name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        choices=("both", "native", "verifier"),
        default="both",
        help="Benchmark native, verifier, or both engines.",
    )
    parser.add_argument(
        "--model",
        default=_resolve_default_model(),
        help="Local model path or HF model name.",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=128,
        help="Prompt token length used for prefill.",
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=128,
        help="Measured decode token count per repeat.",
    )
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=16,
        help="Decode tokens to run after bootstrap before timing starts.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Measured repeats.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization passed to vLLM.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Model dtype passed to vLLM.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forward trust_remote_code to tokenizer/model loading.",
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable eager mode. Use --no-enforce-eager to allow compile/cudagraph.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON in addition to the human-readable summary.",
    )
    return parser


def _resolve_default_model() -> str:
    preferred = Path("/data/zz/hf/Qwen/Qwen3-0.6B-msfull")
    if preferred.is_dir():
        return str(preferred)
    return "Qwen/Qwen3-0.6B"


def _required_total_tokens(config: BenchmarkConfig) -> int:
    return 1 + config.warmup_tokens + config.decode_tokens


def _build_prompt_token_ids(model: str, prompt_len: int, *,
                            trust_remote_code: bool) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=trust_remote_code,
        local_files_only=Path(model).exists(),
    )
    seed_text = "Benchmark prompt. " * max(prompt_len, 64)
    token_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not token_ids:
        raise RuntimeError("tokenizer returned no prompt tokens")

    prompt_token_ids: list[int] = []
    while len(prompt_token_ids) < prompt_len:
        prompt_token_ids.extend(token_ids)
    return prompt_token_ids[:prompt_len]


def _make_sampling_params(max_tokens: int, *, output_kind=None):
    from vllm.sampling_params import SamplingParams

    kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "ignore_eos": True,
        "max_tokens": max_tokens,
    }
    if output_kind is not None:
        kwargs["output_kind"] = output_kind
    return SamplingParams(**kwargs)


def _synchronize() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")


def _ensure_cuda_platform_if_needed() -> None:
    import sys
    import torch

    import vllm.platforms as platforms
    from vllm.platforms.cuda import CudaPlatform

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


def _build_vllm_config(config: BenchmarkConfig):
    from vllm.engine.arg_utils import EngineArgs

    _ensure_cuda_platform_if_needed()
    engine_args = EngineArgs(
        model=config.model,
        trust_remote_code=config.trust_remote_code,
        dtype=config.dtype,
        tensor_parallel_size=1,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_model_len=config.max_model_len,
        enforce_eager=config.enforce_eager,
        enable_prefix_caching=False,
        disable_log_stats=True,
    )
    return engine_args.create_engine_config()


def _teardown_llm_engine(engine) -> None:
    if engine is None:
        return
    from vllm.distributed import cleanup_dist_env_and_memory

    try:
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
            return
        renderer = getattr(engine, "renderer", None)
        if renderer is not None and hasattr(renderer, "shutdown"):
            renderer.shutdown()
        engine.engine_core.shutdown()
    finally:
        cleanup_dist_env_and_memory()


def _run_native_benchmark(
    config: BenchmarkConfig,
    prompt_token_ids: list[int],
) -> list[RepeatResult]:
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.engine.arg_utils import EngineArgs

    engine = None
    results: list[RepeatResult] = []
    try:
        engine_args = EngineArgs(
            model=config.model,
            trust_remote_code=config.trust_remote_code,
            dtype=config.dtype,
            tensor_parallel_size=1,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            enforce_eager=config.enforce_eager,
            enable_prefix_caching=False,
            disable_log_stats=True,
        )
        engine = LLMEngine.from_engine_args(engine_args, enable_multiprocessing=False)

        for repeat_idx in range(config.repeats):
            req_id = f"native-{repeat_idx}-{time.time_ns()}"
            params = _make_sampling_params(
                _required_total_tokens(config),
                output_kind=RequestOutputKind.DELTA,
            )
            engine.add_request(req_id, {"prompt_token_ids": prompt_token_ids}, params)

            counter = DecodeMeasurementCounter(
                warmup_tokens=config.warmup_tokens,
                target_tokens=config.decode_tokens,
            )
            measured_token_ids: list[int] = []
            measured_total_s = 0.0

            while not counter.is_complete():
                _synchronize()
                t0 = time.perf_counter()
                outputs = engine.step()
                _synchronize()
                step_s = time.perf_counter() - t0

                step_token_ids: list[int] = []
                for output in outputs:
                    if not output.outputs:
                        continue
                    step_token_ids.extend(output.outputs[0].token_ids)

                measured_step_token_ids = counter.consume(step_token_ids)
                if measured_step_token_ids:
                    measured_token_ids.extend(measured_step_token_ids)
                    measured_total_s += step_s

            results.append(
                RepeatResult(
                    total_s=measured_total_s,
                    tokens_per_s=config.decode_tokens / measured_total_s,
                    per_token_ms=measured_total_s * 1000.0 / config.decode_tokens,
                    token_ids=measured_token_ids,
                )
            )

        return results
    finally:
        _teardown_llm_engine(engine)


def _run_verifier_benchmark(
    config: BenchmarkConfig,
    prompt_token_ids: list[int],
) -> list[RepeatResult]:
    from vllm.config import set_current_vllm_config
    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.dssd.verifier.engine import VerifierDecodeEngine
    from vllm.dssd.verifier.sampler import DSSDVerifierSampler
    from vllm.dssd.verifier.state_bridge import VerifierStateBridge
    from vllm.dssd.verifier.types import VerifierRoundRequest
    from vllm.utils.hashing import get_hash_fn_by_name
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import (
        generate_scheduler_kv_cache_config,
        get_kv_cache_configs,
        get_request_block_hasher,
        init_none_hash,
    )
    from vllm.v1.worker.gpu_worker import Worker

    worker = None
    results: list[RepeatResult] = []
    try:
        vllm_config = _build_vllm_config(config)
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

            kv_cache_manager = KVCacheManager(
                kv_cache_config=scheduler_kv_cache_config,
                max_model_len=vllm_config.model_config.max_model_len,
                hash_block_size=vllm_config.cache_config.block_size,
            )

            hash_fn = get_hash_fn_by_name(
                vllm_config.cache_config.prefix_caching_hash_algo
            )
            init_none_hash(hash_fn)
            block_hasher = get_request_block_hasher(
                vllm_config.cache_config.block_size,
                hash_fn,
            )

            engine = VerifierDecodeEngine(
                vllm_config=vllm_config,
                worker=worker,
                scheduler=BenchmarkVerifierSchedulerAdapter(
                    kv_cache_manager=kv_cache_manager,
                    block_hasher=block_hasher,
                ),
                state_bridge=VerifierStateBridge(),
                verifier_sampler=DSSDVerifierSampler(
                    sampler=worker.model_runner.sampler,
                    num_speculative_steps=worker.model_runner.num_speculative_steps,
                ),
            )

            for repeat_idx in range(config.repeats):
                req_id = f"verifier-{repeat_idx}-{time.time_ns()}"
                opened = engine.open_session(
                    req_id=req_id,
                    prompt_token_ids=prompt_token_ids,
                    sampling_params=_make_sampling_params(_required_total_tokens(config)),
                )

                session = engine.sessions[req_id]
                next_token = opened.bootstrap_token_id

                for _ in range(config.warmup_tokens):
                    result = engine.verify_round(
                        session,
                        VerifierRoundRequest(
                            req_id=req_id,
                            committed_token_id=next_token,
                            draft_token_ids=[],
                            draft_q_values=[],
                        ),
                    )
                    assert result.bonus_token_id is not None
                    next_token = result.bonus_token_id

                measured_token_ids: list[int] = []
                measured_total_s = 0.0
                for _ in range(config.decode_tokens):
                    _synchronize()
                    t0 = time.perf_counter()
                    result = engine.verify_round(
                        session,
                        VerifierRoundRequest(
                            req_id=req_id,
                            committed_token_id=next_token,
                            draft_token_ids=[],
                            draft_q_values=[],
                        ),
                    )
                    _synchronize()
                    measured_total_s += time.perf_counter() - t0
                    assert result.bonus_token_id is not None
                    next_token = result.bonus_token_id
                    measured_token_ids.append(next_token)

                results.append(
                    RepeatResult(
                        total_s=measured_total_s,
                        tokens_per_s=config.decode_tokens / measured_total_s,
                        per_token_ms=measured_total_s * 1000.0 / config.decode_tokens,
                        token_ids=measured_token_ids,
                    )
                )
                engine.close_session(session)

        return results
    finally:
        if worker is not None:
            try:
                worker.shutdown()
            finally:
                cleanup_dist_env_and_memory()


def _summarize(results: list[RepeatResult]) -> dict[str, float]:
    throughputs = [result.tokens_per_s for result in results]
    latencies = [result.per_token_ms for result in results]
    totals = [result.total_s for result in results]
    return {
        "mean_tokens_per_s": statistics.mean(throughputs),
        "stdev_tokens_per_s": statistics.stdev(throughputs)
        if len(throughputs) > 1
        else 0.0,
        "mean_per_token_ms": statistics.mean(latencies),
        "stdev_per_token_ms": statistics.stdev(latencies)
        if len(latencies) > 1
        else 0.0,
        "mean_total_s": statistics.mean(totals),
    }


def _print_results(label: str, results: list[RepeatResult]) -> None:
    summary = _summarize(results)
    print(f"\n[{label}]")
    for idx, result in enumerate(results, 1):
        print(
            f"repeat={idx} "
            f"tokens/s={result.tokens_per_s:.2f} "
            f"per_token_ms={result.per_token_ms:.3f} "
            f"total_s={result.total_s:.4f}"
        )
    print(
        "summary "
        f"mean_tokens/s={summary['mean_tokens_per_s']:.2f} "
        f"stdev_tokens/s={summary['stdev_tokens_per_s']:.2f} "
        f"mean_per_token_ms={summary['mean_per_token_ms']:.3f}"
    )


def _validate_token_alignment(
    native_results: list[RepeatResult],
    verifier_results: list[RepeatResult],
) -> None:
    if len(native_results) != len(verifier_results):
        raise AssertionError("native/verifier repeat counts do not match")
    for repeat_idx, (native, verifier) in enumerate(
        zip(native_results, verifier_results, strict=True),
        start=1,
    ):
        if native.token_ids != verifier.token_ids:
            raise AssertionError(
                f"token mismatch at repeat={repeat_idx}: "
                f"native={native.token_ids} verifier={verifier.token_ids}"
            )


def main() -> None:
    args = build_parser().parse_args()

    _require_cuda()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"

    from vllm import envs

    envs.disable_envs_cache()
    prompt_token_ids = _build_prompt_token_ids(
        args.model,
        args.prompt_len,
        trust_remote_code=args.trust_remote_code,
    )
    max_model_len = max(
        args.prompt_len + 1 + args.warmup_tokens + args.decode_tokens + 32,
        args.prompt_len + 64,
    )
    config = BenchmarkConfig(
        model=args.model,
        prompt_len=args.prompt_len,
        decode_tokens=args.decode_tokens,
        warmup_tokens=args.warmup_tokens,
        repeats=args.repeats,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
    )

    print("Config")
    print(json.dumps(asdict(config), indent=2))

    native_results: list[RepeatResult] | None = None
    verifier_results: list[RepeatResult] | None = None
    if args.engine in ("both", "native"):
        native_results = _run_native_benchmark(config, prompt_token_ids)
        _print_results("native", native_results)

    if args.engine in ("both", "verifier"):
        verifier_results = _run_verifier_benchmark(config, prompt_token_ids)
        _print_results("verifier", verifier_results)

    payload: dict[str, Any] = {"config": asdict(config)}
    if native_results is not None:
        payload["native"] = {
            "repeats": [asdict(result) for result in native_results],
            "summary": _summarize(native_results),
        }
    if verifier_results is not None:
        payload["verifier"] = {
            "repeats": [asdict(result) for result in verifier_results],
            "summary": _summarize(verifier_results),
        }
    if native_results is not None and verifier_results is not None:
        _validate_token_alignment(native_results, verifier_results)
        payload["speedup_verifier_over_native"] = (
            payload["verifier"]["summary"]["mean_tokens_per_s"]
            / payload["native"]["summary"]["mean_tokens_per_s"]
        )
        print(
            "\nSpeedup verifier/native = "
            f"{payload['speedup_verifier_over_native']:.4f}x"
        )

    if args.json:
        print(json.dumps(payload, indent=2))

    gc.collect()


if __name__ == "__main__":
    main()
