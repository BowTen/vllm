#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark single-request generation speed for native vLLM vs DSSD verifier.

This benchmark is intentionally narrow:
- batch size is fixed to 1
- the timer covers one full request from enqueue/open_session to the last token
- throughput is computed from the full generated token count in that request

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
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()


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
    native_model_runner: str


@dataclass
class RepeatResult:
    total_s: float
    tokens_per_s: float
    per_token_ms: float
    token_ids: list[int]


def _generated_tokens_per_request(config: BenchmarkConfig) -> int:
    return _required_total_tokens(config)


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
        "--native-model-runner",
        choices=("v1", "v2"),
        default="v1",
        help="Model runner used by the native engine benchmark.",
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
    parser.add_argument(
        "--json-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--validate-token-alignment",
        action="store_true",
        help="Assert native and verifier measured token IDs match in both mode.",
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


def _set_model_runner_env(model_runner: str) -> str | None:
    from vllm import envs

    old_value = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    if model_runner == "v2":
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    elif model_runner == "v1":
        os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
    else:
        raise ValueError(f"unsupported model runner: {model_runner}")
    envs.disable_envs_cache()
    return old_value


def _restore_model_runner_env(old_value: str | None) -> None:
    from vllm import envs

    if old_value is None:
        os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
    else:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = old_value
    envs.disable_envs_cache()


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
    old_model_runner = _set_model_runner_env(config.native_model_runner)
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
            results.append(
                _run_native_repeat(
                    engine,
                    config,
                    req_id=f"native-{repeat_idx}-{time.time_ns()}",
                    prompt_token_ids=prompt_token_ids,
                )
            )

        return results
    finally:
        _teardown_llm_engine(engine)
        _restore_model_runner_env(old_model_runner)


def _run_verifier_benchmark(
    config: BenchmarkConfig,
    prompt_token_ids: list[int],
) -> list[RepeatResult]:
    from vllm.config import set_current_vllm_config
    from vllm.distributed import cleanup_dist_env_and_memory
    from vllm.dssd.verifier.engine import VerifierDecodeEngine
    from vllm.dssd.verifier.sampler import DSSDVerifierSampler
    from vllm.dssd.verifier.state_bridge import VerifierStateBridge
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
    old_model_runner = _set_model_runner_env("v2")
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
            worker.compile_or_warm_up_model()

            kv_cache_manager = KVCacheManager(
                kv_cache_config=scheduler_kv_cache_config,
                max_model_len=vllm_config.model_config.max_model_len,
                hash_block_size=vllm_config.cache_config.block_size,
                enable_caching=vllm_config.cache_config.enable_prefix_caching,
                use_eagle=False,
                log_stats=False,
                enable_kv_cache_events=False,
                dcp_world_size=vllm_config.parallel_config.decode_context_parallel_size,
                pcp_world_size=(
                    vllm_config.parallel_config.prefill_context_parallel_size
                ),
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
                results.append(
                    _run_verifier_repeat(
                        engine,
                        config,
                        req_id=f"verifier-{repeat_idx}-{time.time_ns()}",
                        prompt_token_ids=prompt_token_ids,
                    )
                )

        return results
    finally:
        if worker is not None:
            try:
                worker.shutdown()
            finally:
                cleanup_dist_env_and_memory()
        _restore_model_runner_env(old_model_runner)


def _run_native_repeat(
    engine,
    config: BenchmarkConfig,
    *,
    req_id: str,
    prompt_token_ids: list[int],
) -> RepeatResult:
    from vllm.sampling_params import RequestOutputKind

    target_tokens = _generated_tokens_per_request(config)
    params = _make_sampling_params(
        target_tokens,
        output_kind=RequestOutputKind.DELTA,
    )
    generated_token_ids: list[int] = []

    _synchronize()
    t0 = time.perf_counter()
    engine.add_request(req_id, {"prompt_token_ids": prompt_token_ids}, params)
    while len(generated_token_ids) < target_tokens:
        outputs = engine.step()
        for output in outputs:
            if not output.outputs:
                continue
            generated_token_ids.extend(output.outputs[0].token_ids)
    _synchronize()
    total_s = time.perf_counter() - t0
    generated_token_ids = generated_token_ids[:target_tokens]
    return RepeatResult(
        total_s=total_s,
        tokens_per_s=target_tokens / total_s,
        per_token_ms=total_s * 1000.0 / target_tokens,
        token_ids=generated_token_ids,
    )


def _run_verifier_repeat(
    engine,
    config: BenchmarkConfig,
    *,
    req_id: str,
    prompt_token_ids: list[int],
) -> RepeatResult:
    target_tokens = _generated_tokens_per_request(config)
    _synchronize()
    t0 = time.perf_counter()
    generated_token_ids = engine.generate_local(
        req_id=req_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=_make_sampling_params(target_tokens),
    )
    _synchronize()
    total_s = time.perf_counter() - t0
    return RepeatResult(
        total_s=total_s,
        tokens_per_s=len(generated_token_ids) / total_s,
        per_token_ms=total_s * 1000.0 / len(generated_token_ids),
        token_ids=generated_token_ids,
    )


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


def _repeat_results_from_payload(
    payload: dict[str, Any],
    engine: str,
) -> list[RepeatResult]:
    return [RepeatResult(**repeat) for repeat in payload[engine]["repeats"]]


def _print_payload(payload: dict[str, Any]) -> None:
    print("Config")
    print(json.dumps(payload["config"], indent=2))
    if "native" in payload:
        _print_results("native", _repeat_results_from_payload(payload, "native"))
    if "verifier" in payload:
        _print_results("verifier", _repeat_results_from_payload(payload, "verifier"))
    if "speedup_verifier_over_native" in payload:
        print(
            "\nSpeedup verifier/native = "
            f"{payload['speedup_verifier_over_native']:.4f}x"
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


def _merge_both_payloads(
    native_payload: dict[str, Any],
    verifier_payload: dict[str, Any],
    *,
    validate_token_alignment: bool,
) -> dict[str, Any]:
    if native_payload["config"] != verifier_payload["config"]:
        raise RuntimeError(
            "native and verifier subprocesses used different benchmark configs"
        )

    if validate_token_alignment:
        _validate_token_alignment(
            _repeat_results_from_payload(native_payload, "native"),
            _repeat_results_from_payload(verifier_payload, "verifier"),
        )

    payload = {
        "config": native_payload["config"],
        "native": native_payload["native"],
        "verifier": verifier_payload["verifier"],
    }
    payload["speedup_verifier_over_native"] = (
        payload["verifier"]["summary"]["mean_tokens_per_s"]
        / payload["native"]["summary"]["mean_tokens_per_s"]
    )
    return payload


def _loads_trailing_json(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    start = text.rfind("{")
    decoder = json.JSONDecoder()
    while start >= 0:
        try:
            payload, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.rfind("{", 0, start)
            continue
        if text[end:].strip() == "" and isinstance(payload, dict):
            return payload
        start = text.rfind("{", 0, start)
    raise RuntimeError(
        "could not parse benchmark JSON payload from subprocess stdout\n"
        f"stdout tail:\n{text[-2000:]}"
    )


def run_subprocess_for_engine(
    args: argparse.Namespace,
    engine: str,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--engine",
        engine,
        "--model",
        args.model,
        "--prompt-len",
        str(args.prompt_len),
        "--decode-tokens",
        str(args.decode_tokens),
        "--warmup-tokens",
        str(args.warmup_tokens),
        "--repeats",
        str(args.repeats),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--dtype",
        args.dtype,
        "--native-model-runner",
        args.native_model_runner,
        "--json-only",
    ]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    cmd.append("--enforce-eager" if args.enforce_eager else "--no-enforce-eager")

    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"{engine} subprocess failed with exit code {exc.returncode}\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc
    return _loads_trailing_json(completed.stdout)


def _run_both_in_subprocesses(args: argparse.Namespace) -> dict[str, Any]:
    native_payload = run_subprocess_for_engine(args, "native")
    verifier_payload = run_subprocess_for_engine(args, "verifier")
    return _merge_both_payloads(
        native_payload,
        verifier_payload,
        validate_token_alignment=args.validate_token_alignment,
    )


def _run_single_engine_payload(args: argparse.Namespace) -> dict[str, Any]:
    _require_cuda()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

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
        native_model_runner=args.native_model_runner,
    )

    payload: dict[str, Any] = {"config": asdict(config)}
    if args.engine == "native":
        native_results = _run_native_benchmark(config, prompt_token_ids)
        payload["native"] = {
            "repeats": [asdict(result) for result in native_results],
            "summary": _summarize(native_results),
        }
    else:
        verifier_results = _run_verifier_benchmark(config, prompt_token_ids)
        payload["verifier"] = {
            "repeats": [asdict(result) for result in verifier_results],
            "summary": _summarize(verifier_results),
        }
    return payload


def main() -> None:
    args = build_parser().parse_args()
    if args.engine == "both":
        payload = _run_both_in_subprocesses(args)
    else:
        payload = _run_single_engine_payload(args)

    if args.json_only:
        print(json.dumps(payload))
    else:
        _print_payload(payload)
        if args.json:
            print(json.dumps(payload, indent=2))
    gc.collect()


if __name__ == "__main__":
    main()
