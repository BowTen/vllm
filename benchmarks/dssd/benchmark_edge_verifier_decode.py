#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark decode-only speed for the current DSSD edge+verifier system.

This benchmark intentionally measures the current split setup:
- edge runtime runs in the current process
- verifier runtime runs in a separate local process
- edge talks to verifier through HTTPVerifierTransport
- model startup is excluded from timing
- prompt prefill and bootstrap token are excluded from timing
- an optional decode warmup window is excluded from timing

Example:
    PYTHONPATH=$PWD ./.venv/bin/python benchmarks/dssd/benchmark_edge_verifier_decode.py \
      --model /data/zz/hf/Qwen/Qwen3-0.6B-msfull \
      --edge-cuda-visible-devices 0 \
      --verifier-cuda-visible-devices 1 \
      --decode-tokens 64 \
      --json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any


@dataclass
class BenchmarkConfig:
    model: str
    edge_model: str
    verifier_model: str
    edge_model_runner: str
    verifier_model_runner: str
    edge_cuda_visible_devices: str
    verifier_cuda_visible_devices: str
    verifier_host: str
    verifier_port: int
    prompt_len: int
    decode_tokens: int
    warmup_tokens: int
    repeats: int
    gamma: int
    gpu_memory_utilization: float
    max_model_len: int
    kv_cache_memory_bytes: int
    max_num_batched_tokens: int
    max_num_seqs: int
    enforce_eager: bool
    async_scheduling: bool


@dataclass
class RepeatResult:
    total_s: float
    tokens_per_s: float
    per_token_ms: float
    token_ids: list[int]


class DecodeMeasurementCounter:
    """Track decode tokens while excluding bootstrap and warmup."""

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=_resolve_default_model(),
        help="Default model path or HF model name used by both edge and verifier.",
    )
    parser.add_argument(
        "--edge-model",
        default=None,
        help="Override the edge model path or HF model name.",
    )
    parser.add_argument(
        "--verifier-model",
        default=None,
        help="Override the verifier model path or HF model name.",
    )
    parser.add_argument(
        "--edge-model-runner",
        choices=("v1", "v2"),
        default="v1",
        help="Model runner used by the edge runtime.",
    )
    parser.add_argument(
        "--verifier-model-runner",
        choices=("v1", "v2"),
        default="v1",
        help="Model runner used by the verifier runtime.",
    )
    parser.add_argument(
        "--edge-cuda-visible-devices",
        default="0",
        help="CUDA_VISIBLE_DEVICES used by the benchmark process for edge runtime.",
    )
    parser.add_argument(
        "--verifier-cuda-visible-devices",
        default="1",
        help="CUDA_VISIBLE_DEVICES used by the verifier subprocess.",
    )
    parser.add_argument(
        "--verifier-host",
        default="127.0.0.1",
        help="Verifier server bind host.",
    )
    parser.add_argument(
        "--verifier-port",
        type=int,
        default=18021,
        help="Verifier server bind port.",
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
        "--gamma",
        type=int,
        default=0,
        help="Edge draft length per verifier round.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.01,
        help="GPU memory utilization passed to vLLM. Keep this low when "
        "kv_cache_memory_bytes is used on a shared GPU.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional explicit max_model_len override.",
    )
    parser.add_argument(
        "--kv-cache-memory-bytes",
        type=int,
        default=128 * 1024 * 1024,
        help="KV cache memory budget passed to vLLM.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=64,
        help="max_num_batched_tokens passed to vLLM.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=2,
        help="max_num_seqs passed to vLLM.",
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable eager mode. Use --no-enforce-eager to allow compile/cudagraph.",
    )
    parser.add_argument(
        "--async-scheduling",
        action="store_true",
        default=False,
        help="Forward async_scheduling to runtime creation.",
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


def _build_prompt_token_ids(model: str, prompt_len: int) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model,
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


def _resolved_max_num_batched_tokens(args: argparse.Namespace) -> int:
    return max(args.max_num_batched_tokens, args.prompt_len)


def _make_sampling_params(max_tokens: int):
    from vllm.sampling_params import SamplingParams

    return SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=max_tokens,
    )


def _require_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")


def _synchronize() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _start_verifier_server(
    config: BenchmarkConfig,
) -> tuple[subprocess.Popen[str], str, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    ready_dir = Path(tempfile.mkdtemp(prefix="dssd-bench-ready-"))
    ready_file = ready_dir / "verifier-ready.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = config.verifier_cuda_visible_devices
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm.dssd.entrypoints.verifier_server",
            "--host",
            config.verifier_host,
            "--port",
            str(config.verifier_port),
            "--ready-file",
            str(ready_file),
            "--model",
            config.verifier_model,
            "--model-runner-version",
            config.verifier_model_runner,
            "--gamma",
            str(config.gamma),
            "--max-model-len",
            str(config.max_model_len),
            "--gpu-memory-utilization",
            str(config.gpu_memory_utilization),
            "--kv-cache-memory-bytes",
            str(config.kv_cache_memory_bytes),
            "--max-num-batched-tokens",
            str(config.max_num_batched_tokens),
            "--max-num-seqs",
            str(config.max_num_seqs),
            "--async-scheduling",
        ]
        if config.async_scheduling
        else [
            sys.executable,
            "-m",
            "vllm.dssd.entrypoints.verifier_server",
            "--host",
            config.verifier_host,
            "--port",
            str(config.verifier_port),
            "--ready-file",
            str(ready_file),
            "--model",
            config.verifier_model,
            "--model-runner-version",
            config.verifier_model_runner,
            "--gamma",
            str(config.gamma),
            "--max-model-len",
            str(config.max_model_len),
            "--gpu-memory-utilization",
            str(config.gpu_memory_utilization),
            "--kv-cache-memory-bytes",
            str(config.kv_cache_memory_bytes),
            "--max-num-batched-tokens",
            str(config.max_num_batched_tokens),
            "--max-num-seqs",
            str(config.max_num_seqs),
        ],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    server_url = _wait_for_server_url(ready_file, proc)
    return proc, server_url, ready_dir


def _wait_for_server_url(
    ready_file: Path,
    proc: subprocess.Popen[str],
    *,
    timeout_s: float = 120.0,
) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise AssertionError(
                "verifier server exited early\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )
        if ready_file.exists():
            return json.loads(ready_file.read_text())["server_url"]
        time.sleep(0.05)
    raise AssertionError("verifier ready file was not created in time")


def _stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        proc.communicate()
        return
    proc.terminate()
    try:
        proc.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate(timeout=15)


def _build_edge_service(config: BenchmarkConfig, server_url: str):
    from vllm.dssd.entrypoints import runtime_factory

    runtime_args = SimpleNamespace(
        model=config.edge_model,
        model_runner_version=config.edge_model_runner,
        verifier_url=server_url,
        eos_token_id=-1,
        gamma=config.gamma,
        enforce_eager=config.enforce_eager,
        async_scheduling=config.async_scheduling,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_utilization,
        kv_cache_memory_bytes=config.kv_cache_memory_bytes,
        max_num_batched_tokens=config.max_num_batched_tokens,
        max_num_seqs=config.max_num_seqs,
    )
    return runtime_factory.build_real_edge_service(runtime_args)


def _run_dssd_benchmark(
    config: BenchmarkConfig,
    prompt_token_ids: list[int],
) -> list[RepeatResult]:
    verifier_proc = None
    ready_dir = None
    edge_service = None
    edge_cleanup = None
    results: list[RepeatResult] = []
    try:
        verifier_proc, server_url, ready_dir = _start_verifier_server(config)
        edge_service, edge_cleanup = _build_edge_service(config, server_url)

        for repeat_idx in range(config.repeats):
            req_id = f"dssd-{repeat_idx}-{time.time_ns()}"
            sampling_params = _make_sampling_params(
                1 + config.warmup_tokens + config.decode_tokens + max(config.gamma, 1)
            )
            opened = edge_service.open_session(
                req_id=req_id,
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
            )
            session = edge_service.decode_engine.sessions[req_id]
            next_token = opened.bootstrap_token_id
            counter = DecodeMeasurementCounter(
                warmup_tokens=config.warmup_tokens,
                target_tokens=config.decode_tokens,
            )
            counter.consume([next_token])

            measured_token_ids: list[int] = []
            measured_total_s = 0.0
            try:
                while not counter.is_complete():
                    output_len_before = len(session.committed_output_ids())
                    round_state = edge_service.decode_engine.draft(
                        session,
                        next_token,
                        config.gamma,
                    )

                    from vllm.dssd.protocol import VerifyRoundRequest

                    _synchronize()
                    t0 = time.perf_counter()
                    response = edge_service.verifier.verify_round(
                        VerifyRoundRequest(
                            req_id=req_id,
                            committed_token_id=next_token,
                            draft_token_ids=list(round_state.draft_token_ids),
                            draft_q_values=list(round_state.draft_q_values),
                        )
                    )
                    next_token, _ = edge_service._commit_verify_result(  # noqa: SLF001
                        session,
                        response,
                    )
                    _synchronize()
                    step_s = time.perf_counter() - t0

                    new_output_ids = list(
                        session.committed_output_ids()[output_len_before:]
                    )
                    measured_step_token_ids = counter.consume(new_output_ids)
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
            finally:
                if req_id in edge_service.decode_engine.sessions:
                    edge_service.close_session(req_id)

        return results
    finally:
        if verifier_proc is not None:
            _stop_process(verifier_proc)
        # This benchmark exits the process via _hard_exit() immediately after
        # printing results. Running the real runtime cleanup here can hang in
        # distributed teardown for V1-based runs, so rely on process exit to
        # reclaim local edge runtime state instead.
        if ready_dir is not None:
            for path in ready_dir.glob("*"):
                path.unlink(missing_ok=True)
            ready_dir.rmdir()


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


def main() -> None:
    args = build_parser().parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.edge_cuda_visible_devices
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = (
        "1" if args.edge_model_runner == "v2" else "0"
    )

    _require_cuda()

    edge_model = args.edge_model or args.model
    verifier_model = args.verifier_model or args.model
    prompt_token_ids = _build_prompt_token_ids(edge_model, args.prompt_len)
    max_model_len = args.max_model_len or max(
        args.prompt_len + 1 + args.warmup_tokens + args.decode_tokens + args.gamma + 32,
        args.prompt_len + 64,
    )
    config = BenchmarkConfig(
        model=args.model,
        edge_model=edge_model,
        verifier_model=verifier_model,
        edge_model_runner=args.edge_model_runner,
        verifier_model_runner=args.verifier_model_runner,
        edge_cuda_visible_devices=args.edge_cuda_visible_devices,
        verifier_cuda_visible_devices=args.verifier_cuda_visible_devices,
        verifier_host=args.verifier_host,
        verifier_port=args.verifier_port,
        prompt_len=args.prompt_len,
        decode_tokens=args.decode_tokens,
        warmup_tokens=args.warmup_tokens,
        repeats=args.repeats,
        gamma=args.gamma,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        max_num_batched_tokens=_resolved_max_num_batched_tokens(args),
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        async_scheduling=args.async_scheduling,
    )

    print("Config")
    print(json.dumps(asdict(config), indent=2))

    results = _run_dssd_benchmark(config, prompt_token_ids)
    _print_results("dssd-edge+verifier", results)

    if args.json:
        payload: dict[str, Any] = {
            "config": asdict(config),
            "dssd": {
                "repeats": [asdict(result) for result in results],
                "summary": _summarize(results),
            },
        }
        print(json.dumps(payload, indent=2))

    gc.collect()


def _hard_exit(code: int) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        _hard_exit(1)
    _hard_exit(0)
