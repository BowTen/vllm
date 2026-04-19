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
from urllib import request as urllib_request

from vllm.dssd.transport import FakeNetwork
from vllm.dssd.transport.fake_network import LinkTimingModel


_CALIBRATION_PAYLOAD_BYTES = (1, 65536, 1048576)
_CALIBRATION_REPEATS = 5


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
    kv_cache_memory_bytes: int | None
    max_num_batched_tokens: int
    max_num_seqs: int
    enforce_eager: bool
    async_scheduling: bool
    warmup_repeats: int = 0
    prompt_jsonl: str | None = None
    request_latency_ms: float = 0.0
    request_bandwidth_bytes_per_s: float | None = None
    response_latency_ms: float = 0.0
    response_bandwidth_bytes_per_s: float | None = None
    network_calibration: dict[str, Any] | None = None


@dataclass
class RepeatResult:
    request_total_s: float
    request_tokens_per_s: float
    request_per_token_ms: float
    decode_total_s: float
    decode_tokens_per_s: float
    decode_per_token_ms: float
    verify_path_total_s: float
    verify_path_tokens_per_s: float
    verify_path_per_token_ms: float
    token_ids: list[int]
    draft_acceptance_rate: float
    all_accept_round_rate: float
    avg_accepted_len_per_round: float


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
        "--prompt-jsonl",
        default=None,
        help=(
            "Optional JSONL file containing real prompts. The benchmark uses "
            "the first non-empty prompt and truncates it to --prompt-len tokens."
        ),
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
        help=(
            "Decode tokens to run after bootstrap before decode/verify-path "
            "timing starts. These tokens are still included in request timing."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Measured repeats.",
    )
    parser.add_argument(
        "--warmup-repeats",
        type=int,
        default=0,
        help="Full requests to run before timing repeats. Excluded from results.",
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
        default=None,
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
    parser.add_argument(
        "--request-latency-ms",
        type=float,
        default=0.0,
        help="Simulated one-way latency added before each edge->verifier request.",
    )
    parser.add_argument(
        "--request-bandwidth-bytes-per-s",
        type=float,
        default=None,
        help="Simulated bandwidth limit for each edge->verifier request payload.",
    )
    parser.add_argument(
        "--response-latency-ms",
        type=float,
        default=0.0,
        help="Simulated one-way latency added after each verifier->edge response.",
    )
    parser.add_argument(
        "--response-bandwidth-bytes-per-s",
        type=float,
        default=None,
        help="Simulated bandwidth limit for each verifier->edge response payload.",
    )
    return parser


def _resolve_default_model() -> str:
    preferred = Path("/data/zz/hf/Qwen/Qwen3-0.6B-msfull")
    if preferred.is_dir():
        return str(preferred)
    return "Qwen/Qwen3-0.6B"


def _load_prompt_text_from_jsonl(prompt_jsonl: str) -> str:
    prompt_path = Path(prompt_jsonl)
    with prompt_path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{prompt_path}:{line_number}: invalid JSON"
                ) from exc
            if not isinstance(item, dict):
                raise ValueError(
                    f"{prompt_path}:{line_number}: expected JSON object"
                )
            prompt = item.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"{prompt_path}:{line_number}: JSON object must contain a "
                    "non-empty string 'prompt'"
                )
            return prompt
    raise ValueError(f"{prompt_path}: no prompt found")


def _build_prompt_token_ids(
    model: str,
    prompt_len: int,
    prompt_jsonl: str | None = None,
) -> list[int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model,
        local_files_only=Path(model).exists(),
    )
    seed_text = (
        _load_prompt_text_from_jsonl(prompt_jsonl)
        if prompt_jsonl is not None
        else "Benchmark prompt. " * max(prompt_len, 64)
    )
    token_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not token_ids:
        raise RuntimeError("tokenizer returned no prompt tokens")
    if prompt_jsonl is not None:
        if len(token_ids) < prompt_len:
            raise ValueError(
                "prompt-jsonl prompt tokenized to "
                f"{len(token_ids)} tokens, shorter than --prompt-len "
                f"{prompt_len}"
            )
        return token_ids[:prompt_len]

    prompt_token_ids: list[int] = []
    while len(prompt_token_ids) < prompt_len:
        prompt_token_ids.extend(token_ids)
    return prompt_token_ids[:prompt_len]


def _resolved_max_num_batched_tokens(args: argparse.Namespace) -> int:
    return max(args.max_num_batched_tokens, args.prompt_len)


def _build_network(
    *,
    direction: str,
    latency_ms: float,
    bandwidth_bytes_per_s: float | None,
    local_link_model: LinkTimingModel | None = None,
) -> FakeNetwork | None:
    if latency_ms <= 0.0 and bandwidth_bytes_per_s is None:
        return None
    return FakeNetwork(
        fixed_latency_ms=latency_ms,
        bandwidth_bytes_per_s=bandwidth_bytes_per_s,
        local_link_model=local_link_model,
        warning_label=direction,
    )


def _fit_link_timing_model(
    payload_bytes: list[int],
    elapsed_s: list[float],
) -> LinkTimingModel:
    if len(payload_bytes) != len(elapsed_s):
        raise ValueError("payload_bytes and elapsed_s must have the same length")
    if len(payload_bytes) < 2:
        raise ValueError("at least two samples are required for link fitting")

    x_values = [float(value) for value in payload_bytes]
    y_values = [float(value) for value in elapsed_s]
    x_mean = statistics.mean(x_values)
    y_mean = statistics.mean(y_values)
    numerator = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values, strict=True)
    )
    denominator = sum((x_value - x_mean) ** 2 for x_value in x_values)
    slope = numerator / denominator if denominator > 0.0 else 0.0
    intercept = y_mean - slope * x_mean
    bandwidth = None if slope <= 0.0 else 1.0 / slope
    return LinkTimingModel(
        fixed_latency_ms=max(intercept, 0.0) * 1000.0,
        bandwidth_bytes_per_s=bandwidth,
    )


def _measure_calibration_roundtrip_s(
    *,
    server_url: str,
    request_bytes: int,
    response_bytes: int,
) -> float:
    opener = urllib_request.build_opener(urllib_request.ProxyHandler({}))
    request_body = b"0" * request_bytes
    http_request = urllib_request.Request(
        url=f"{server_url}/calibrate?response_bytes={response_bytes}",
        data=request_body,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    t0 = time.perf_counter()
    with opener.open(http_request, timeout=30.0) as response:
        response_body = response.read()
    elapsed_s = time.perf_counter() - t0
    if len(response_body) != response_bytes:
        raise AssertionError(
            "calibration response size mismatch: "
            f"expected {response_bytes}, got {len(response_body)}"
        )
    return elapsed_s


def _calibrate_direction(
    *,
    server_url: str,
    direction: str,
) -> dict[str, Any]:
    sample_times_s: list[float] = []
    for payload_bytes in _CALIBRATION_PAYLOAD_BYTES:
        if direction == "request":
            _measure_calibration_roundtrip_s(
                server_url=server_url,
                request_bytes=payload_bytes,
                response_bytes=1,
            )
        elif direction == "response":
            _measure_calibration_roundtrip_s(
                server_url=server_url,
                request_bytes=1,
                response_bytes=payload_bytes,
            )
        timings = []
        for _ in range(_CALIBRATION_REPEATS):
            if direction == "request":
                timings.append(
                    _measure_calibration_roundtrip_s(
                        server_url=server_url,
                        request_bytes=payload_bytes,
                        response_bytes=1,
                    )
                )
            elif direction == "response":
                timings.append(
                    _measure_calibration_roundtrip_s(
                        server_url=server_url,
                        request_bytes=1,
                        response_bytes=payload_bytes,
                    )
                )
            else:
                raise ValueError(f"unknown direction: {direction}")
        sample_times_s.append(statistics.median(timings))

    local_link_model = _fit_link_timing_model(
        payload_bytes=list(_CALIBRATION_PAYLOAD_BYTES),
        elapsed_s=sample_times_s,
    )
    return {
        "sample_payload_bytes": list(_CALIBRATION_PAYLOAD_BYTES),
        "sample_elapsed_ms": [value * 1000.0 for value in sample_times_s],
        "local_fixed_latency_ms": local_link_model.fixed_latency_ms,
        "local_bandwidth_bytes_per_s": local_link_model.bandwidth_bytes_per_s,
        "local_link_model": local_link_model,
    }


def _calibrate_http_networks(
    config: BenchmarkConfig,
    server_url: str,
) -> tuple[FakeNetwork | None, FakeNetwork | None, dict[str, Any] | None]:
    calibration: dict[str, Any] = {}

    request_network = None
    if config.request_latency_ms > 0.0 or config.request_bandwidth_bytes_per_s is not None:
        request_info = _calibrate_direction(
            server_url=server_url,
            direction="request",
        )
        calibration["request"] = {
            key: value for key, value in request_info.items() if key != "local_link_model"
        }
        request_network = _build_network(
            direction="request",
            latency_ms=config.request_latency_ms,
            bandwidth_bytes_per_s=config.request_bandwidth_bytes_per_s,
            local_link_model=request_info["local_link_model"],
        )

    response_network = None
    if config.response_latency_ms > 0.0 or config.response_bandwidth_bytes_per_s is not None:
        response_info = _calibrate_direction(
            server_url=server_url,
            direction="response",
        )
        calibration["response"] = {
            key: value for key, value in response_info.items() if key != "local_link_model"
        }
        response_network = _build_network(
            direction="response",
            latency_ms=config.response_latency_ms,
            bandwidth_bytes_per_s=config.response_bandwidth_bytes_per_s,
            local_link_model=response_info["local_link_model"],
        )

    return request_network, response_network, calibration or None


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
    base_cmd = [
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
        "--max-num-batched-tokens",
        str(config.max_num_batched_tokens),
        "--max-num-seqs",
        str(config.max_num_seqs),
    ]
    if config.kv_cache_memory_bytes is not None:
        base_cmd.extend(
            [
                "--kv-cache-memory-bytes",
                str(config.kv_cache_memory_bytes),
            ]
        )
    if config.async_scheduling:
        base_cmd.append("--async-scheduling")
    proc = subprocess.Popen(
        base_cmd,
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

    (
        request_network,
        response_network,
        calibration,
    ) = _calibrate_http_networks(config, server_url)
    config.network_calibration = calibration

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
        request_network=request_network,
        response_network=response_network,
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

        total_repeats = config.warmup_repeats + config.repeats
        for repeat_idx in range(total_repeats):
            req_id = f"dssd-{repeat_idx}-{time.time_ns()}"
            sampling_params = _make_sampling_params(
                1 + config.warmup_tokens + config.decode_tokens + max(config.gamma, 1)
            )
            from vllm.dssd.protocol import VerifyRoundRequest

            _synchronize()
            request_t0 = time.perf_counter()
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
            measured_decode_total_s = 0.0
            measured_verify_path_total_s = 0.0
            total_rounds = 0
            all_accept_rounds = 0
            total_draft_tokens = 0
            total_accepted_tokens = 0
            try:
                while not counter.is_complete():
                    output_len_before = len(session.committed_output_ids())
                    _synchronize()
                    round_t0 = time.perf_counter()
                    round_state = edge_service.decode_engine.draft(
                        session,
                        next_token,
                        config.gamma,
                    )

                    _synchronize()
                    verify_path_t0 = time.perf_counter()
                    response = edge_service.verifier.verify_round(
                        VerifyRoundRequest(
                            req_id=req_id,
                            committed_token_id=next_token,
                            draft_token_ids=list(round_state.draft_token_ids),
                            draft_q_values=list(round_state.draft_q_values),
                        )
                    )
                    draft_len = len(round_state.draft_token_ids)
                    total_rounds += 1
                    total_draft_tokens += draft_len
                    total_accepted_tokens += response.accepted_len
                    if response.accepted_len == draft_len:
                        all_accept_rounds += 1
                    next_token, _ = edge_service._commit_verify_result(  # noqa: SLF001
                        session,
                        response,
                    )
                    _synchronize()
                    round_end = time.perf_counter()
                    round_s = round_end - round_t0
                    verify_path_s = round_end - verify_path_t0

                    new_output_ids = list(
                        session.committed_output_ids()[output_len_before:]
                    )
                    measured_step_token_ids = counter.consume(new_output_ids)
                    if measured_step_token_ids:
                        measured_token_ids.extend(measured_step_token_ids)
                        measured_decode_total_s += round_s
                        measured_verify_path_total_s += verify_path_s

                _synchronize()
                request_total_s = time.perf_counter() - request_t0

                if repeat_idx >= config.warmup_repeats:
                    results.append(
                        RepeatResult(
                            request_total_s=request_total_s,
                            request_tokens_per_s=(
                                config.decode_tokens / request_total_s
                            ),
                            request_per_token_ms=(
                                request_total_s * 1000.0 / config.decode_tokens
                            ),
                            decode_total_s=measured_decode_total_s,
                            decode_tokens_per_s=(
                                config.decode_tokens / measured_decode_total_s
                            ),
                            decode_per_token_ms=(
                                measured_decode_total_s
                                * 1000.0
                                / config.decode_tokens
                            ),
                            verify_path_total_s=measured_verify_path_total_s,
                            verify_path_tokens_per_s=(
                                config.decode_tokens
                                / measured_verify_path_total_s
                            ),
                            verify_path_per_token_ms=(
                                measured_verify_path_total_s
                                * 1000.0
                                / config.decode_tokens
                            ),
                            token_ids=measured_token_ids,
                            draft_acceptance_rate=(
                                total_accepted_tokens / total_draft_tokens
                                if total_draft_tokens > 0
                                else 0.0
                            ),
                            all_accept_round_rate=(
                                all_accept_rounds / total_rounds
                                if total_rounds > 0
                                else 0.0
                            ),
                            avg_accepted_len_per_round=(
                                total_accepted_tokens / total_rounds
                                if total_rounds > 0
                                else 0.0
                            ),
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
    request_throughputs = [result.request_tokens_per_s for result in results]
    request_latencies = [result.request_per_token_ms for result in results]
    request_totals = [result.request_total_s for result in results]
    decode_throughputs = [result.decode_tokens_per_s for result in results]
    decode_latencies = [result.decode_per_token_ms for result in results]
    decode_totals = [result.decode_total_s for result in results]
    verify_path_throughputs = [
        result.verify_path_tokens_per_s for result in results
    ]
    verify_path_latencies = [
        result.verify_path_per_token_ms for result in results
    ]
    verify_path_totals = [result.verify_path_total_s for result in results]
    draft_acceptance_rates = [result.draft_acceptance_rate for result in results]
    all_accept_round_rates = [result.all_accept_round_rate for result in results]
    avg_accepted_lens = [result.avg_accepted_len_per_round for result in results]
    return {
        "mean_request_tokens_per_s": statistics.mean(request_throughputs),
        "stdev_request_tokens_per_s": statistics.stdev(request_throughputs)
        if len(request_throughputs) > 1
        else 0.0,
        "mean_request_per_token_ms": statistics.mean(request_latencies),
        "stdev_request_per_token_ms": statistics.stdev(request_latencies)
        if len(request_latencies) > 1
        else 0.0,
        "mean_request_total_s": statistics.mean(request_totals),
        "mean_decode_tokens_per_s": statistics.mean(decode_throughputs),
        "stdev_decode_tokens_per_s": statistics.stdev(decode_throughputs)
        if len(decode_throughputs) > 1
        else 0.0,
        "mean_decode_per_token_ms": statistics.mean(decode_latencies),
        "stdev_decode_per_token_ms": statistics.stdev(decode_latencies)
        if len(decode_latencies) > 1
        else 0.0,
        "mean_decode_total_s": statistics.mean(decode_totals),
        "mean_verify_path_tokens_per_s": statistics.mean(verify_path_throughputs),
        "stdev_verify_path_tokens_per_s": statistics.stdev(verify_path_throughputs)
        if len(verify_path_throughputs) > 1
        else 0.0,
        "mean_verify_path_per_token_ms": statistics.mean(verify_path_latencies),
        "stdev_verify_path_per_token_ms": statistics.stdev(verify_path_latencies)
        if len(verify_path_latencies) > 1
        else 0.0,
        "mean_verify_path_total_s": statistics.mean(verify_path_totals),
        "mean_draft_acceptance_rate": statistics.mean(draft_acceptance_rates),
        "mean_all_accept_round_rate": statistics.mean(all_accept_round_rates),
        "mean_avg_accepted_len_per_round": statistics.mean(avg_accepted_lens),
    }


def _print_results(label: str, results: list[RepeatResult]) -> None:
    summary = _summarize(results)
    print(f"\n[{label}]")
    for idx, result in enumerate(results, 1):
        print(
            f"repeat={idx} "
            f"request_tokens/s={result.request_tokens_per_s:.2f} "
            f"request_total_s={result.request_total_s:.4f} "
            f"decode_tokens/s={result.decode_tokens_per_s:.2f} "
            f"decode_total_s={result.decode_total_s:.4f} "
            f"verify_path_tokens/s={result.verify_path_tokens_per_s:.2f} "
            f"verify_path_total_s={result.verify_path_total_s:.4f} "
            f"draft_accept={result.draft_acceptance_rate:.3f} "
            f"all_accept_round={result.all_accept_round_rate:.3f} "
            f"avg_accepted_len={result.avg_accepted_len_per_round:.3f}"
        )
    print(
        "summary "
        f"mean_request_tokens/s={summary['mean_request_tokens_per_s']:.2f} "
        f"mean_decode_tokens/s={summary['mean_decode_tokens_per_s']:.2f} "
        f"mean_verify_path_tokens/s={summary['mean_verify_path_tokens_per_s']:.2f} "
        f"mean_draft_accept={summary['mean_draft_acceptance_rate']:.3f} "
        f"mean_all_accept_round={summary['mean_all_accept_round_rate']:.3f}"
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
    prompt_token_ids = _build_prompt_token_ids(
        edge_model,
        args.prompt_len,
        args.prompt_jsonl,
    )
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
        warmup_repeats=args.warmup_repeats,
        prompt_jsonl=args.prompt_jsonl,
        repeats=args.repeats,
        gamma=args.gamma,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        max_num_batched_tokens=_resolved_max_num_batched_tokens(args),
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        async_scheduling=args.async_scheduling,
        request_latency_ms=args.request_latency_ms,
        request_bandwidth_bytes_per_s=args.request_bandwidth_bytes_per_s,
        response_latency_ms=args.response_latency_ms,
        response_bandwidth_bytes_per_s=args.response_bandwidth_bytes_per_s,
    )

    print("Config")
    print(json.dumps(asdict(config), indent=2))

    results = _run_dssd_benchmark(config, prompt_token_ids)
    _print_results("dssd-edge+verifier", results)

    if args.json:
        payload: dict[str, Any] = {
            "config": asdict(config),
            "network_calibration": config.network_calibration,
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
