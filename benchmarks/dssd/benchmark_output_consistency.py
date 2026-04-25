#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check greedy output consistency between DSSD and target-only generation.

The script runs one model pair and one gamma per process. Keeping each gamma in
its own process avoids long-lived vLLM runtime teardown issues in V1-based DSSD
experiments.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from benchmarks.dssd.benchmark_edge_verifier_decode import (
    BenchmarkConfig,
    _build_edge_service,
    _hard_exit,
    _resolved_max_num_batched_tokens,
    _start_verifier_server,
    _stop_process,
)


@dataclass
class ConsistencyCaseResult:
    prompt_index: int
    prompt_tokens: int
    output_len: int
    gamma: int
    matched: bool
    first_mismatch_index: int | None
    target_token_ids: list[int]
    dssd_token_ids: list[int]
    draft_acceptance_rate: float
    all_accept_round_rate: float
    avg_accepted_len_per_round: float


@dataclass
class ConsistencySummary:
    total_prompts: int
    matched_prompts: int
    output_consistency_rate: float
    mean_draft_acceptance_rate: float
    mean_all_accept_round_rate: float
    mean_avg_accepted_len_per_round: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-model", required=True)
    parser.add_argument("--verifier-model", required=True)
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer used to encode prompts. Defaults to --edge-model.",
    )
    parser.add_argument("--prompt-jsonl", required=True)
    parser.add_argument("--prompt-count", type=int, default=5)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--gamma", type=int, required=True)
    parser.add_argument("--edge-model-runner", choices=("v1", "v2"), default="v1")
    parser.add_argument(
        "--verifier-model-runner",
        choices=("v1", "v2"),
        default="v1",
    )
    parser.add_argument("--edge-cuda-visible-devices", default="0")
    parser.add_argument("--verifier-cuda-visible-devices", default="1")
    parser.add_argument("--verifier-host", default="127.0.0.1")
    parser.add_argument("--verifier-port", type=int, default=18021)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--max-num-seqs", type=int, default=2)
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
    parser.add_argument("--async-scheduling", action="store_true", default=False)
    parser.add_argument("--output-json", required=True)
    return parser


def _load_prompt_texts(prompt_jsonl: str, prompt_count: int) -> list[str]:
    prompt_path = Path(prompt_jsonl)
    prompts: list[str] = []
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
            prompt = item.get("prompt") if isinstance(item, dict) else None
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(
                    f"{prompt_path}:{line_number}: missing non-empty prompt"
                )
            prompts.append(prompt)
            if len(prompts) >= prompt_count:
                break

    if len(prompts) < prompt_count:
        raise ValueError(
            f"{prompt_path}: requested {prompt_count} prompts, found {len(prompts)}"
        )
    return prompts


def _build_prompt_token_ids(
    *,
    tokenizer_name: str,
    prompt_jsonl: str,
    prompt_count: int,
    prompt_len: int,
) -> list[list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        local_files_only=Path(tokenizer_name).exists(),
    )
    prompts = _load_prompt_texts(prompt_jsonl, prompt_count)
    prompt_token_ids: list[list[int]] = []
    for prompt_index, prompt in enumerate(prompts):
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) < prompt_len:
            raise ValueError(
                f"prompt {prompt_index} tokenized to {len(token_ids)} tokens, "
                f"shorter than --prompt-len {prompt_len}"
            )
        prompt_token_ids.append(token_ids[:prompt_len])
    return prompt_token_ids


def _make_sampling_params(max_tokens: int):
    from vllm.sampling_params import SamplingParams

    return SamplingParams(
        temperature=0.0,
        ignore_eos=True,
        max_tokens=max_tokens,
    )


def _first_mismatch_index(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _summary(results: list[ConsistencyCaseResult]) -> ConsistencySummary:
    matched = sum(1 for result in results if result.matched)
    total = len(results)
    return ConsistencySummary(
        total_prompts=total,
        matched_prompts=matched,
        output_consistency_rate=matched / total if total else 0.0,
        mean_draft_acceptance_rate=(
            sum(result.draft_acceptance_rate for result in results) / total
            if total
            else 0.0
        ),
        mean_all_accept_round_rate=(
            sum(result.all_accept_round_rate for result in results) / total
            if total
            else 0.0
        ),
        mean_avg_accepted_len_per_round=(
            sum(result.avg_accepted_len_per_round for result in results) / total
            if total
            else 0.0
        ),
    )


def _resolved_max_model_len(args: argparse.Namespace) -> int:
    if args.max_model_len > 0:
        return args.max_model_len
    return max(
        args.prompt_len + args.output_len + args.gamma + 32,
        args.prompt_len + 64,
    )


def _build_config(args: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig(
        model=args.verifier_model,
        edge_model=args.edge_model,
        verifier_model=args.verifier_model,
        edge_model_runner=args.edge_model_runner,
        verifier_model_runner=args.verifier_model_runner,
        edge_cuda_visible_devices=args.edge_cuda_visible_devices,
        verifier_cuda_visible_devices=args.verifier_cuda_visible_devices,
        verifier_host=args.verifier_host,
        verifier_port=args.verifier_port,
        prompt_len=args.prompt_len,
        decode_tokens=args.output_len,
        warmup_tokens=0,
        repeats=1,
        warmup_repeats=0,
        gamma=args.gamma,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=_resolved_max_model_len(args),
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        max_num_batched_tokens=_resolved_max_num_batched_tokens(args),
        max_num_seqs=args.max_num_seqs,
        enforce_eager=args.enforce_eager,
        async_scheduling=args.async_scheduling,
        prompt_jsonl=args.prompt_jsonl,
        request_latency_ms=0.0,
        request_bandwidth_bytes_per_s=None,
        response_latency_ms=0.0,
        response_bandwidth_bytes_per_s=None,
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.edge_cuda_visible_devices
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = (
        "1" if args.edge_model_runner == "v2" else "0"
    )

    prompt_token_ids = _build_prompt_token_ids(
        tokenizer_name=args.tokenizer or args.edge_model,
        prompt_jsonl=args.prompt_jsonl,
        prompt_count=args.prompt_count,
        prompt_len=args.prompt_len,
    )
    config = _build_config(args)
    verifier_proc = None
    ready_dir = None
    results: list[ConsistencyCaseResult] = []
    try:
        verifier_proc, server_url, ready_dir = _start_verifier_server(config)
        edge_service, _edge_cleanup = _build_edge_service(config, server_url)
        sampling_params = _make_sampling_params(args.output_len)

        for prompt_index, token_ids in enumerate(prompt_token_ids):
            target_ids = edge_service.verifier.generate(
                req_id=f"target-{args.gamma}-{prompt_index}-{time.time_ns()}",
                prompt_token_ids=token_ids,
                sampling_params=sampling_params,
            )
            dssd_ids, stats = edge_service.generate_with_stats(
                req_id=f"dssd-{args.gamma}-{prompt_index}-{time.time_ns()}",
                prompt_token_ids=token_ids,
                sampling_params=sampling_params,
            )
            mismatch_index = _first_mismatch_index(target_ids, dssd_ids)
            results.append(
                ConsistencyCaseResult(
                    prompt_index=prompt_index,
                    prompt_tokens=len(token_ids),
                    output_len=args.output_len,
                    gamma=args.gamma,
                    matched=mismatch_index is None,
                    first_mismatch_index=mismatch_index,
                    target_token_ids=list(target_ids),
                    dssd_token_ids=list(dssd_ids),
                    draft_acceptance_rate=stats.draft_acceptance_rate,
                    all_accept_round_rate=stats.all_accept_round_rate,
                    avg_accepted_len_per_round=stats.avg_accepted_len_per_round,
                )
            )

        summary = _summary(results)
        return {
            "config": asdict(config),
            "tokenizer": args.tokenizer or args.edge_model,
            "prompt_count": args.prompt_count,
            "summary": asdict(summary),
            "results": [asdict(result) for result in results],
        }
    finally:
        if verifier_proc is not None:
            _stop_process(verifier_proc)
        if ready_dir is not None:
            for path in ready_dir.glob("*"):
                path.unlink(missing_ok=True)
            ready_dir.rmdir()
        gc.collect()


def main() -> None:
    args = build_parser().parse_args()
    payload = _run(args)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        _hard_exit(1)
    _hard_exit(0)
