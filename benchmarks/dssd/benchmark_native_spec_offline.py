#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline native speculative decoding benchmark with fixed token prompts.

This script exists to benchmark native vLLM speculative decoding on exactly the
same prompt_token_ids slice used by DSSD experiments. It avoids `vllm bench
latency` so callers can:

- reuse a prompt from a JSONL file verbatim
- truncate to an exact prompt token length
- read speculative acceptance counters from `llm.get_metrics()`

Example:
    CUDA_VISIBLE_DEVICES=0 \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    VLLM_USE_V2_MODEL_RUNNER=0 \
    .venv/bin/python benchmarks/dssd/benchmark_native_spec_offline.py \
      --model /root/autodl-tmp/opt-6.7b \
      --draft-model /root/autodl-tmp/opt-125m \
      --tokenizer /root/autodl-tmp/opt-125m \
      --prompt-jsonl benchmarks/dssd/prompts/high_acceptance_opt.jsonl \
      --prompt-len 128 \
      --output-len 256 \
      --num-speculative-tokens 4 \
      --num-iters-warmup 2 \
      --num-iters 5 \
      --max-model-len 416 \
      --gpu-memory-utilization 0.9 \
      --no-enforce-eager \
      --output-json benchmarks/dssd/results/native-spec-opt-6.7b-draft-opt-125m-high-accept-p128-o256-k4-v1-no-eager.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


@dataclass
class BenchmarkResult:
    model: str
    draft_model: str
    tokenizer: str
    prompt_jsonl: str
    prompt_index: int
    prompt_len: int
    output_len: int
    num_speculative_tokens: int
    num_iters_warmup: int
    num_iters: int
    enforce_eager: bool
    avg_latency: float
    output_tokens_per_second: float
    num_drafts: float
    num_draft_tokens: float
    num_accepted_tokens: float
    acceptance_rate: float
    acceptance_length: float
    raw_prompt_preview: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline native speculative decoding benchmark.",
    )
    parser.add_argument("--model", required=True, help="Target model path.")
    parser.add_argument(
        "--draft-model",
        required=True,
        help="Draft model path in speculative_config.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer path used to encode the prompt. Defaults to --draft-model.",
    )
    parser.add_argument(
        "--prompt-jsonl",
        required=True,
        help="JSONL file containing prompt records with a `prompt` field.",
    )
    parser.add_argument(
        "--prompt-index",
        type=int,
        default=0,
        help="0-based record index selected from --prompt-jsonl.",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=128,
        help="Exact number of prompt tokens to keep.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=256,
        help="Requested decode length.",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        required=True,
        help="Draft length k used by native speculative decoding.",
    )
    parser.add_argument(
        "--num-iters-warmup",
        type=int,
        default=2,
        help="Warmup iterations excluded from timing.",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=5,
        help="Measured iterations.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Passed through to LLM(...).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=416,
        help="Passed through to LLM(...).",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Passed through to LLM(...).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Sampling seed.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional output JSON path.",
    )
    eager_group = parser.add_mutually_exclusive_group()
    eager_group.add_argument(
        "--enforce-eager",
        dest="enforce_eager",
        action="store_true",
        help="Enable eager execution.",
    )
    eager_group.add_argument(
        "--no-enforce-eager",
        dest="enforce_eager",
        action="store_false",
        help="Disable eager execution.",
    )
    parser.set_defaults(enforce_eager=False)
    return parser.parse_args()


def _load_prompt_text(prompt_jsonl: str, prompt_index: int) -> str:
    records: list[dict[str, Any]] = []
    with open(prompt_jsonl, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON on line {line_no} of {prompt_jsonl}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"line {line_no} of {prompt_jsonl} is not a JSON object"
                )
            records.append(record)

    if not records:
        raise ValueError(f"{prompt_jsonl} contains no prompt records")
    if prompt_index < 0 or prompt_index >= len(records):
        raise IndexError(
            f"--prompt-index {prompt_index} out of range for {len(records)} records"
        )

    prompt = records[prompt_index].get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(
            f"record {prompt_index} in {prompt_jsonl} has no non-empty `prompt`"
        )
    return prompt


def _build_prompt_token_ids(
    tokenizer_name: str,
    prompt_jsonl: str,
    prompt_index: int,
    prompt_len: int,
) -> tuple[str, list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        local_files_only=Path(tokenizer_name).exists(),
    )
    prompt_text = _load_prompt_text(prompt_jsonl, prompt_index)
    token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if len(token_ids) < prompt_len:
        raise ValueError(
            f"prompt tokenized to {len(token_ids)} tokens, shorter than "
            f"--prompt-len {prompt_len}"
        )
    return prompt_text, token_ids[:prompt_len]


def _synchronize_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _metric_value(metrics: list[Any], name: str) -> float:
    for metric in metrics:
        if metric.name == name:
            return float(metric.value)
    raise KeyError(f"metric {name!r} not found in llm.get_metrics()")


def _run(args: argparse.Namespace) -> BenchmarkResult:
    tokenizer_name = args.tokenizer or args.draft_model
    prompt_text, prompt_token_ids = _build_prompt_token_ids(
        tokenizer_name=tokenizer_name,
        prompt_jsonl=args.prompt_jsonl,
        prompt_index=args.prompt_index,
        prompt_len=args.prompt_len,
    )
    prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
    sampling_params = SamplingParams(
        max_tokens=args.output_len,
        temperature=0.0,
        ignore_eos=True,
        detokenize=False,
        seed=args.seed,
    )
    llm = LLM(
        model=args.model,
        tokenizer=tokenizer_name,
        speculative_config={
            "model": args.draft_model,
            "method": "draft_model",
            "num_speculative_tokens": args.num_speculative_tokens,
            "draft_tensor_parallel_size": 1,
        },
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        disable_log_stats=False,
        enable_prefix_caching=False,
    )
    try:
        for _ in range(args.num_iters_warmup):
            llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)

        metrics_before = llm.get_metrics()
        total_latency = 0.0
        for _ in range(args.num_iters):
            _synchronize_if_needed()
            start = time.perf_counter()
            llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
            _synchronize_if_needed()
            total_latency += time.perf_counter() - start
        metrics_after = llm.get_metrics()
    finally:
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    num_drafts = (
        _metric_value(metrics_after, "vllm:spec_decode_num_drafts")
        - _metric_value(metrics_before, "vllm:spec_decode_num_drafts")
    )
    num_draft_tokens = (
        _metric_value(metrics_after, "vllm:spec_decode_num_draft_tokens")
        - _metric_value(metrics_before, "vllm:spec_decode_num_draft_tokens")
    )
    num_accepted_tokens = (
        _metric_value(metrics_after, "vllm:spec_decode_num_accepted_tokens")
        - _metric_value(metrics_before, "vllm:spec_decode_num_accepted_tokens")
    )
    acceptance_rate = (
        num_accepted_tokens / num_draft_tokens
        if num_draft_tokens > 0
        else math.nan
    )
    acceptance_length = 1.0 + (
        num_accepted_tokens / num_drafts if num_drafts > 0 else 0.0
    )
    avg_latency = total_latency / max(args.num_iters, 1)
    output_tokens_per_second = (
        args.output_len / avg_latency if avg_latency > 0 else math.nan
    )
    prompt_preview = prompt_text[:120].replace("\n", "\\n")

    return BenchmarkResult(
        model=args.model,
        draft_model=args.draft_model,
        tokenizer=tokenizer_name,
        prompt_jsonl=args.prompt_jsonl,
        prompt_index=args.prompt_index,
        prompt_len=args.prompt_len,
        output_len=args.output_len,
        num_speculative_tokens=args.num_speculative_tokens,
        num_iters_warmup=args.num_iters_warmup,
        num_iters=args.num_iters,
        enforce_eager=args.enforce_eager,
        avg_latency=avg_latency,
        output_tokens_per_second=output_tokens_per_second,
        num_drafts=num_drafts,
        num_draft_tokens=num_draft_tokens,
        num_accepted_tokens=num_accepted_tokens,
        acceptance_rate=acceptance_rate,
        acceptance_length=acceptance_length,
        raw_prompt_preview=prompt_preview,
    )


def main() -> None:
    args = _parse_args()
    result = _run(args)
    result_json = json.dumps(asdict(result), indent=2, ensure_ascii=False)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result_json + "\n", encoding="utf-8")
    print(result_json)


if __name__ == "__main__":
    main()
