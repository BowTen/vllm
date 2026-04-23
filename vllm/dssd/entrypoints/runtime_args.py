from __future__ import annotations

import argparse

_DEFAULT_MAX_MODEL_LEN = 64
_DEFAULT_GPU_MEMORY_UTILIZATION = 0.01
_DEFAULT_KV_CACHE_MEMORY_BYTES = None
_DEFAULT_MAX_NUM_BATCHED_TOKENS = 64
_DEFAULT_MAX_NUM_SEQS = 2


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
