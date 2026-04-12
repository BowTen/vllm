# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "benchmarks" / "dssd" / "benchmark_verifier_vs_vllm.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_verifier_vs_vllm",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verifier_benchmark_script_exists_and_parses_engine_flag() -> None:
    module = _load_module()

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--engine",
            "verifier",
            "--model",
            "/tmp/model",
            "--prompt-len",
            "128",
            "--decode-tokens",
            "64",
        ]
    )

    assert args.engine == "verifier"
    assert args.model == "/tmp/model"
    assert args.prompt_len == 128
    assert args.decode_tokens == 64


def test_decode_only_counter_excludes_bootstrap_and_warmup() -> None:
    module = _load_module()

    counter = module.DecodeMeasurementCounter(warmup_tokens=2, target_tokens=2)

    assert counter.consume([101]) == []
    assert counter.consume([102]) == []
    assert counter.consume([103]) == []
    assert counter.consume([104]) == [104]
    assert counter.consume([105, 106]) == [105]
    assert counter.is_complete()
