# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "benchmarks" / "dssd" / "benchmark_edge_verifier_decode.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_edge_verifier_decode",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dssd_system_benchmark_script_exists_and_parses_device_flags() -> None:
    module = _load_module()

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--model",
            "/tmp/model",
            "--edge-cuda-visible-devices",
            "0",
            "--verifier-cuda-visible-devices",
            "1",
            "--prompt-len",
            "64",
            "--decode-tokens",
            "32",
            "--gamma",
            "2",
        ]
    )

    assert args.model == "/tmp/model"
    assert args.edge_cuda_visible_devices == "0"
    assert args.verifier_cuda_visible_devices == "1"
    assert args.prompt_len == 64
    assert args.decode_tokens == 32
    assert args.gamma == 2


def test_dssd_system_decode_counter_handles_multi_token_rounds() -> None:
    module = _load_module()

    counter = module.DecodeMeasurementCounter(warmup_tokens=2, target_tokens=3)

    assert counter.consume([101]) == []
    assert counter.consume([102, 103]) == []
    assert counter.consume([104, 105]) == [104, 105]
    assert counter.consume([106, 107]) == [106]
    assert counter.is_complete()
