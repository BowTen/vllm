# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
VLLM_DIR = ROOT / "vllm"
DSSD_DIR = VLLM_DIR / "v1" / "dssd"


def _install_package_stub(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_install_package_stub("vllm", VLLM_DIR)
_install_package_stub("vllm.v1", VLLM_DIR / "v1")
_install_package_stub("vllm.v1.dssd", DSSD_DIR)
_install_package_stub("vllm.v1.dssd.worker", DSSD_DIR / "worker")

verifier_runner_module = _load_module(
    "vllm.v1.dssd.worker.verifier_runner",
    DSSD_DIR / "worker" / "verifier_runner.py",
)

build_verifier_result = verifier_runner_module.build_verifier_result


def test_build_verifier_result_extracts_target_positions():
    result = build_verifier_result(
        draft_token_ids=[11, 12],
        q_values=[0.6, 0.4],
        target_probs=[[0.2, 0.8], [0.9, 0.1], [0.7, 0.3]],
    )

    assert result.accepted_count == 0
    assert result.seq_probs == [[0.2, 0.8], [0.9, 0.1]]
    assert result.bonus_probs == [0.7, 0.3]
