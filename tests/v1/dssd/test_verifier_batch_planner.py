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
_install_package_stub("vllm.v1.dssd.engine", DSSD_DIR / "engine")

batch_planner_module = _load_module(
    "vllm.v1.dssd.engine.batch_planner",
    DSSD_DIR / "engine" / "batch_planner.py",
)
session_runner_module = _load_module(
    "vllm.v1.dssd.engine.session_runner",
    DSSD_DIR / "engine" / "session_runner.py",
)

VerifierRoundBatcher = batch_planner_module.VerifierRoundBatcher
DSSDSessionRunner = session_runner_module.DSSDSessionRunner


def test_batcher_groups_requests_by_gamma_and_signature():
    batcher = VerifierRoundBatcher()
    batcher.add({"session_id": "a", "gamma": 4, "sampling_signature": "s1"})
    batcher.add({"session_id": "b", "gamma": 4, "sampling_signature": "s1"})
    batcher.add({"session_id": "c", "gamma": 2, "sampling_signature": "s1"})

    groups = batcher.flush()

    assert sorted(len(group) for group in groups) == [1, 2]


def test_session_runner_exposes_verifier_batcher():
    runner = DSSDSessionRunner()

    assert isinstance(runner.verifier_batcher, VerifierRoundBatcher)
