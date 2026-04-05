# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
VLLM_DIR = ROOT / "vllm"
DSSD_WORKER_DIR = VLLM_DIR / "v1" / "dssd" / "worker"


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
_install_package_stub("vllm.v1.dssd", VLLM_DIR / "v1" / "dssd")
_install_package_stub("vllm.v1.dssd.worker", DSSD_WORKER_DIR)

resample_module = _load_module(
    "vllm.v1.dssd.worker.resample",
    DSSD_WORKER_DIR / "resample.py",
)

compute_residual_distribution = resample_module.compute_residual_distribution


def test_compute_residual_distribution_clamps_negative_mass():
    p = torch.tensor([0.7, 0.2, 0.1])
    q = torch.tensor([0.2, 0.5, 0.3])

    residual = compute_residual_distribution(p, q)

    assert torch.all(residual >= 0)
    assert torch.isclose(residual.sum(), torch.tensor(1.0))
