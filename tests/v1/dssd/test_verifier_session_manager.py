# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
VLLM_DIR = ROOT / "vllm"
VERIFIER_DIR = VLLM_DIR / "v1" / "dssd" / "verifier"


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
_install_package_stub("vllm.v1.dssd.verifier", VERIFIER_DIR)

session_module = _load_module(
    "vllm.v1.dssd.verifier.session",
    VERIFIER_DIR / "session.py",
)

DSSDVerifierSessionManager = session_module.DSSDVerifierSessionManager


def test_verifier_session_manager_replays_cached_response():
    manager = DSSDVerifierSessionManager()
    manager.create_session("vs-1", sampling_params_fingerprint="abc")
    manager.cache_response("vs-1", seq_no=3, response={"accepted_count": 2})
    assert manager.get_cached_response("vs-1", seq_no=3) == {"accepted_count": 2}


def test_verifier_session_manager_rejects_out_of_order_seq():
    manager = DSSDVerifierSessionManager()
    manager.create_session("vs-1", sampling_params_fingerprint="abc")
    manager.update_seq_no("vs-1", 4)

    with pytest.raises(ValueError, match="out-of-order"):
        manager.ensure_next_seq_no("vs-1", 6)
