# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
VLLM_DIR = ROOT / "vllm"
DSSD_EDGE_DIR = VLLM_DIR / "v1" / "dssd" / "edge"


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
_install_package_stub("vllm.v1.dssd.edge", DSSD_EDGE_DIR)

session_module = _load_module(
    "vllm.v1.dssd.edge.session",
    DSSD_EDGE_DIR / "session.py",
)

DSSDEdgeRoundCache = session_module.DSSDEdgeRoundCache
DSSDEdgeSessionState = session_module.DSSDEdgeSessionState


def test_edge_session_starts_with_empty_pending_delta():
    session = DSSDEdgeSessionState(
        request_id="req-1",
        local_session_id="edge-1",
        verifier_binding_id="bind-1",
        verifier_session_id="vs-1",
        prompt_token_ids=[1, 2, 3],
    )

    assert session.pending_prefix_delta_token_ids == []
    assert session.committed_token_ids == []
    assert session.seq_no == 0


def test_edge_round_cache_preserves_round_state():
    cache = DSSDEdgeRoundCache(
        seq_no=2,
        gamma=4,
        draft_token_ids=[10, 11],
        q_values=[0.6, 0.4],
        q_dists_handle="handle-1",
    )

    assert cache.seq_no == 2
    assert cache.gamma == 4
    assert cache.q_dists_handle == "handle-1"
