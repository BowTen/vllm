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

coordinator_module = _load_module(
    "vllm.v1.dssd.edge.coordinator",
    DSSD_EDGE_DIR / "coordinator.py",
)
protocol_module = _load_module(
    "vllm.v1.dssd.protocol",
    VLLM_DIR / "v1" / "dssd" / "protocol.py",
)

DSSDRoundCoordinator = coordinator_module.DSSDRoundCoordinator
VerifyRoundResponse = protocol_module.VerifyRoundResponse


def test_round_coordinator_commits_bonus_token_on_full_accept():
    coordinator = DSSDRoundCoordinator(edge_engine=None, transport=None)
    response = VerifyRoundResponse(
        verifier_session_id="vs-1",
        seq_no=0,
        accepted_count=2,
        all_accepted=True,
        bonus_token_id=99,
        reject_index=None,
        reject_target_probs=None,
        finished=False,
        finish_reason=None,
    )

    committed = coordinator._build_committed_tokens(
        [10, 11],
        response,
        resampled_token=None,
    )

    assert committed == [10, 11, 99]


def test_round_coordinator_uses_resampled_token_after_reject():
    coordinator = DSSDRoundCoordinator(edge_engine=None, transport=None)
    response = VerifyRoundResponse(
        verifier_session_id="vs-1",
        seq_no=0,
        accepted_count=1,
        all_accepted=False,
        bonus_token_id=None,
        reject_index=1,
        reject_target_probs=[0.2, 0.8],
        finished=False,
        finish_reason=None,
    )

    committed = coordinator._build_committed_tokens(
        [10, 11],
        response,
        resampled_token=42,
    )

    assert committed == [10, 42]
