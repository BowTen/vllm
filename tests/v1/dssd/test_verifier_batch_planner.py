# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import Mock


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
_install_package_stub("vllm.v1.dssd.edge", DSSD_DIR / "edge")
_install_package_stub("vllm.v1.dssd.worker", DSSD_DIR / "worker")
_install_package_stub("vllm.v1.sample", VLLM_DIR / "v1" / "sample")
_install_package_stub("vllm.v1.sample.ops", VLLM_DIR / "v1" / "sample" / "ops")

distributed_module = types.ModuleType("vllm.distributed")
distributed_module.get_pp_group = Mock(side_effect=AssertionError)
sys.modules["vllm.distributed"] = distributed_module

topk_topp_sampler_module = types.ModuleType("vllm.v1.sample.ops.topk_topp_sampler")
topk_topp_sampler_module.apply_top_k_top_p = lambda logits, *_args: logits
sys.modules["vllm.v1.sample.ops.topk_topp_sampler"] = topk_topp_sampler_module

sampler_module = types.ModuleType("vllm.v1.sample.sampler")


class _Sampler:
    def __init__(self, logprobs_mode="raw_logprobs") -> None:
        self.logprobs_mode = logprobs_mode

    def __call__(self, **_kwargs):
        raise NotImplementedError


sampler_module.Sampler = _Sampler
sys.modules["vllm.v1.sample.sampler"] = sampler_module

sampling_params_module = types.ModuleType("vllm.sampling_params")


class _SamplingParams:
    def __init__(
        self,
        *,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        seed=None,
        min_tokens=0,
        max_tokens=16,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.seed = seed
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

    def clone(self):
        return _SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            seed=self.seed,
            min_tokens=self.min_tokens,
            max_tokens=self.max_tokens,
        )


sampling_params_module.SamplingParams = _SamplingParams
sys.modules["vllm.sampling_params"] = sampling_params_module

_load_module("vllm.v1.dssd.edge.session", DSSD_DIR / "edge" / "session.py")
_load_module("vllm.v1.dssd.protocol", DSSD_DIR / "protocol.py")
_load_module("vllm.v1.dssd.engine.session_store", DSSD_DIR / "engine" / "session_store.py")
_load_module("vllm.v1.dssd.worker.draft_runner", DSSD_DIR / "worker" / "draft_runner.py")
_load_module("vllm.v1.dssd.worker.verifier_runner", DSSD_DIR / "worker" / "verifier_runner.py")

batch_planner_module = _load_module(
    "vllm.v1.dssd.engine.batch_planner",
    DSSD_DIR / "engine" / "batch_planner.py",
)
session_runner_module = _load_module(
    "vllm.v1.dssd.engine.session_runner",
    DSSD_DIR / "engine" / "session_runner.py",
)

VerifierRoundBatcher = batch_planner_module.VerifierRoundBatcher
VerifierRoundBatchItem = batch_planner_module.VerifierRoundBatchItem
VerifierRoundBatchKey = batch_planner_module.VerifierRoundBatchKey
DSSDSessionRunner = session_runner_module.DSSDSessionRunner


def test_batcher_groups_requests_by_typed_key_and_preserves_order():
    batcher = VerifierRoundBatcher()
    batcher.add(
        VerifierRoundBatchItem(
            verifier_session_id="a",
            batch_key=VerifierRoundBatchKey(gamma=4, sampling_signature="s1"),
            payload="ra",
        )
    )
    batcher.add(
        VerifierRoundBatchItem(
            verifier_session_id="b",
            batch_key=VerifierRoundBatchKey(gamma=4, sampling_signature="s1"),
            payload="rb",
        )
    )
    batcher.add(
        VerifierRoundBatchItem(
            verifier_session_id="c",
            batch_key=VerifierRoundBatchKey(gamma=2, sampling_signature="s1"),
            payload="rc",
        )
    )

    groups = batcher.flush()

    assert [group.batch_key for group in groups] == [
        VerifierRoundBatchKey(gamma=4, sampling_signature="s1"),
        VerifierRoundBatchKey(gamma=2, sampling_signature="s1"),
    ]
    assert [[item.verifier_session_id for item in group.items] for group in groups] == [
        ["a", "b"],
        ["c"],
    ]


def test_batcher_flush_clears_pending_items():
    batcher = VerifierRoundBatcher()
    batcher.add(
        VerifierRoundBatchItem(
            verifier_session_id="only",
            batch_key=VerifierRoundBatchKey(gamma=1, sampling_signature="sig"),
            payload=None,
        )
    )

    first_flush = batcher.flush()
    second_flush = batcher.flush()

    assert len(first_flush) == 1
    assert second_flush == []


def test_session_runner_exposes_verifier_batcher():
    runner = DSSDSessionRunner()

    assert isinstance(runner.verifier_batcher, VerifierRoundBatcher)
