# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[2]
DSSD_CONFIG_PATH = ROOT / "vllm" / "config" / "dssd.py"


def _config_decorator(cls=None, **_kwargs):
    def decorate(target_cls):
        return dataclass(target_cls)

    if cls is None:
        return decorate
    return decorate(cls)


def _load_dssd_module(monkeypatch: pytest.MonkeyPatch):
    vllm_pkg = types.ModuleType("vllm")
    vllm_pkg.__path__ = []  # type: ignore[attr-defined]
    config_pkg = types.ModuleType("vllm.config")
    config_pkg.__path__ = []  # type: ignore[attr-defined]
    utils_mod = types.ModuleType("vllm.config.utils")
    utils_mod.config = _config_decorator

    monkeypatch.setitem(sys.modules, "vllm", vllm_pkg)
    monkeypatch.setitem(sys.modules, "vllm.config", config_pkg)
    monkeypatch.setitem(sys.modules, "vllm.config.utils", utils_mod)

    spec = importlib.util.spec_from_file_location(
        "vllm.config.dssd",
        DSSD_CONFIG_PATH,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "vllm.config.dssd", module)
    spec.loader.exec_module(module)
    return module


def test_dssd_config_accepts_valid_edge_config(monkeypatch: pytest.MonkeyPatch):
    dssd_mod = _load_dssd_module(monkeypatch)

    cfg = dssd_mod.DSSDConfig(
        enabled=True,
        role="edge",
        gamma=4,
        verifier_url="http://127.0.0.1:9000",
    ).validate()

    assert cfg.enabled is True
    assert cfg.role == "edge"
    assert cfg.gamma == 4
    assert cfg.verifier_url == "http://127.0.0.1:9000"
    assert isinstance(cfg.network_simulation, dssd_mod.DSSDNetworkSimulationConfig)


def test_dssd_config_accepts_baseline_mode_without_verifier_url(
    monkeypatch: pytest.MonkeyPatch,
):
    dssd_mod = _load_dssd_module(monkeypatch)

    cfg = dssd_mod.DSSDConfig(
        enabled=True,
        role="edge",
        gamma=4,
        experiment_mode="baseline",
    ).validate()

    assert cfg.experiment_mode == "baseline"
    assert cfg.verifier_url is None


def test_dssd_config_accepts_experiment_result_path(
    monkeypatch: pytest.MonkeyPatch,
):
    dssd_mod = _load_dssd_module(monkeypatch)

    cfg = dssd_mod.DSSDConfig(
        enabled=True,
        role="edge",
        gamma=4,
        verifier_url="http://127.0.0.1:9000",
        experiment_result_path="/tmp/dssd-results.jsonl",
    ).validate()

    assert cfg.experiment_result_path == "/tmp/dssd-results.jsonl"


def test_dssd_config_disabled_short_circuits_validation(
    monkeypatch: pytest.MonkeyPatch,
):
    dssd_mod = _load_dssd_module(monkeypatch)

    cfg = dssd_mod.DSSDConfig(enabled=False, role=None, gamma=0, verifier_url=None)

    assert cfg.validate() is cfg


def test_dssd_config_rejects_invalid_role(monkeypatch: pytest.MonkeyPatch):
    dssd_mod = _load_dssd_module(monkeypatch)

    with pytest.raises(ValueError, match="role"):
        dssd_mod.DSSDConfig(enabled=True, role="draft", gamma=4).validate()


def test_dssd_config_rejects_gamma_below_one(monkeypatch: pytest.MonkeyPatch):
    dssd_mod = _load_dssd_module(monkeypatch)

    with pytest.raises(ValueError, match="gamma"):
        dssd_mod.DSSDConfig(enabled=True, role="verifier", gamma=0).validate()


def test_dssd_config_rejects_invalid_experiment_mode(
    monkeypatch: pytest.MonkeyPatch,
):
    dssd_mod = _load_dssd_module(monkeypatch)

    with pytest.raises(ValueError, match="experiment_mode"):
        dssd_mod.DSSDConfig(
            enabled=True,
            role="edge",
            gamma=4,
            experiment_mode="mystery",
        ).validate()


def test_dssd_config_rejects_edge_without_verifier_url(
    monkeypatch: pytest.MonkeyPatch,
):
    dssd_mod = _load_dssd_module(monkeypatch)

    with pytest.raises(ValueError, match="verifier_url"):
        dssd_mod.DSSDConfig(enabled=True, role="edge", gamma=4).validate()
