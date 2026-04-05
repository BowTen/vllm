# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[2]
VLLM_DIR = ROOT / "vllm"
DSSD_SERVE_DIR = VLLM_DIR / "entrypoints" / "serve" / "dssd"


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
_install_package_stub("vllm.entrypoints", VLLM_DIR / "entrypoints")
_install_package_stub("vllm.entrypoints.serve", VLLM_DIR / "entrypoints" / "serve")
_install_package_stub("vllm.entrypoints.serve.dssd", DSSD_SERVE_DIR)

router_module = _load_module(
    "vllm.entrypoints.serve.dssd.api_router",
    DSSD_SERVE_DIR / "api_router.py",
)

attach_router = router_module.attach_router


def test_verifier_router_exposes_bind_endpoint():
    app = FastAPI()
    app.state.dssd_verifier_service = object()
    attach_router(app)

    client = TestClient(app)
    paths = {route.path for route in app.routes}

    assert "/server/dssd/bind" in paths
    response = client.post("/server/dssd/bind")
    assert response.status_code == 200
