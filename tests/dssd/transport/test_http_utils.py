# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import builtins
import importlib.util
import sys
from pathlib import Path

from vllm.sampling_params import SamplingParams


def _load_http_utils_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "vllm"
        / "dssd"
        / "transport"
        / "http_utils.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_http_utils_module",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_http_utils_module_load_is_lazy_for_edge_server_imports(
    monkeypatch,
) -> None:
    blocked_imports = {
        "msgspec",
        "torch",
        "vllm.lora.request",
        "vllm.sampling_params",
    }
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in blocked_imports:
            raise AssertionError(f"unexpected eager import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    for module_name in blocked_imports:
        monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded_import)

    module = _load_http_utils_module()

    assert module is not None


def test_edge_generate_request_round_trips_sampling_params() -> None:
    from vllm.dssd.transport.http_utils import (
        edge_generate_request_from_payload,
        edge_generate_request_to_payload,
    )

    payload = edge_generate_request_to_payload(
        req_id="req-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=4, temperature=0.3),
        lora_request=None,
    )

    decoded = edge_generate_request_from_payload(payload)

    assert decoded["req_id"] == "req-1"
    assert decoded["prompt_token_ids"] == [1, 2, 3]
    assert decoded["sampling_params"].max_tokens == 4
    assert decoded["sampling_params"].temperature == 0.3
    assert decoded["lora_request"] is None


def test_edge_generate_response_round_trips_output_ids() -> None:
    from vllm.dssd.transport.http_utils import (
        edge_generate_response_from_payload,
        edge_generate_response_to_payload,
    )

    payload = edge_generate_response_to_payload(
        req_id="req-7",
        output_ids=[17, 19, 20, 21],
    )

    decoded = edge_generate_response_from_payload(payload)

    assert decoded == {
        "req_id": "req-7",
        "output_ids": [17, 19, 20, 21],
    }


def test_edge_complete_request_and_response_round_trip() -> None:
    from vllm.dssd.transport.http_utils import (
        edge_complete_request_from_payload,
        edge_complete_request_to_payload,
        edge_complete_response_from_payload,
        edge_complete_response_to_payload,
    )

    request_payload = edge_complete_request_to_payload(
        req_id="req-text",
        prompt="Once upon a time",
        sampling_params=SamplingParams(max_tokens=8, temperature=0.2),
        lora_request=None,
    )
    request = edge_complete_request_from_payload(request_payload)

    assert request["req_id"] == "req-text"
    assert request["prompt"] == "Once upon a time"
    assert request["sampling_params"].max_tokens == 8
    assert request["sampling_params"].temperature == 0.2
    assert request["lora_request"] is None

    response_payload = edge_complete_response_to_payload(
        req_id="req-text",
        text=" and then",
    )

    assert edge_complete_response_from_payload(response_payload) == {
        "req_id": "req-text",
        "text": " and then",
    }
