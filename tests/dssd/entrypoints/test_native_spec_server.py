# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from urllib import error, request
from urllib import response as urllib_response

import pytest


def _load_native_spec_server_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "vllm"
        / "dssd"
        / "entrypoints"
        / "native_spec_server.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_native_spec_server_module",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_spec_server_main_hard_exits_after_cleanup(
    monkeypatch,
    tmp_path,
) -> None:
    native_spec_server = _load_native_spec_server_module()
    ready_file = tmp_path / "native-ready.json"
    events: list[str] = []

    class _ExitCalled(Exception):
        pass

    def fake_cleanup() -> None:
        events.append("cleanup")

    class FakeServer:
        server_address = ("127.0.0.1", 19090)

        def serve_forever(self) -> None:
            events.append("serve_forever")

        def server_close(self) -> None:
            events.append("server_close")

    monkeypatch.setattr(
        native_spec_server,
        "build_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: SimpleNamespace(
                service_factory="fake.factory",
                host="127.0.0.1",
                port=19090,
                ready_file=str(ready_file),
            )
        ),
    )
    monkeypatch.setattr(
        native_spec_server,
        "_resolve_obj_by_qualname",
        lambda qualname: (lambda args: (object(), fake_cleanup)),
    )
    monkeypatch.setattr(
        native_spec_server,
        "_build_server",
        lambda **kwargs: FakeServer(),
    )
    monkeypatch.setattr(
        os,
        "_exit",
        lambda code: (_ for _ in ()).throw(_ExitCalled(code)),
    )

    try:
        native_spec_server.main()
    except _ExitCalled as exc:
        assert exc.args == (0,)
    else:
        raise AssertionError("expected os._exit to be called")

    assert events == ["serve_forever", "server_close", "cleanup"]
    assert json.loads(ready_file.read_text()) == {
        "server_url": "http://127.0.0.1:19090"
    }


def test_native_spec_server_root_serves_simplified_page() -> None:
    native_spec_server = _load_native_spec_server_module()
    server, thread = _start_server(
        native_spec_server,
        service=SimpleNamespace(),
    )

    try:
        with _get(server, "/") as response:
            body = response.read().decode("utf-8")
            content_type = response.headers["Content-Type"]
    finally:
        _stop_server(server, thread)

    assert response.status == 200
    assert content_type == "text/html; charset=utf-8"
    assert "Native vLLM Speculative Benchmark" in body
    assert "benchmark_complete" in body
    assert "num_speculative_tokens" in body
    assert "采样温度" in body
    assert "随机 seed" in body
    assert "Target-only" not in body
    assert "network_simulation" not in body
    assert "网络延迟" not in body


def test_native_spec_server_benchmark_complete_handler_round_trip() -> None:
    native_spec_server = _load_native_spec_server_module()

    def complete(**kwargs):
        assert kwargs == {
            "req_id": "req-native",
            "prompt": "hello world",
            "sampling_params": {
                "max_tokens": 8,
                "temperature": 0.9,
                "ignore_eos": True,
            },
        }
        return {
            "req_id": "req-native",
            "text": "native output",
            "server_inference_seconds": 0.5,
            "prompt_token_count": 2,
            "output_token_count": 8,
            "tokens_per_second": 16.0,
            "num_drafts": 3.0,
            "num_draft_tokens": 12.0,
            "num_accepted_tokens": 9.0,
            "draft_acceptance_rate": 0.75,
            "acceptance_length": 4.0,
            "num_speculative_tokens": 4,
        }

    server, thread = _start_server(
        native_spec_server,
        service=SimpleNamespace(complete=complete),
    )

    try:
        response = _post_json(
            server,
            "/benchmark_complete",
            {
                "req_id": "req-native",
                "prompt": "hello world",
                "sampling_params": {
                    "max_tokens": 8,
                    "temperature": 0.9,
                    "ignore_eos": True,
                },
            },
        )
    finally:
        _stop_server(server, thread)

    assert response == {
        "req_id": "req-native",
        "text": "native output",
        "server_inference_seconds": 0.5,
        "prompt_token_count": 2,
        "output_token_count": 8,
        "tokens_per_second": 16.0,
        "num_drafts": 3.0,
        "num_draft_tokens": 12.0,
        "num_accepted_tokens": 9.0,
        "draft_acceptance_rate": 0.75,
        "acceptance_length": 4.0,
        "num_speculative_tokens": 4,
    }


def test_native_spec_service_accepts_null_seed_for_random_sampling() -> None:
    native_spec_server = _load_native_spec_server_module()
    seen_seeds: list[int | None] = []

    class FakeTokenizer:

        def encode(self, prompt, *, add_special_tokens):
            return [1, 2]

        def decode(self, token_ids, *, skip_special_tokens):
            return "decoded"

    class FakeLLM:

        def __init__(self):
            self.metrics_calls = 0

        def get_metrics(self):
            self.metrics_calls += 1
            offset = 0.0 if self.metrics_calls == 1 else 1.0
            return [
                SimpleNamespace(
                    name="vllm:spec_decode_num_drafts",
                    value=offset,
                ),
                SimpleNamespace(
                    name="vllm:spec_decode_num_draft_tokens",
                    value=offset,
                ),
                SimpleNamespace(
                    name="vllm:spec_decode_num_accepted_tokens",
                    value=offset,
                ),
            ]

        def generate(self, prompts, *, sampling_params, use_tqdm):
            seen_seeds.append(sampling_params.seed)
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(token_ids=[3, 4], text="sampled")
                    ]
                )
            ]

    service = native_spec_server.NativeSpecBenchmarkService(
        llm=FakeLLM(),
        tokenizer=FakeTokenizer(),
        default_num_speculative_tokens=4,
        seed=0,
    )

    response = service.complete(
        req_id="req-random-seed",
        prompt="hello",
        sampling_params={
            "max_tokens": 2,
            "temperature": 0.9,
            "ignore_eos": True,
            "seed": None,
        },
    )

    assert seen_seeds == [None]
    assert response["text"] == "sampled"


def test_native_spec_server_returns_structured_500() -> None:
    native_spec_server = _load_native_spec_server_module()

    def complete(**kwargs):
        raise RuntimeError("native spec failed")

    server, thread = _start_server(
        native_spec_server,
        service=SimpleNamespace(complete=complete),
    )

    try:
        with pytest.raises(error.HTTPError) as exc_info:
            _post_json(
                server,
                "/benchmark_complete",
                {
                    "req_id": "req-native",
                    "prompt": "hello world",
                    "sampling_params": {"max_tokens": 8, "temperature": 0.9},
                },
            )
    finally:
        _stop_server(server, thread)

    assert exc_info.value.code == 500
    assert json.loads(exc_info.value.read().decode("utf-8")) == {
        "error": "native spec failed",
        "error_type": "RuntimeError",
    }


def _start_server(native_spec_server, *, service):
    import threading

    server = native_spec_server._build_server(
        host="127.0.0.1",
        port=0,
        service=service,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop_server(server, thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _post_json(server, path: str, payload: dict) -> dict:
    http_request = request.Request(
        url=f"http://127.0.0.1:{server.server_address[1]}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(http_request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def _get(server, path: str) -> urllib_response.addinfourl:
    http_request = request.Request(
        url=f"http://127.0.0.1:{server.server_address[1]}{path}",
        method="GET",
    )
    return request.urlopen(http_request, timeout=5)
