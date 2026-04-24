# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import builtins
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib import error, request
from urllib import response as urllib_response

import pytest


def _load_edge_server_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / "vllm"
        / "dssd"
        / "entrypoints"
        / "edge_server.py"
    )
    spec = importlib.util.spec_from_file_location(
        "test_edge_server_module",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_edge_runner_main_hard_exits_after_cleanup(monkeypatch, capsys) -> None:
    from vllm.dssd.entrypoints import edge_runner

    events: list[str] = []

    class _ExitCalled(Exception):
        pass

    def fake_cleanup() -> None:
        events.append("cleanup")

    class FakeService:
        def generate(self, **kwargs):
            events.append("generate")
            return [1, 2, 3]

    monkeypatch.setattr(
        edge_runner,
        "build_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: SimpleNamespace(
                service_factory="fake.factory",
                verifier_url="http://127.0.0.1:18011",
                req_id="req-1",
                prompt_token_ids="1,2,3",
                max_tokens=4,
                temperature=0.0,
            )
        ),
    )
    monkeypatch.setattr(
        edge_runner,
        "resolve_obj_by_qualname",
        lambda qualname: (lambda args: (FakeService(), fake_cleanup)),
    )
    monkeypatch.setattr(
        os,
        "_exit",
        lambda code: (_ for _ in ()).throw(_ExitCalled(code)),
    )

    try:
        edge_runner.main()
    except _ExitCalled as exc:
        assert exc.args == (0,)
    else:
        raise AssertionError("expected os._exit to be called")

    assert events == ["generate", "cleanup"]
    assert capsys.readouterr().out.strip().endswith(
        '{"req_id": "req-1", "output_ids": [1, 2, 3]}'
    )


def test_verifier_server_main_hard_exits_after_cleanup(monkeypatch) -> None:
    from vllm.dssd.entrypoints import verifier_server

    events: list[str] = []

    class _ExitCalled(Exception):
        pass

    def fake_cleanup() -> None:
        events.append("cleanup")

    class FakeServer:
        server_address = ("127.0.0.1", 18011)

        def serve_forever(self) -> None:
            events.append("serve_forever")

        def server_close(self) -> None:
            events.append("server_close")

    monkeypatch.setattr(
        verifier_server,
        "build_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: SimpleNamespace(
                service_factory="fake.factory",
                host="127.0.0.1",
                port=18011,
                ready_file=None,
            )
        ),
    )
    monkeypatch.setattr(
        verifier_server,
        "resolve_obj_by_qualname",
        lambda qualname: (lambda args: (object(), fake_cleanup)),
    )
    monkeypatch.setattr(
        verifier_server,
        "_build_server",
        lambda **kwargs: FakeServer(),
    )
    monkeypatch.setattr(
        os,
        "_exit",
        lambda code: (_ for _ in ()).throw(_ExitCalled(code)),
    )

    try:
        verifier_server.main()
    except _ExitCalled as exc:
        assert exc.args == (0,)
    else:
        raise AssertionError("expected os._exit to be called")

    assert events == ["serve_forever", "server_close", "cleanup"]


def test_edge_server_main_hard_exits_after_cleanup(monkeypatch,
                                                   tmp_path) -> None:
    edge_server = _load_edge_server_module()

    events: list[str] = []
    ready_file = tmp_path / "edge-ready.json"

    class _ExitCalled(Exception):
        pass

    def fake_cleanup() -> None:
        events.append("cleanup")

    class FakeServer:
        server_address = ("127.0.0.1", 18021)

        def serve_forever(self) -> None:
            events.append("serve_forever")

        def server_close(self) -> None:
            events.append("server_close")

    monkeypatch.setattr(
        edge_server,
        "build_parser",
        lambda: SimpleNamespace(
            parse_args=lambda: SimpleNamespace(
                service_factory="fake.factory",
                host="127.0.0.1",
                port=18021,
                ready_file=str(ready_file),
            )
        ),
    )
    monkeypatch.setattr(
        edge_server,
        "_resolve_obj_by_qualname",
        lambda qualname: (lambda args: (object(), fake_cleanup)),
    )
    monkeypatch.setattr(
        edge_server,
        "_build_server",
        lambda **kwargs: FakeServer(),
    )
    monkeypatch.setattr(
        os,
        "_exit",
        lambda code: (_ for _ in ()).throw(_ExitCalled(code)),
    )

    try:
        edge_server.main()
    except _ExitCalled as exc:
        assert exc.args == (0,)
    else:
        raise AssertionError("expected os._exit to be called")

    assert events == ["serve_forever", "server_close", "cleanup"]
    assert ready_file.exists()


def test_edge_server_module_loads_without_msgspec_in_raw_mode(
    monkeypatch,
) -> None:
    monkeypatch.setenv("VLLM_DSSD_TEST_EDGE_SERVER_RAW_SAMPLING_PARAMS", "1")
    monkeypatch.delitem(sys.modules, "msgspec", raising=False)
    real_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "msgspec":
            raise AssertionError("msgspec import should stay lazy in raw mode")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    edge_server = _load_edge_server_module()

    assert edge_server is not None


def test_edge_server_raw_sampling_flag_is_test_scoped(monkeypatch) -> None:
    edge_server = _load_edge_server_module()

    monkeypatch.delenv("VLLM_DSSD_TEST_EDGE_SERVER_RAW_SAMPLING_PARAMS",
                       raising=False)
    assert not edge_server._should_decode_raw_sampling_params(
        "tests.entrypoints.dssd_fake_factories.build_persistent_edge_service"
    )

    monkeypatch.setenv("VLLM_DSSD_TEST_EDGE_SERVER_RAW_SAMPLING_PARAMS", "1")
    assert edge_server._should_decode_raw_sampling_params(
        "tests.entrypoints.dssd_fake_factories.build_persistent_edge_service"
    )
    assert not edge_server._should_decode_raw_sampling_params(
        "vllm.dssd.entrypoints.runtime_factory.build_real_edge_service"
    )


def test_edge_server_reuses_http_utils_edge_generate_helper(monkeypatch) -> None:
    edge_server = _load_edge_server_module()
    from vllm.dssd.transport import http_utils

    sentinel = {"req_id": "req-1", "prompt_token_ids": [1]}
    monkeypatch.setattr(
        http_utils,
        "edge_generate_request_from_payload",
        lambda payload: sentinel,
    )

    assert edge_server._edge_generate_request_from_payload({}) is sentinel


def test_edge_server_generate_handler_round_trip(monkeypatch) -> None:
    edge_server = _load_edge_server_module()
    _install_edge_server_http_shims(edge_server, monkeypatch)
    server, thread = _start_edge_server(
        edge_server,
        edge_service=SimpleNamespace(
            generate=lambda **kwargs: _assert_generate_kwargs(kwargs) or [7, 8, 9]
        ),
    )

    try:
        response = _post_json(
            server,
            "/generate",
            {
                "req_id": "req-1",
                "prompt_token_ids": [1, 2, 3],
                "sampling_params": {"max_tokens": 4, "temperature": 0.0},
                "lora_request": None,
            },
        )
    finally:
        _stop_edge_server(server, thread)

    assert response == {
        "req_id": "req-1",
        "output_ids": [7, 8, 9],
    }


def test_edge_server_complete_handler_round_trip(monkeypatch) -> None:
    edge_server = _load_edge_server_module()
    _install_edge_server_http_shims(edge_server, monkeypatch)
    server, thread = _start_edge_server(
        edge_server,
        edge_service=SimpleNamespace(
            complete=lambda **kwargs: _assert_complete_kwargs(kwargs)
            or " completed text"
        ),
    )

    try:
        response = _post_json(
            server,
            "/complete",
            {
                "req_id": "req-text",
                "prompt": "hello",
                "sampling_params": {"max_tokens": 4, "temperature": 0.0},
                "lora_request": None,
            },
        )
    finally:
        _stop_edge_server(server, thread)

    assert response == {
        "req_id": "req-text",
        "text": " completed text",
    }


def test_edge_server_root_serves_benchmark_page(monkeypatch) -> None:
    edge_server = _load_edge_server_module()
    _install_edge_server_http_shims(edge_server, monkeypatch)
    server, thread = _start_edge_server(
        edge_server,
        edge_service=SimpleNamespace(),
    )

    try:
        with _get(server, "/") as response:
            body = response.read().decode("utf-8")
            content_type = response.headers["Content-Type"]
    finally:
        _stop_edge_server(server, thread)

    assert response.status == 200
    assert content_type == "text/html; charset=utf-8"
    assert "DSSD Edge Benchmark" in body
    assert "benchmark_complete" in body
    assert "Draft Acceptance" in body
    assert "Target-only" in body
    assert "mode: selectedMode" in body
    assert "服务端推理耗时" in body
    assert "服务端 token / s" in body
    assert "采样温度" in body
    assert "temperature: temperature" in body
    assert "网络延迟 ms" in body
    assert "带宽 Mbps" in body
    assert "network_simulation" in body
    assert "latency_ms: networkLatencyMs" in body
    assert "bandwidth_mbps: networkBandwidthMbps" in body


def test_edge_server_benchmark_complete_handler_round_trip(monkeypatch) -> None:
    edge_server = _load_edge_server_module()
    _install_edge_server_http_shims(edge_server, monkeypatch)
    perf_times = iter([10.0, 12.5])
    monkeypatch.setattr(
        edge_server,
        "time",
        SimpleNamespace(perf_counter=lambda: next(perf_times)),
        raising=False,
    )

    class FakeTokenizer:
        def __call__(self, text):
            if text == "hello world":
                return SimpleNamespace(input_ids=[11, 12])
            raise AssertionError(f"unexpected prompt: {text}")

        def decode(self, output_ids, skip_special_tokens=True):
            assert skip_special_tokens is True
            assert output_ids == [7, 8, 9]
            return "decoded output"

    server, thread = _start_edge_server(
        edge_server,
        edge_service=SimpleNamespace(
            tokenizer=FakeTokenizer(),
            generate_with_stats=lambda **kwargs: _assert_benchmark_kwargs(kwargs)
            or (
                [7, 8, 9],
                _make_fake_generation_stats(
                    total_rounds=2,
                    total_draft_tokens=4,
                    total_accepted_tokens=3,
                    all_accept_rounds=1,
                ),
            ),
        ),
    )

    try:
        response = _post_json(
            server,
            "/benchmark_complete",
            {
                "req_id": "req-bench",
                "mode": "dssd",
                "prompt": "hello world",
                "sampling_params": {"max_tokens": 4, "temperature": 0.0},
                "lora_request": None,
            },
        )
    finally:
        _stop_edge_server(server, thread)

    assert response == {
        "req_id": "req-bench",
        "mode": "dssd",
        "text": "decoded output",
        "server_inference_seconds": 2.5,
        "prompt_token_count": 2,
        "output_token_count": 3,
        "total_rounds": 2,
        "total_draft_tokens": 4,
        "total_accepted_tokens": 3,
        "draft_acceptance_rate": 0.75,
        "all_accept_round_rate": 0.5,
        "avg_accepted_len_per_round": 1.5,
    }


def test_edge_server_benchmark_complete_applies_temporary_network_simulation(
    monkeypatch,
) -> None:
    edge_server = _load_edge_server_module()
    _install_edge_server_http_shims(edge_server, monkeypatch)

    class FakeTokenizer:
        def __call__(self, text):
            assert text == "hello world"
            return SimpleNamespace(input_ids=[11, 12])

        def decode(self, output_ids, skip_special_tokens=True):
            assert skip_special_tokens is True
            assert output_ids == [7, 8, 9]
            return "decoded output"

    original_request_network = object()
    original_response_network = object()
    fake_verifier = SimpleNamespace(
        request_network=original_request_network,
        response_network=original_response_network,
    )

    def generate_with_stats(**kwargs):
        _assert_benchmark_kwargs(kwargs)
        assert fake_verifier.request_network is not original_request_network
        assert fake_verifier.response_network is not original_response_network
        assert fake_verifier.request_network.fixed_latency_ms == 25.0
        assert fake_verifier.response_network.fixed_latency_ms == 25.0
        assert fake_verifier.request_network.bandwidth_bytes_per_s == 12_500_000
        assert fake_verifier.response_network.bandwidth_bytes_per_s == 12_500_000
        return [7, 8, 9], _make_fake_generation_stats(
            total_rounds=2,
            total_draft_tokens=4,
            total_accepted_tokens=3,
            all_accept_rounds=1,
        )

    edge_service = SimpleNamespace(
        tokenizer=FakeTokenizer(),
        verifier=fake_verifier,
        generate_with_stats=generate_with_stats,
    )
    server, thread = _start_edge_server(edge_server, edge_service=edge_service)

    try:
        response = _post_json(
            server,
            "/benchmark_complete",
            {
                "req_id": "req-bench",
                "mode": "dssd",
                "prompt": "hello world",
                "sampling_params": {"max_tokens": 4, "temperature": 0.0},
                "network_simulation": {
                    "latency_ms": 25,
                    "bandwidth_mbps": 100,
                },
                "lora_request": None,
            },
        )
    finally:
        _stop_edge_server(server, thread)

    assert response["text"] == "decoded output"
    assert fake_verifier.request_network is original_request_network
    assert fake_verifier.response_network is original_response_network


def test_edge_server_benchmark_complete_rejects_invalid_network_simulation(
    monkeypatch,
) -> None:
    edge_server = _load_edge_server_module()
    _install_edge_server_http_shims(edge_server, monkeypatch)

    server, thread = _start_edge_server(
        edge_server,
        edge_service=SimpleNamespace(),
    )

    try:
        with pytest.raises(error.HTTPError) as exc_info:
            _post_json(
                server,
                "/benchmark_complete",
                {
                    "req_id": "req-bench",
                    "mode": "dssd",
                    "prompt": "hello world",
                    "sampling_params": {"max_tokens": 4, "temperature": 0.0},
                    "network_simulation": [],
                    "lora_request": None,
                },
            )
    finally:
        _stop_edge_server(server, thread)

    assert exc_info.value.code == 500
    assert json.loads(exc_info.value.read().decode("utf-8")) == {
        "error": "network_simulation must be an object",
        "error_type": "ValueError",
    }


def test_edge_server_benchmark_complete_target_only_uses_verifier_generate(
    monkeypatch,
) -> None:
    edge_server = _load_edge_server_module()
    _install_edge_server_http_shims(edge_server, monkeypatch)
    perf_times = iter([1.0, 1.75])
    monkeypatch.setattr(
        edge_server,
        "time",
        SimpleNamespace(perf_counter=lambda: next(perf_times)),
        raising=False,
    )

    class FakeTokenizer:
        def __call__(self, text):
            assert text == "hello world"
            return SimpleNamespace(input_ids=[11, 12])

        def decode(self, output_ids, skip_special_tokens=True):
            assert skip_special_tokens is True
            assert output_ids == [31, 32]
            return "target output"

    class FakeVerifier:
        def __init__(self) -> None:
            self.generate_calls = []

        def generate(self, **kwargs):
            self.generate_calls.append(kwargs)
            return [31, 32]

    fake_verifier = FakeVerifier()
    edge_service = SimpleNamespace(
        tokenizer=FakeTokenizer(),
        verifier=fake_verifier,
        generate_with_stats=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("target-only mode must not call DSSD generation")
        ),
    )
    server, thread = _start_edge_server(edge_server, edge_service=edge_service)

    try:
        response = _post_json(
            server,
            "/benchmark_complete",
            {
                "req_id": "req-target",
                "mode": "target_only",
                "prompt": "hello world",
                "sampling_params": {"max_tokens": 4, "temperature": 0.0},
                "lora_request": None,
            },
        )
    finally:
        _stop_edge_server(server, thread)

    assert fake_verifier.generate_calls == [
        {
            "req_id": "req-target",
            "prompt_token_ids": [11, 12],
            "sampling_params": {"max_tokens": 4, "temperature": 0.0},
            "lora_request": None,
        }
    ]
    assert response == {
        "req_id": "req-target",
        "mode": "target_only",
        "text": "target output",
        "server_inference_seconds": 0.75,
        "prompt_token_count": 2,
        "output_token_count": 2,
        "total_rounds": 0,
        "total_draft_tokens": 0,
        "total_accepted_tokens": 0,
        "draft_acceptance_rate": None,
        "all_accept_round_rate": None,
        "avg_accepted_len_per_round": None,
    }


def test_edge_server_generate_handler_returns_structured_500(monkeypatch) -> None:
    edge_server = _load_edge_server_module()
    _install_edge_server_http_shims(edge_server, monkeypatch)

    def raise_generation_error(**kwargs):
        _assert_generate_kwargs(kwargs)
        raise RuntimeError("edge failed")

    server, thread = _start_edge_server(
        edge_server,
        edge_service=SimpleNamespace(generate=raise_generation_error),
    )

    try:
        with pytest.raises(error.HTTPError) as exc_info:
            _post_json(
                server,
                "/generate",
                {
                    "req_id": "req-2",
                    "prompt_token_ids": [4, 5],
                    "sampling_params": {"max_tokens": 2, "temperature": 0.0},
                    "lora_request": None,
                },
            )
    finally:
        _stop_edge_server(server, thread)

    assert exc_info.value.code == 500
    assert json.loads(exc_info.value.read().decode("utf-8")) == {
        "error": "edge failed",
        "error_type": "RuntimeError",
    }


def test_edge_server_benchmark_complete_handler_returns_structured_500(
    monkeypatch,
) -> None:
    edge_server = _load_edge_server_module()
    _install_edge_server_http_shims(edge_server, monkeypatch)

    class FakeTokenizer:
        def __call__(self, text):
            assert text == "hello world"
            return SimpleNamespace(input_ids=[11, 12])

        def decode(self, output_ids, skip_special_tokens=True):
            return "unused"

    def raise_generation_error(**kwargs):
        _assert_benchmark_kwargs(kwargs)
        raise RuntimeError("benchmark failed")

    server, thread = _start_edge_server(
        edge_server,
        edge_service=SimpleNamespace(
            tokenizer=FakeTokenizer(),
            generate_with_stats=raise_generation_error,
        ),
    )

    try:
        with pytest.raises(error.HTTPError) as exc_info:
            _post_json(
                server,
                "/benchmark_complete",
                {
                    "req_id": "req-bench",
                    "prompt": "hello world",
                    "sampling_params": {"max_tokens": 4, "temperature": 0.0},
                    "lora_request": None,
                },
            )
    finally:
        _stop_edge_server(server, thread)

    assert exc_info.value.code == 500
    assert json.loads(exc_info.value.read().decode("utf-8")) == {
        "error": "benchmark failed",
        "error_type": "RuntimeError",
    }


def _assert_generate_kwargs(kwargs: dict) -> None:
    assert kwargs["req_id"].startswith("req-")
    assert kwargs["sampling_params"] == {
        "max_tokens": kwargs["sampling_params"]["max_tokens"],
        "temperature": 0.0,
    }
    assert kwargs["lora_request"] is None


def _assert_complete_kwargs(kwargs: dict) -> None:
    assert kwargs["req_id"] == "req-text"
    assert kwargs["prompt"] == "hello"
    assert kwargs["sampling_params"] == {
        "max_tokens": 4,
        "temperature": 0.0,
    }
    assert kwargs["lora_request"] is None


def _assert_benchmark_kwargs(kwargs: dict) -> None:
    assert kwargs["req_id"] == "req-bench"
    assert kwargs["prompt_token_ids"] == [11, 12]
    assert kwargs["sampling_params"] == {
        "max_tokens": 4,
        "temperature": 0.0,
    }
    assert kwargs["lora_request"] is None


def _make_fake_generation_stats(
    *,
    total_rounds: int,
    total_draft_tokens: int,
    total_accepted_tokens: int,
    all_accept_rounds: int,
):
    return SimpleNamespace(
        total_rounds=total_rounds,
        total_draft_tokens=total_draft_tokens,
        total_accepted_tokens=total_accepted_tokens,
        all_accept_rounds=all_accept_rounds,
        draft_acceptance_rate=(
            total_accepted_tokens / total_draft_tokens
            if total_draft_tokens
            else 0.0
        ),
        all_accept_round_rate=(
            all_accept_rounds / total_rounds if total_rounds else 0.0
        ),
        avg_accepted_len_per_round=(
            total_accepted_tokens / total_rounds if total_rounds else 0.0
        ),
    )


def _start_edge_server(edge_server, *, edge_service):
    import threading

    server = edge_server._build_server(
        host="127.0.0.1",
        port=0,
        edge_service=edge_service,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _stop_edge_server(server, thread) -> None:
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


def _install_edge_server_http_shims(edge_server, monkeypatch) -> None:
    monkeypatch.setattr(
        edge_server,
        "_load_json",
        lambda payload: json.loads(payload.decode("utf-8")),
    )
    monkeypatch.setattr(
        edge_server,
        "_dump_json",
        lambda payload: json.dumps(payload).encode("utf-8"),
    )
    monkeypatch.setattr(
        edge_server,
        "_edge_generate_request_from_payload",
        lambda payload: {
            "req_id": payload["req_id"],
            "prompt_token_ids": list(payload["prompt_token_ids"]),
            "sampling_params": dict(payload["sampling_params"]),
            "lora_request": payload.get("lora_request"),
        },
    )
    monkeypatch.setattr(
        edge_server,
        "_edge_generate_response_to_payload",
        lambda *, req_id, output_ids: {
            "req_id": req_id,
            "output_ids": list(output_ids),
        },
    )
    monkeypatch.setattr(
        edge_server,
        "_edge_complete_request_from_payload",
        lambda payload: {
            "req_id": payload["req_id"],
            "prompt": payload["prompt"],
            "sampling_params": dict(payload["sampling_params"]),
            "lora_request": payload.get("lora_request"),
        },
    )
    monkeypatch.setattr(
        edge_server,
        "_edge_complete_response_to_payload",
        lambda *, req_id, text: {
            "req_id": req_id,
            "text": text,
        },
    )
