# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace


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
