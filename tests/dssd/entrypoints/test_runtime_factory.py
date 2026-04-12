# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace


def test_verifier_server_parser_defaults_to_real_service_factory() -> None:
    from vllm.dssd.entrypoints.verifier_server import build_parser

    args = build_parser().parse_args([])

    assert (
        args.service_factory
        == "vllm.dssd.entrypoints.runtime_factory.build_real_verifier_service"
    )


def test_edge_runner_parser_defaults_to_real_service_factory() -> None:
    from vllm.dssd.entrypoints.edge_runner import build_parser

    args = build_parser().parse_args(
        [
            "--verifier-url",
            "http://127.0.0.1:8000",
            "--req-id",
            "req-1",
            "--prompt-token-ids",
            "1,2,3",
            "--max-tokens",
            "4",
            "--eos-token-id",
            "2",
            "--gamma",
            "2",
        ]
    )

    assert (
        args.service_factory
        == "vllm.dssd.entrypoints.runtime_factory.build_real_edge_service"
    )


def test_build_real_verifier_service_wraps_runtime_and_cleanup(
    monkeypatch,
) -> None:
    from vllm.dssd.entrypoints import runtime_factory
    from vllm.dssd.service import DSSDVerifierService

    shutdown_calls: list[str] = []
    runtime = SimpleNamespace(
        worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                sampler=object(),
                num_speculative_steps=2,
            ),
            shutdown=lambda: shutdown_calls.append("shutdown"),
        ),
        vllm_config=object(),
        kv_cache_manager=None,
    )
    monkeypatch.setattr(
        runtime_factory,
        "_init_real_runtime",
        lambda args: (runtime, object()),
    )

    service, returned_cleanup = runtime_factory.build_real_verifier_service(
        SimpleNamespace()
    )

    assert isinstance(service, DSSDVerifierService)
    assert service.decode_engine.worker is runtime.worker
    returned_cleanup()
    assert shutdown_calls == ["shutdown"]


def test_build_real_edge_service_wraps_runtime_transport_and_cleanup(
    monkeypatch,
) -> None:
    from vllm.dssd.entrypoints import runtime_factory
    from vllm.dssd.service import DSSDEdgeService
    from vllm.dssd.transport import HTTPVerifierTransport

    shutdown_calls: list[str] = []
    runtime = SimpleNamespace(
        worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                sampler=object(),
            ),
            shutdown=lambda: shutdown_calls.append("shutdown"),
        ),
        vllm_config=object(),
        kv_cache_manager=None,
    )
    monkeypatch.setattr(
        runtime_factory,
        "_init_real_runtime",
        lambda args: (runtime, object()),
    )

    service, returned_cleanup = runtime_factory.build_real_edge_service(
        SimpleNamespace(
            verifier_url="http://127.0.0.1:9000",
            eos_token_id=2,
            gamma=3,
        )
    )

    assert isinstance(service, DSSDEdgeService)
    assert service.decode_engine.worker is runtime.worker
    assert isinstance(service.verifier, HTTPVerifierTransport)
    assert service.verifier.server_url == "http://127.0.0.1:9000"
    assert service.eos_token_id == 2
    assert service.gamma == 3
    returned_cleanup()
    assert shutdown_calls == ["shutdown"]
