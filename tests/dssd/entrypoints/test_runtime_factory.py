# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import pytest


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
    sentinel_block_hasher = object()
    monkeypatch.setattr(
        runtime_factory,
        "_make_request_block_hasher",
        lambda vllm_config: sentinel_block_hasher,
    )

    service, returned_cleanup = runtime_factory.build_real_verifier_service(
        SimpleNamespace()
    )

    assert isinstance(service, DSSDVerifierService)
    assert service.decode_engine.worker is runtime.worker
    assert service.decode_engine.scheduler.request_block_hasher is sentinel_block_hasher
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
    sentinel_block_hasher = object()
    monkeypatch.setattr(
        runtime_factory,
        "_make_request_block_hasher",
        lambda vllm_config: sentinel_block_hasher,
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
    assert service.decode_engine.scheduler.request_block_hasher is sentinel_block_hasher
    assert isinstance(service.verifier, HTTPVerifierTransport)
    assert service.verifier.server_url == "http://127.0.0.1:9000"
    assert service.eos_token_id == 2
    assert service.gamma == 3
    returned_cleanup()
    assert shutdown_calls == ["shutdown"]


def test_init_real_runtime_disables_prefix_caching(monkeypatch) -> None:
    from vllm.dssd.entrypoints import runtime_factory

    captured_kwargs = {}

    class FakeEngineArgs:
        def __init__(self, **kwargs) -> None:
            captured_kwargs.update(kwargs)

        def create_engine_config(self):
            raise RuntimeError("stop after capture")

    monkeypatch.setattr(runtime_factory, "EngineArgs", FakeEngineArgs)

    with pytest.raises(RuntimeError, match="stop after capture"):
        runtime_factory._init_real_runtime(
            SimpleNamespace(
                model="/tmp/model",
                enforce_eager=True,
                async_scheduling=False,
                max_model_len=64,
                gpu_memory_utilization=0.01,
                kv_cache_memory_bytes=128 * 1024 * 1024,
                max_num_batched_tokens=64,
                max_num_seqs=2,
            )
        )

    assert captured_kwargs["enable_prefix_caching"] is False
