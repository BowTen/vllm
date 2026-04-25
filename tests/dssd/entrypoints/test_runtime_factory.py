# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import contextlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

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


def test_verifier_server_parser_defaults_to_real_service_factory() -> None:
    from vllm.dssd.entrypoints.verifier_server import build_parser

    args = build_parser().parse_args([])

    assert (
        args.service_factory
        == "vllm.dssd.entrypoints.runtime_factory.build_real_verifier_service"
    )
    assert args.model_runner_version == "v1"
    assert args.gamma == 0
    assert args.kv_cache_memory_bytes is None


def test_verifier_server_parser_accepts_model_runner_version() -> None:
    from vllm.dssd.entrypoints.verifier_server import build_parser

    args = build_parser().parse_args(
        [
            "--model-runner-version",
            "v1",
        ]
    )

    assert args.model_runner_version == "v1"


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
    assert args.kv_cache_memory_bytes is None


def test_edge_runner_parser_accepts_model_runner_version() -> None:
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
            "--model-runner-version",
            "v1",
        ]
    )

    assert args.model_runner_version == "v1"


def test_edge_server_parser_defaults_to_real_service_factory(monkeypatch) -> None:
    edge_server = _load_edge_server_module()

    monkeypatch.setattr(edge_server, "_add_runtime_args", lambda parser: None)

    args = edge_server.build_parser().parse_args(
        [
            "--verifier-url",
            "http://127.0.0.1:8000",
            "--gamma",
            "2",
        ]
    )

    assert (
        args.service_factory
        == "vllm.dssd.entrypoints.runtime_factory.build_real_edge_service"
    )
    assert args.host == "127.0.0.1"
    assert args.port == 6006
    assert args.eos_token_id is None
    assert args.model_runner_version == "v1"


def test_edge_server_parser_accepts_model_runner_version(monkeypatch) -> None:
    edge_server = _load_edge_server_module()

    monkeypatch.setattr(edge_server, "_add_runtime_args", lambda parser: None)

    args = edge_server.build_parser().parse_args(
        [
            "--verifier-url",
            "http://127.0.0.1:8000",
            "--eos-token-id",
            "2",
            "--gamma",
            "2",
            "--model-runner-version",
            "v1",
        ]
    )

    assert args.model_runner_version == "v1"


def test_edge_server_reuses_shared_runtime_arg_helper() -> None:
    edge_server = _load_edge_server_module()
    runtime_args_path = Path(
        edge_server._add_runtime_args.__code__.co_filename,
    )

    assert runtime_args_path.name == "runtime_args.py"


def test_build_real_verifier_service_wraps_runtime_and_cleanup(
    monkeypatch,
) -> None:
    from vllm.dssd.entrypoints import runtime_factory
    from vllm.dssd.service import DSSDVerifierService

    shutdown_calls: list[str] = []
    cleanup_calls: list[str] = []
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

    def fake_cleanup() -> None:
        cleanup_calls.append("cleanup")
        runtime.worker.shutdown()

    monkeypatch.setattr(
        runtime_factory,
        "_init_real_runtime",
        lambda args, **kwargs: (runtime, fake_cleanup),
    )
    sentinel_block_hasher = object()
    monkeypatch.setattr(
        runtime_factory,
        "_make_request_block_hasher",
        lambda vllm_config: sentinel_block_hasher,
    )

    service, returned_cleanup = runtime_factory.build_real_verifier_service(
        SimpleNamespace(model_runner_version="v2", gamma=2)
    )

    assert isinstance(service, DSSDVerifierService)
    assert service.decode_engine.worker is runtime.worker
    assert service.decode_engine.scheduler.request_block_hasher is sentinel_block_hasher
    returned_cleanup()
    assert shutdown_calls == ["shutdown"]
    assert cleanup_calls == ["cleanup"]


def test_build_real_verifier_service_uses_requested_gamma_for_v2(
    monkeypatch,
) -> None:
    from vllm.dssd.entrypoints import runtime_factory

    runtime = SimpleNamespace(
        worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                sampler=object(),
                num_speculative_steps=0,
            ),
            shutdown=lambda: None,
        ),
        vllm_config=object(),
        kv_cache_manager=None,
    )
    captured = {}

    monkeypatch.setattr(
        runtime_factory,
        "_init_real_runtime",
        lambda args, **kwargs: (runtime, lambda: None),
    )
    monkeypatch.setattr(
        runtime_factory,
        "_make_request_block_hasher",
        lambda _config: None,
    )

    class FakeVerifierStateBridge:
        pass

    class FakeDSSDVerifierSampler:
        def __init__(self, sampler, num_speculative_steps) -> None:
            captured["sampler"] = sampler
            captured["num_speculative_steps"] = num_speculative_steps

    class FakeVerifierDecodeEngine:
        def __init__(self, **kwargs) -> None:
            captured["kwargs"] = kwargs

    monkeypatch.setattr(
        runtime_factory,
        "VerifierStateBridge",
        FakeVerifierStateBridge,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_factory,
        "DSSDVerifierSampler",
        FakeDSSDVerifierSampler,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_factory,
        "VerifierDecodeEngine",
        FakeVerifierDecodeEngine,
        raising=False,
    )

    service, _cleanup = runtime_factory.build_real_verifier_service(
        SimpleNamespace(model_runner_version="v2", gamma=3)
    )

    assert service.decode_engine.__class__ is FakeVerifierDecodeEngine
    assert captured["num_speculative_steps"] == 3


def test_build_real_verifier_service_v1_sets_speculative_config_from_gamma(
    monkeypatch,
) -> None:
    from vllm.dssd.entrypoints import runtime_factory

    runtime = SimpleNamespace(
        worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                sampler=object(),
                num_spec_tokens=2,
            ),
            shutdown=lambda: None,
        ),
        vllm_config=object(),
        kv_cache_manager=None,
    )
    captured = {}

    def fake_init_real_runtime(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return runtime, lambda: None

    monkeypatch.setattr(
        runtime_factory,
        "_init_real_runtime",
        fake_init_real_runtime,
    )
    monkeypatch.setattr(
        runtime_factory,
        "_make_request_block_hasher",
        lambda _config: None,
    )

    runtime_factory.build_real_verifier_service(
        SimpleNamespace(model_runner_version="v1", gamma=2)
    )

    assert captured["kwargs"]["use_v2_model_runner"] is False
    assert captured["args"].speculative_config == {
        "method": "ngram",
        "num_speculative_tokens": 2,
        "prompt_lookup_min": 1,
        "prompt_lookup_max": 1,
    }


def test_build_real_verifier_service_selects_v1_backend(monkeypatch) -> None:
    from vllm.dssd.entrypoints import runtime_factory

    runtime = SimpleNamespace(
        worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                sampler=object(),
                num_spec_tokens=2,
            ),
            shutdown=lambda: None,
        ),
        vllm_config=object(),
        kv_cache_manager=None,
    )
    captured = {}
    init_kwargs = {}

    monkeypatch.setattr(
        runtime_factory,
        "_init_real_runtime",
        lambda args, **kwargs: (
            init_kwargs.update(kwargs) or (runtime, lambda: None)
        ),
    )
    monkeypatch.setattr(
        runtime_factory,
        "_make_request_block_hasher",
        lambda _config: None,
    )

    class FakeVerifierStateBridgeV1:
        pass

    class FakeDSSDVerifierSamplerV1:
        def __init__(self, sampler) -> None:
            captured["sampler"] = sampler

    class FakeVerifierDecodeEngineV1:
        def __init__(self, **kwargs) -> None:
            captured["kwargs"] = kwargs

    monkeypatch.setattr(
        runtime_factory,
        "VerifierStateBridgeV1",
        FakeVerifierStateBridgeV1,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_factory,
        "DSSDVerifierSamplerV1",
        FakeDSSDVerifierSamplerV1,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_factory,
        "VerifierDecodeEngineV1",
        FakeVerifierDecodeEngineV1,
        raising=False,
    )

    service, _cleanup = runtime_factory.build_real_verifier_service(
        SimpleNamespace(model_runner_version="v1")
    )

    assert service.decode_engine.__class__ is FakeVerifierDecodeEngineV1
    assert init_kwargs["use_v2_model_runner"] is False
    assert isinstance(captured["kwargs"]["state_bridge"], FakeVerifierStateBridgeV1)


def test_build_real_verifier_service_rejects_async_v1(monkeypatch) -> None:
    from vllm.dssd.entrypoints import runtime_factory

    init_calls: list[str] = []
    monkeypatch.setattr(
        runtime_factory,
        "_init_real_runtime",
        lambda *args, **kwargs: init_calls.append("called"),
    )

    with pytest.raises(ValueError, match="async_scheduling=False"):
        runtime_factory.build_real_verifier_service(
            SimpleNamespace(
                model_runner_version="v1",
                async_scheduling=True,
            )
        )

    assert init_calls == []


def test_build_real_verifier_service_cleans_up_when_v1_engine_init_fails(
    monkeypatch,
) -> None:
    from vllm.dssd.entrypoints import runtime_factory

    cleanup_calls: list[str] = []
    shutdown_calls: list[str] = []
    runtime = SimpleNamespace(
        worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                sampler=object(),
                num_spec_tokens=2,
            ),
            shutdown=lambda: shutdown_calls.append("shutdown"),
        ),
        vllm_config=object(),
        kv_cache_manager=None,
    )

    def fake_cleanup() -> None:
        cleanup_calls.append("cleanup")
        runtime.worker.shutdown()

    monkeypatch.setattr(
        runtime_factory,
        "_init_real_runtime",
        lambda args, **kwargs: (runtime, fake_cleanup),
    )
    monkeypatch.setattr(
        runtime_factory,
        "_make_request_block_hasher",
        lambda _config: object(),
    )

    class FakeVerifierStateBridgeV1:
        pass

    class FakeDSSDVerifierSamplerV1:
        def __init__(self, sampler) -> None:
            del sampler

    class FailingVerifierDecodeEngineV1:
        def __init__(self, **kwargs) -> None:
            del kwargs
            raise RuntimeError("engine init failed")

    monkeypatch.setattr(
        runtime_factory,
        "VerifierStateBridgeV1",
        FakeVerifierStateBridgeV1,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_factory,
        "DSSDVerifierSamplerV1",
        FakeDSSDVerifierSamplerV1,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_factory,
        "VerifierDecodeEngineV1",
        FailingVerifierDecodeEngineV1,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="engine init failed"):
        runtime_factory.build_real_verifier_service(
            SimpleNamespace(
                model_runner_version="v1",
                async_scheduling=False,
            )
        )

    assert cleanup_calls == ["cleanup"]
    assert shutdown_calls == ["shutdown"]


def test_build_real_edge_service_wraps_runtime_transport_and_cleanup(
    monkeypatch,
) -> None:
    from vllm.dssd.entrypoints import runtime_factory
    from vllm.dssd.edge import DSSDEdgeDraftSampler, EdgeStateBridge
    from vllm.dssd.service import DSSDEdgeService
    from vllm.dssd.transport import HTTPVerifierTransport

    shutdown_calls: list[str] = []
    cleanup_calls: list[str] = []
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

    def fake_cleanup() -> None:
        cleanup_calls.append("cleanup")
        runtime.worker.shutdown()

    monkeypatch.setattr(
        runtime_factory,
        "_init_real_runtime",
        lambda args, **kwargs: (runtime, fake_cleanup),
    )
    sentinel_block_hasher = object()
    sentinel_tokenizer = object()
    monkeypatch.setattr(
        runtime_factory,
        "_make_request_block_hasher",
        lambda vllm_config: sentinel_block_hasher,
    )
    monkeypatch.setattr(
        runtime_factory,
        "_get_tokenizer",
        lambda args, vllm_config: sentinel_tokenizer,
    )

    service, returned_cleanup = runtime_factory.build_real_edge_service(
        SimpleNamespace(
            model="fake-model",
            verifier_url="http://127.0.0.1:9000",
            eos_token_id=2,
            gamma=3,
            model_runner_version="v2",
        )
    )

    assert isinstance(service, DSSDEdgeService)
    assert service.decode_engine.worker is runtime.worker
    assert service.decode_engine.scheduler.request_block_hasher is sentinel_block_hasher
    assert isinstance(service.verifier, HTTPVerifierTransport)
    assert service.verifier.server_url == "http://127.0.0.1:9000"
    assert isinstance(service.decode_engine.state_bridge, EdgeStateBridge)
    assert isinstance(service.decode_engine.draft_sampler, DSSDEdgeDraftSampler)
    assert service.tokenizer is sentinel_tokenizer
    assert service.eos_token_id == 2
    assert service.gamma == 3
    returned_cleanup()
    assert shutdown_calls == ["shutdown"]
    assert cleanup_calls == ["cleanup"]


def test_build_real_edge_service_passes_network_simulation_to_transport(
    monkeypatch,
) -> None:
    from vllm.dssd.entrypoints import runtime_factory

    runtime = SimpleNamespace(
        worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                sampler=object(),
            ),
            shutdown=lambda: None,
        ),
        vllm_config=object(),
        kv_cache_manager=None,
    )
    request_network = object()
    response_network = object()

    monkeypatch.setattr(
        runtime_factory,
        "_init_real_runtime",
        lambda args, **kwargs: (runtime, lambda: None),
    )
    monkeypatch.setattr(
        runtime_factory,
        "_make_request_block_hasher",
        lambda vllm_config: object(),
    )
    monkeypatch.setattr(
        runtime_factory,
        "_get_tokenizer",
        lambda args, vllm_config: object(),
    )

    service, _returned_cleanup = runtime_factory.build_real_edge_service(
        SimpleNamespace(
            verifier_url="http://127.0.0.1:9000",
            eos_token_id=2,
            gamma=3,
            model_runner_version="v2",
            request_network=request_network,
            response_network=response_network,
        )
    )

    assert service.verifier.request_network is request_network
    assert service.verifier.response_network is response_network


def test_build_real_edge_service_infers_eos_token_id_from_tokenizer(
    monkeypatch,
) -> None:
    from vllm.dssd.entrypoints import runtime_factory

    runtime = SimpleNamespace(
        worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                sampler=object(),
            ),
            shutdown=lambda: None,
        ),
        vllm_config=object(),
        kv_cache_manager=None,
    )
    tokenizer = SimpleNamespace(eos_token_id=151645)

    monkeypatch.setattr(
        runtime_factory,
        "_init_real_runtime",
        lambda args, **kwargs: (runtime, lambda: None),
    )
    monkeypatch.setattr(
        runtime_factory,
        "_make_request_block_hasher",
        lambda vllm_config: object(),
    )
    monkeypatch.setattr(
        runtime_factory,
        "_get_tokenizer",
        lambda args, vllm_config: tokenizer,
    )

    service, _returned_cleanup = runtime_factory.build_real_edge_service(
        SimpleNamespace(
            verifier_url="http://127.0.0.1:9000",
            eos_token_id=None,
            gamma=3,
            model_runner_version="v2",
        )
    )

    assert service.tokenizer is tokenizer
    assert service.eos_token_id == 151645


def test_build_real_edge_service_requires_eos_when_tokenizer_has_none(
    monkeypatch,
) -> None:
    from vllm.dssd.entrypoints import runtime_factory

    runtime = SimpleNamespace(
        worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                sampler=object(),
            ),
            shutdown=lambda: None,
        ),
        vllm_config=object(),
        kv_cache_manager=None,
    )

    monkeypatch.setattr(
        runtime_factory,
        "_init_real_runtime",
        lambda args, **kwargs: (runtime, lambda: None),
    )
    monkeypatch.setattr(
        runtime_factory,
        "_make_request_block_hasher",
        lambda vllm_config: object(),
    )
    monkeypatch.setattr(
        runtime_factory,
        "_get_tokenizer",
        lambda args, vllm_config: SimpleNamespace(eos_token_id=None),
    )

    with pytest.raises(RuntimeError, match="could not infer eos_token_id"):
        runtime_factory.build_real_edge_service(
            SimpleNamespace(
                verifier_url="http://127.0.0.1:9000",
                eos_token_id=None,
                gamma=3,
                model_runner_version="v2",
            )
        )


def test_build_real_edge_service_selects_v1_backend(monkeypatch) -> None:
    from vllm.dssd.entrypoints import runtime_factory
    from vllm.dssd.service import DSSDEdgeService

    shutdown_calls: list[str] = []
    cleanup_calls: list[str] = []
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

    def fake_cleanup() -> None:
        cleanup_calls.append("cleanup")
        runtime.worker.shutdown()

    monkeypatch.setattr(
        runtime_factory,
        "_init_real_runtime",
        lambda args, **kwargs: (runtime, fake_cleanup),
    )
    sentinel_block_hasher = object()
    monkeypatch.setattr(
        runtime_factory,
        "_make_request_block_hasher",
        lambda vllm_config: sentinel_block_hasher,
    )
    monkeypatch.setattr(
        runtime_factory,
        "_get_tokenizer",
        lambda args, vllm_config: object(),
    )

    created = {}

    class FakeEdgeStateBridgeV1:
        pass

    class FakeDSSDEdgeDraftSamplerV1:
        def __init__(self, sampler) -> None:
            created["draft_sampler_input"] = sampler

    class FakeEdgeDecodeEngineV1:
        def __init__(self, **kwargs) -> None:
            created["kwargs"] = kwargs

    monkeypatch.setattr(
        runtime_factory,
        "EdgeDecodeEngineV1",
        FakeEdgeDecodeEngineV1,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_factory,
        "EdgeStateBridgeV1",
        FakeEdgeStateBridgeV1,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_factory,
        "DSSDEdgeDraftSamplerV1",
        FakeDSSDEdgeDraftSamplerV1,
        raising=False,
    )

    service, returned_cleanup = runtime_factory.build_real_edge_service(
        SimpleNamespace(
            verifier_url="http://127.0.0.1:9000",
            eos_token_id=2,
            gamma=3,
            model_runner_version="v1",
            async_scheduling=False,
        )
    )

    assert isinstance(service, DSSDEdgeService)
    assert isinstance(service.decode_engine, FakeEdgeDecodeEngineV1)
    assert isinstance(created["kwargs"]["state_bridge"], FakeEdgeStateBridgeV1)
    assert isinstance(
        created["kwargs"]["draft_sampler"], FakeDSSDEdgeDraftSamplerV1
    )
    assert created["kwargs"]["scheduler"].request_block_hasher is sentinel_block_hasher
    assert service.verifier.server_url == "http://127.0.0.1:9000"
    returned_cleanup()
    assert shutdown_calls == ["shutdown"]
    assert cleanup_calls == ["cleanup"]


def test_build_real_edge_service_rejects_async_v1(monkeypatch) -> None:
    from vllm.dssd.entrypoints import runtime_factory

    runtime = SimpleNamespace(
        worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                sampler=object(),
            ),
            shutdown=lambda: None,
        ),
        vllm_config=object(),
        kv_cache_manager=None,
    )
    monkeypatch.setattr(
        runtime_factory,
        "_init_real_runtime",
        lambda args, **kwargs: (runtime, object()),
    )
    monkeypatch.setattr(
        runtime_factory,
        "_make_request_block_hasher",
        lambda vllm_config: object(),
    )
    monkeypatch.setattr(
        runtime_factory,
        "EdgeDecodeEngineV1",
        object(),
        raising=False,
    )

    with pytest.raises(ValueError, match="async_scheduling=False"):
        runtime_factory.build_real_edge_service(
            SimpleNamespace(
                verifier_url="http://127.0.0.1:9000",
                eos_token_id=2,
                gamma=3,
                model_runner_version="v1",
                async_scheduling=True,
            )
        )


def test_build_real_edge_service_rejects_invalid_model_runner_version(
    monkeypatch,
) -> None:
    from vllm.dssd.entrypoints import runtime_factory

    init_calls: list[str] = []
    monkeypatch.setattr(
        runtime_factory,
        "_init_real_runtime",
        lambda *args, **kwargs: init_calls.append("called"),
    )

    with pytest.raises(ValueError, match="model_runner_version"):
        runtime_factory.build_real_edge_service(
            SimpleNamespace(
                verifier_url="http://127.0.0.1:9000",
                eos_token_id=2,
                gamma=3,
                model_runner_version="v3",
                async_scheduling=False,
            )
        )

    assert init_calls == []


def test_build_real_edge_service_cleans_up_when_v1_engine_init_fails(
    monkeypatch,
) -> None:
    from vllm.dssd.entrypoints import runtime_factory

    cleanup_calls: list[str] = []
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

    def fake_cleanup() -> None:
        cleanup_calls.append("cleanup")
        runtime.worker.shutdown()

    monkeypatch.setattr(
        runtime_factory,
        "_init_real_runtime",
        lambda args, **kwargs: (runtime, fake_cleanup),
    )
    monkeypatch.setattr(
        runtime_factory,
        "_make_request_block_hasher",
        lambda vllm_config: object(),
    )

    class FakeEdgeStateBridgeV1:
        pass

    class FakeDSSDEdgeDraftSamplerV1:
        def __init__(self, sampler) -> None:
            del sampler

    class FailingEdgeDecodeEngineV1:
        def __init__(self, **kwargs) -> None:
            del kwargs
            raise RuntimeError("engine init failed")

    monkeypatch.setattr(
        runtime_factory,
        "EdgeStateBridgeV1",
        FakeEdgeStateBridgeV1,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_factory,
        "DSSDEdgeDraftSamplerV1",
        FakeDSSDEdgeDraftSamplerV1,
        raising=False,
    )
    monkeypatch.setattr(
        runtime_factory,
        "EdgeDecodeEngineV1",
        FailingEdgeDecodeEngineV1,
        raising=False,
    )

    with pytest.raises(RuntimeError, match="engine init failed"):
        runtime_factory.build_real_edge_service(
            SimpleNamespace(
                verifier_url="http://127.0.0.1:9000",
                eos_token_id=2,
                gamma=3,
                model_runner_version="v1",
                async_scheduling=False,
            )
        )

    assert cleanup_calls == ["cleanup"]
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


def test_init_real_runtime_warms_up_worker_after_kv_cache_init(monkeypatch) -> None:
    from vllm.dssd.entrypoints import runtime_factory

    events: list[str] = []

    class FakeEngineArgs:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def create_engine_config(self):
            return fake_vllm_config

    class FakeWorker:
        def __init__(self, **kwargs) -> None:
            del kwargs
            self.model_runner = SimpleNamespace(sampler=object())

        def init_device(self) -> None:
            events.append("init_device")

        def load_model(self) -> None:
            events.append("load_model")

        def get_kv_cache_spec(self):
            events.append("get_kv_cache_spec")
            return object()

        def determine_available_memory(self):
            events.append("determine_available_memory")
            return 123

        def initialize_from_config(self, kv_cache_config) -> None:
            del kv_cache_config
            events.append("initialize_from_config")

        def compile_or_warm_up_model(self) -> None:
            events.append("compile_or_warm_up_model")

        def shutdown(self) -> None:
            events.append("shutdown")

    fake_vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16,
            enable_prefix_caching=False,
        ),
        model_config=SimpleNamespace(max_model_len=128),
    )

    monkeypatch.setattr(runtime_factory, "EngineArgs", FakeEngineArgs)
    monkeypatch.setattr(runtime_factory, "Worker", FakeWorker)
    monkeypatch.setattr(
        runtime_factory,
        "set_current_vllm_config",
        lambda _cfg: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        runtime_factory,
        "get_kv_cache_configs",
        lambda *args, **kwargs: [object()],
    )
    monkeypatch.setattr(
        runtime_factory,
        "generate_scheduler_kv_cache_config",
        lambda _cfgs: SimpleNamespace(num_blocks=7),
    )
    monkeypatch.setattr(
        runtime_factory,
        "KVCacheManager",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(runtime_factory, "envs", SimpleNamespace(disable_envs_cache=lambda: None))

    runtime, cleanup = runtime_factory._init_real_runtime(
        SimpleNamespace(
            model="/tmp/model",
            enforce_eager=False,
            async_scheduling=False,
            max_model_len=64,
            gpu_memory_utilization=0.01,
            kv_cache_memory_bytes=128 * 1024 * 1024,
            max_num_batched_tokens=64,
            max_num_seqs=2,
        )
    )

    try:
        assert runtime.worker is not None
        assert runtime.kv_cache_manager.enable_caching is False
        assert events == [
            "init_device",
            "load_model",
            "determine_available_memory",
            "get_kv_cache_spec",
            "initialize_from_config",
            "compile_or_warm_up_model",
        ]
    finally:
        cleanup()
