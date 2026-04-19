# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "benchmarks" / "dssd" / "benchmark_edge_verifier_decode.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_edge_verifier_decode",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dssd_system_benchmark_script_exists_and_parses_device_flags() -> None:
    module = _load_module()

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--model",
            "/tmp/model",
            "--edge-cuda-visible-devices",
            "0",
            "--verifier-cuda-visible-devices",
            "1",
            "--prompt-len",
            "64",
            "--decode-tokens",
            "32",
            "--gamma",
            "2",
        ]
    )

    assert args.model == "/tmp/model"
    assert args.edge_cuda_visible_devices == "0"
    assert args.verifier_cuda_visible_devices == "1"
    assert args.prompt_len == 64
    assert args.decode_tokens == 32
    assert args.gamma == 2


def test_dssd_system_benchmark_parser_defaults_to_mrv1() -> None:
    module = _load_module()

    args = module.build_parser().parse_args([])

    assert args.edge_model_runner == "v1"
    assert args.verifier_model_runner == "v1"


def test_dssd_system_benchmark_parser_accepts_model_runner_flags() -> None:
    module = _load_module()

    args = module.build_parser().parse_args(
        [
            "--edge-model-runner",
            "v2",
            "--verifier-model-runner",
            "v2",
        ]
    )

    assert args.edge_model_runner == "v2"
    assert args.verifier_model_runner == "v2"


def test_start_verifier_server_passes_model_runner_version_and_gamma(
    monkeypatch,
    tmp_path,
) -> None:
    module = _load_module()
    captured = {}

    class FakeProc:
        def poll(self):
            return None

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return FakeProc()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(module.tempfile, "mkdtemp", lambda prefix: str(tmp_path))
    monkeypatch.setattr(
        module,
        "_wait_for_server_url",
        lambda ready_file, proc: "http://127.0.0.1:18021",
    )

    config = module.BenchmarkConfig(
        model="/tmp/model",
        edge_model="/tmp/model",
        verifier_model="/tmp/model",
        edge_model_runner="v1",
        verifier_model_runner="v1",
        edge_cuda_visible_devices="0",
        verifier_cuda_visible_devices="0",
        verifier_host="127.0.0.1",
        verifier_port=18021,
        prompt_len=18,
        decode_tokens=8,
        warmup_tokens=0,
        repeats=1,
        gamma=2,
        gpu_memory_utilization=0.01,
        max_model_len=64,
        kv_cache_memory_bytes=1024,
        max_num_batched_tokens=16,
        max_num_seqs=2,
        enforce_eager=True,
        async_scheduling=False,
    )

    proc, server_url, ready_dir = module._start_verifier_server(config)

    assert isinstance(proc, FakeProc)
    assert server_url == "http://127.0.0.1:18021"
    assert ready_dir == tmp_path
    cmd = captured["cmd"]
    assert cmd[cmd.index("--model-runner-version") + 1] == "v1"
    assert cmd[cmd.index("--gamma") + 1] == "2"


def test_build_edge_service_passes_model_runner_version(monkeypatch) -> None:
    module = _load_module()
    captured = {}
    sentinel_service = object()
    sentinel_cleanup = object()

    from vllm.dssd.entrypoints import runtime_factory

    def fake_build_real_edge_service(args):
        captured["args"] = args
        return sentinel_service, sentinel_cleanup

    monkeypatch.setattr(
        runtime_factory,
        "build_real_edge_service",
        fake_build_real_edge_service,
    )

    config = module.BenchmarkConfig(
        model="/tmp/model",
        edge_model="/tmp/model",
        verifier_model="/tmp/model",
        edge_model_runner="v1",
        verifier_model_runner="v2",
        edge_cuda_visible_devices="0",
        verifier_cuda_visible_devices="0",
        verifier_host="127.0.0.1",
        verifier_port=18021,
        prompt_len=18,
        decode_tokens=8,
        warmup_tokens=0,
        repeats=1,
        gamma=1,
        gpu_memory_utilization=0.01,
        max_model_len=64,
        kv_cache_memory_bytes=1024,
        max_num_batched_tokens=16,
        max_num_seqs=2,
        enforce_eager=True,
        async_scheduling=False,
    )

    service, cleanup = module._build_edge_service(
        config,
        "http://127.0.0.1:18021",
    )

    assert service is sentinel_service
    assert cleanup is sentinel_cleanup
    assert captured["args"].model_runner_version == "v1"
    assert captured["args"].gamma == 1


def test_dssd_system_benchmark_resolves_max_num_batched_tokens_from_prompt_len(
) -> None:
    module = _load_module()

    args = module.build_parser().parse_args(
        [
            "--prompt-len",
            "128",
            "--max-num-batched-tokens",
            "64",
        ]
    )

    assert module._resolved_max_num_batched_tokens(args) == 128


def test_dssd_system_decode_counter_handles_multi_token_rounds() -> None:
    module = _load_module()

    counter = module.DecodeMeasurementCounter(warmup_tokens=2, target_tokens=3)

    assert counter.consume([101]) == []
    assert counter.consume([102, 103]) == []
    assert counter.consume([104, 105]) == [104, 105]
    assert counter.consume([106, 107]) == [106]
    assert counter.is_complete()


def test_dssd_system_benchmark_stops_verifier_and_skips_edge_cleanup(
    monkeypatch,
    tmp_path,
) -> None:
    module = _load_module()
    order: list[str] = []

    class FakeProc:
        pass

    class FakeEdgeService:
        decode_engine = type("DecodeEngine", (), {"sessions": {}})()

        def open_session(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        module,
        "_start_verifier_server",
        lambda config: (FakeProc(), "http://127.0.0.1:18021", tmp_path),
    )
    monkeypatch.setattr(
        module,
        "_build_edge_service",
        lambda config, server_url: (
            FakeEdgeService(),
            lambda: order.append("edge_cleanup"),
        ),
    )
    monkeypatch.setattr(
        module,
        "_stop_process",
        lambda proc: order.append("stop_verifier"),
    )

    config = module.BenchmarkConfig(
        model="/tmp/model",
        edge_model="/tmp/model",
        verifier_model="/tmp/model",
        edge_model_runner="v1",
        verifier_model_runner="v1",
        edge_cuda_visible_devices="0",
        verifier_cuda_visible_devices="0",
        verifier_host="127.0.0.1",
        verifier_port=18021,
        prompt_len=18,
        decode_tokens=8,
        warmup_tokens=0,
        repeats=1,
        gamma=1,
        gpu_memory_utilization=0.01,
        max_model_len=64,
        kv_cache_memory_bytes=1024,
        max_num_batched_tokens=64,
        max_num_seqs=2,
        enforce_eager=False,
        async_scheduling=False,
    )

    with pytest.raises(RuntimeError, match="boom"):
        module._run_dssd_benchmark(config, [1] * 18)

    assert order == ["stop_verifier"]
