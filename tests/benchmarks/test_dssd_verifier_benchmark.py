# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import vllm.envs as envs


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "benchmarks" / "dssd" / "benchmark_verifier_vs_vllm.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_verifier_vs_vllm",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verifier_benchmark_script_exists_and_parses_engine_flag() -> None:
    module = _load_module()

    parser = module.build_parser()
    args = parser.parse_args(
        [
            "--engine",
            "verifier",
            "--model",
            "/tmp/model",
            "--prompt-len",
            "128",
            "--decode-tokens",
            "64",
        ]
    )

    assert args.engine == "verifier"
    assert args.model == "/tmp/model"
    assert args.prompt_len == 128
    assert args.decode_tokens == 64
    assert args.native_model_runner == "v1"
    assert args.validate_token_alignment is False


def test_verifier_benchmark_parses_validate_token_alignment() -> None:
    module = _load_module()

    args = module.build_parser().parse_args(
        [
            "--engine",
            "both",
            "--model",
            "/tmp/model",
            "--validate-token-alignment",
        ]
    )

    assert args.validate_token_alignment is True


def test_verifier_benchmark_parses_native_model_runner_v2() -> None:
    module = _load_module()

    args = module.build_parser().parse_args(
        [
            "--engine",
            "native",
            "--model",
            "/tmp/model",
            "--native-model-runner",
            "v2",
        ]
    )

    assert args.native_model_runner == "v2"


def test_run_native_repeat_times_full_request_without_per_token_sync(
    monkeypatch,
) -> None:
    module = _load_module()
    config = module.BenchmarkConfig(
        model="/tmp/model",
        prompt_len=4,
        decode_tokens=2,
        warmup_tokens=1,
        repeats=1,
        gpu_memory_utilization=0.9,
        max_model_len=16,
        dtype="auto",
        enforce_eager=True,
        trust_remote_code=False,
        native_model_runner="v1",
    )
    events: list[object] = []

    class FakeEngine:
        def add_request(self, req_id, prompt, params) -> None:
            events.append(("add_request", req_id, prompt, params.max_tokens))

        def step(self):
            events.append("step")
            if events.count("step") == 1:
                return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[10])])]
            return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[11, 12, 13])])]

    perf_counter_values = iter([10.0, 10.5])
    monkeypatch.setattr(module, "_synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(
        module.time,
        "perf_counter",
        lambda: next(perf_counter_values),
    )

    result = module._run_native_repeat(
        FakeEngine(),
        config,
        req_id="native-test",
        prompt_token_ids=[1, 2, 3, 4],
    )

    assert events == [
        "sync",
        ("add_request", "native-test", {"prompt_token_ids": [1, 2, 3, 4]}, 4),
        "step",
        "step",
        "sync",
    ]
    assert result.token_ids == [10, 11, 12, 13]
    assert result.total_s == 0.5
    assert result.tokens_per_s == 8.0
    assert result.per_token_ms == 125.0


def test_run_verifier_repeat_times_full_request_without_per_token_sync(
    monkeypatch,
) -> None:
    module = _load_module()
    config = module.BenchmarkConfig(
        model="/tmp/model",
        prompt_len=4,
        decode_tokens=2,
        warmup_tokens=1,
        repeats=1,
        gpu_memory_utilization=0.9,
        max_model_len=16,
        dtype="auto",
        enforce_eager=True,
        trust_remote_code=False,
        native_model_runner="v1",
    )
    events: list[object] = []

    class FakeEngine:
        def generate_local(self, req_id, prompt_token_ids, sampling_params):
            events.append(
                (
                    "generate_local",
                    req_id,
                    prompt_token_ids,
                    sampling_params.max_tokens,
                )
            )
            return [20, 21, 22, 23]

    perf_counter_values = iter([30.0, 30.25])
    monkeypatch.setattr(module, "_synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(
        module.time,
        "perf_counter",
        lambda: next(perf_counter_values),
    )

    result = module._run_verifier_repeat(
        FakeEngine(),
        config,
        req_id="verifier-test",
        prompt_token_ids=[1, 2, 3, 4],
    )

    assert events == [
        "sync",
        ("generate_local", "verifier-test", [1, 2, 3, 4], 4),
        "sync",
    ]
    assert result.token_ids == [20, 21, 22, 23]
    assert result.total_s == 0.25
    assert result.tokens_per_s == 16.0
    assert result.per_token_ms == 62.5


def test_run_subprocess_for_engine_uses_single_engine_json_only(
    monkeypatch,
) -> None:
    module = _load_module()
    args = module.build_parser().parse_args(
        [
            "--engine",
            "both",
            "--model",
            "/tmp/model",
            "--prompt-len",
            "18",
            "--decode-tokens",
            "32",
            "--warmup-tokens",
            "1",
            "--repeats",
            "2",
            "--gpu-memory-utilization",
            "0.8",
            "--dtype",
            "float16",
            "--native-model-runner",
            "v2",
            "--trust-remote-code",
            "--no-enforce-eager",
            "--validate-token-alignment",
        ]
    )
    calls: list[list[str]] = []
    payload = {
        "config": {
            "model": "/tmp/model",
            "prompt_len": 18,
            "decode_tokens": 32,
            "warmup_tokens": 1,
            "repeats": 2,
            "gpu_memory_utilization": 0.8,
            "max_model_len": 83,
            "dtype": "float16",
            "enforce_eager": False,
            "trust_remote_code": True,
            "native_model_runner": "v2",
        },
        "native": {
            "repeats": [],
            "summary": {"mean_tokens_per_s": 10.0},
        },
    }

    def fake_run(cmd, *, check, capture_output, text):
        assert check is True
        assert capture_output is True
        assert text is True
        calls.append(cmd)
        return SimpleNamespace(stdout=json.dumps(payload))

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.run_subprocess_for_engine(args, "native")

    assert result == payload
    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[:2] == [sys.executable, str(module.SCRIPT_PATH)]
    assert cmd[cmd.index("--engine") + 1] == "native"
    assert cmd[cmd.index("--model") + 1] == "/tmp/model"
    assert "--json-only" in cmd
    assert "--no-enforce-eager" in cmd
    assert "--trust-remote-code" in cmd
    assert "--validate-token-alignment" not in cmd


def test_run_subprocess_for_engine_parses_json_after_vllm_logs(
    monkeypatch,
) -> None:
    module = _load_module()
    args = module.build_parser().parse_args(
        ["--engine", "both", "--model", "/tmp/model"]
    )
    payload = {
        "config": {"model": "/tmp/model"},
        "native": {"repeats": [], "summary": {"mean_tokens_per_s": 10.0}},
    }

    def fake_run(cmd, *, check, capture_output, text):
        del cmd, check, capture_output, text
        return SimpleNamespace(stdout="WARNING some vLLM log\n" + json.dumps(payload))

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.run_subprocess_for_engine(args, "native") == payload


def test_main_both_uses_subprocess_isolation_and_summarizes_speedup(
    monkeypatch,
    capsys,
) -> None:
    module = _load_module()
    calls: list[str] = []
    native_payload = {
        "config": {
            "model": "/tmp/model",
            "prompt_len": 18,
            "decode_tokens": 2,
            "warmup_tokens": 0,
            "repeats": 1,
            "gpu_memory_utilization": 0.9,
            "max_model_len": 82,
            "dtype": "auto",
            "enforce_eager": True,
            "trust_remote_code": False,
            "native_model_runner": "v1",
        },
        "native": {
            "repeats": [
                {
                    "total_s": 0.02,
                    "tokens_per_s": 100.0,
                    "per_token_ms": 10.0,
                    "token_ids": [1, 2],
                }
            ],
            "summary": {
                "mean_tokens_per_s": 100.0,
                "stdev_tokens_per_s": 0.0,
                "mean_per_token_ms": 10.0,
                "stdev_per_token_ms": 0.0,
                "mean_total_s": 0.02,
            },
        },
    }
    verifier_payload = {
        "config": native_payload["config"],
        "verifier": {
            "repeats": [
                {
                    "total_s": 0.01,
                    "tokens_per_s": 200.0,
                    "per_token_ms": 5.0,
                    "token_ids": [3, 4],
                }
            ],
            "summary": {
                "mean_tokens_per_s": 200.0,
                "stdev_tokens_per_s": 0.0,
                "mean_per_token_ms": 5.0,
                "stdev_per_token_ms": 0.0,
                "mean_total_s": 0.01,
            },
        },
    }

    def fake_run_subprocess_for_engine(args, engine):
        del args
        calls.append(engine)
        if engine == "native":
            return native_payload
        return verifier_payload

    def fail_if_parent_initializes_cuda():
        raise AssertionError("parent should not initialize CUDA in both mode")

    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark", "--engine", "both", "--model", "/tmp/model", "--json"],
    )
    monkeypatch.setattr(
        module,
        "run_subprocess_for_engine",
        fake_run_subprocess_for_engine,
    )
    monkeypatch.setattr(module, "_require_cuda", fail_if_parent_initializes_cuda)

    module.main()

    assert calls == ["native", "verifier"]
    output = capsys.readouterr().out
    assert "Speedup verifier/native = 2.0000x" in output
    assert '"speedup_verifier_over_native": 2.0' in output


def test_generated_tokens_per_request_includes_bootstrap_and_warmup() -> None:
    module = _load_module()
    config = module.BenchmarkConfig(
        model="/tmp/model",
        prompt_len=4,
        decode_tokens=2,
        warmup_tokens=1,
        repeats=1,
        gpu_memory_utilization=0.9,
        max_model_len=16,
        dtype="auto",
        enforce_eager=True,
        trust_remote_code=False,
        native_model_runner="v1",
    )

    assert module._generated_tokens_per_request(config) == 4


def test_set_model_runner_env_can_select_v1_and_restore(
    monkeypatch,
) -> None:
    module = _load_module()

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    envs.disable_envs_cache()
    assert envs.VLLM_USE_V2_MODEL_RUNNER is True

    old_value = module._set_model_runner_env("v1")

    assert old_value == "1"
    assert os.environ.get("VLLM_USE_V2_MODEL_RUNNER") is None
    assert envs.VLLM_USE_V2_MODEL_RUNNER is False

    module._restore_model_runner_env(old_value)

    assert os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == "1"
    assert envs.VLLM_USE_V2_MODEL_RUNNER is True


def test_set_model_runner_env_can_select_v2_and_restore(
    monkeypatch,
) -> None:
    module = _load_module()

    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    envs.disable_envs_cache()
    assert envs.VLLM_USE_V2_MODEL_RUNNER is False

    old_value = module._set_model_runner_env("v2")

    assert old_value is None
    assert os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == "1"
    assert envs.VLLM_USE_V2_MODEL_RUNNER is True

    module._restore_model_runner_env(old_value)

    assert os.environ.get("VLLM_USE_V2_MODEL_RUNNER") is None
    assert envs.VLLM_USE_V2_MODEL_RUNNER is False
