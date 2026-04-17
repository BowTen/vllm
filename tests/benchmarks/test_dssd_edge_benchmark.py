# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

import vllm.envs as envs


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "benchmarks" / "dssd" / "benchmark_edge_vs_vllm.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_edge_vs_vllm",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_edge_runtime_enables_v2_model_runner_before_worker_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()

    class StopInit(RuntimeError):
        pass

    seen: dict[str, object] = {}

    class FakeWorker:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            seen["os_env"] = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
            seen["envs_attr"] = envs.VLLM_USE_V2_MODEL_RUNNER
            raise StopInit("stop after checking env")

    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    envs.disable_envs_cache()
    assert envs.VLLM_USE_V2_MODEL_RUNNER is False

    monkeypatch.setattr(module, "build_vllm_config", lambda _args: SimpleNamespace())
    monkeypatch.setattr(
        module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(module, "require_cuda", lambda: None)
    monkeypatch.setattr(module, "Worker", FakeWorker)

    runtime = module.EdgeRuntime(
        SimpleNamespace(
            model="/tmp/model",
            trust_remote_code=False,
        )
    )

    with pytest.raises(StopInit, match="stop after checking env"):
        runtime.__enter__()

    assert seen["os_env"] == "1"
    assert seen["envs_attr"] is True


def test_edge_runtime_warms_up_worker_after_kv_cache_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()

    events: list[str] = []

    class FakeWorker:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
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
            num_gpu_blocks=0,
            block_size=16,
            enable_prefix_caching=False,
        ),
        model_config=SimpleNamespace(max_model_len=128),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        validate_block_size=lambda: events.append("validate_block_size"),
    )
    fake_scheduler_kv_cache_config = SimpleNamespace(
        num_blocks=7,
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=16))],
    )

    monkeypatch.setattr(module, "build_vllm_config", lambda _args: fake_vllm_config)
    monkeypatch.setattr(
        module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(module, "require_cuda", lambda: None)
    monkeypatch.setattr(module, "Worker", FakeWorker)
    monkeypatch.setattr(module, "set_current_vllm_config", lambda _cfg: nullcontext())
    monkeypatch.setattr(module.platforms.current_platform, "update_block_size_for_backend", lambda _cfg: None)
    monkeypatch.setattr(module, "get_kv_cache_configs", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(
        module,
        "generate_scheduler_kv_cache_config",
        lambda _cfgs: fake_scheduler_kv_cache_config,
    )
    monkeypatch.setattr(module, "KVCacheManager", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(
        module,
        "EdgeDecodeEngine",
        lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(module, "EdgeSchedulerAdapter", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(module, "EdgeStateBridge", lambda: object())
    monkeypatch.setattr(module, "DSSDEdgeDraftSampler", lambda sampler: ("draft_sampler", sampler))

    runtime = module.EdgeRuntime(
        SimpleNamespace(
            model="/tmp/model",
            trust_remote_code=False,
        )
    )

    entered_runtime = runtime.__enter__()
    try:
        assert entered_runtime is runtime
        assert events == [
            "init_device",
            "load_model",
            "get_kv_cache_spec",
            "determine_available_memory",
            "validate_block_size",
            "initialize_from_config",
            "compile_or_warm_up_model",
        ]
    finally:
        runtime.__exit__(None, None, None)


def test_parse_args_defaults_native_model_runner_to_v1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()

    monkeypatch.setattr(sys, "argv", ["benchmark", "--model", "/tmp/model"])

    args = module.parse_args()

    assert args.native_model_runner == "v1"


def test_benchmark_native_enables_v2_model_runner_before_llm_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()

    class StopInit(RuntimeError):
        pass

    seen: dict[str, object] = {}

    class FakeLLM:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            seen["os_env"] = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
            seen["envs_attr"] = envs.VLLM_USE_V2_MODEL_RUNNER
            raise StopInit("stop after checking env")

    class FakeTokenizer:
        def __call__(self, prompt: str) -> SimpleNamespace:
            del prompt
            return SimpleNamespace(input_ids=[1, 2, 3])

    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    envs.disable_envs_cache()
    assert envs.VLLM_USE_V2_MODEL_RUNNER is False

    monkeypatch.setattr(module, "require_cuda", lambda: None)
    monkeypatch.setattr(
        module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(module, "LLM", FakeLLM)

    with pytest.raises(StopInit, match="stop after checking env"):
        module.benchmark_native(
            SimpleNamespace(
                model="/tmp/model",
                prompt="hello",
                trust_remote_code=False,
                dtype="auto",
                gpu_memory_utilization=0.9,
                max_model_len=4096,
                enforce_eager=False,
                native_model_runner="v2",
            ),
            SimpleNamespace(),
        )

    assert seen["os_env"] == "1"
    assert seen["envs_attr"] is True


def test_benchmark_native_keeps_v1_model_runner_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()

    class StopInit(RuntimeError):
        pass

    seen: dict[str, object] = {}

    class FakeLLM:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            seen["os_env"] = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
            seen["envs_attr"] = envs.VLLM_USE_V2_MODEL_RUNNER
            raise StopInit("stop after checking env")

    class FakeTokenizer:
        def __call__(self, prompt: str) -> SimpleNamespace:
            del prompt
            return SimpleNamespace(input_ids=[1, 2, 3])

    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    envs.disable_envs_cache()
    assert envs.VLLM_USE_V2_MODEL_RUNNER is False

    monkeypatch.setattr(module, "require_cuda", lambda: None)
    monkeypatch.setattr(
        module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(module, "LLM", FakeLLM)

    with pytest.raises(StopInit, match="stop after checking env"):
        module.benchmark_native(
            SimpleNamespace(
                model="/tmp/model",
                prompt="hello",
                trust_remote_code=False,
                dtype="auto",
                gpu_memory_utilization=0.9,
                max_model_len=4096,
                enforce_eager=False,
                native_model_runner="v1",
            ),
            SimpleNamespace(),
        )

    assert seen["os_env"] is None
    assert seen["envs_attr"] is False


def test_run_subprocess_for_engine_passes_native_model_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()

    captured: dict[str, object] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            stdout='{"engine":"native","model":"/tmp/model","prompt_tokens":1,'
            '"requested_output_tokens":1,"generated_tokens":1,"warmup_iters":1,'
            '"benchmark_iters":1,"total_seconds":1.0,"avg_seconds":1.0,'
            '"tokens_per_second":1.0}'
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.run_subprocess_for_engine(
        SimpleNamespace(
            model="/tmp/model",
            prompt="hello",
            max_new_tokens=128,
            warmup_iters=1,
            benchmark_iters=5,
            dtype="auto",
            gpu_memory_utilization=0.9,
            max_model_len=4096,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
            trust_remote_code=False,
            enforce_eager=False,
            honor_eos=False,
            native_model_runner="v2",
        ),
        "native",
    )

    assert result.engine == "native"
    assert "--native-model-runner" in captured["cmd"]
    native_model_runner_idx = captured["cmd"].index("--native-model-runner")
    assert captured["cmd"][native_model_runner_idx + 1] == "v2"


def test_edge_runtime_exit_cleans_up_distributed_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()

    events: list[str] = []

    class FakeWorker:
        def shutdown(self) -> None:
            events.append("shutdown")

    monkeypatch.setattr(module, "build_vllm_config", lambda _args: SimpleNamespace())
    monkeypatch.setattr(
        module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(module, "cleanup_dist_env_and_memory", lambda: events.append("cleanup"), raising=False)
    monkeypatch.setattr(module.gc, "collect", lambda: events.append("gc_collect"))
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(module.envs, "disable_envs_cache", lambda: events.append("disable_envs_cache"))

    runtime = module.EdgeRuntime(
        SimpleNamespace(
            model="/tmp/model",
            trust_remote_code=False,
        )
    )
    runtime.worker = FakeWorker()
    runtime._tempfile = None
    runtime._old_use_v2_model_runner = None

    runtime.__exit__(None, None, None)

    assert events == [
        "shutdown",
        "cleanup",
        "gc_collect",
        "disable_envs_cache",
    ]
