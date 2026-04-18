# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "benchmarks" / "dssd" / "benchmark_local_engines.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "benchmark_local_engines",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_local_engine_benchmark_parser_defaults_to_all_and_prompt_len() -> None:
    module = _load_module()

    args = module.build_parser().parse_args(["--model", "/tmp/model"])

    assert args.engine == "all"
    assert args.prompt_len == 128
    assert args.native_model_runner == "v1"
    assert args.edge_model_runner == "v2"
    assert args.verifier_model_runner == "v2"
    assert args.phase_timing is False
    assert args.dump_decode_state == 0
    assert args.output_json is None


def test_local_engine_benchmark_parser_accepts_edge_model_runner() -> None:
    module = _load_module()

    args = module.build_parser().parse_args(
        ["--model", "/tmp/model", "--edge-model-runner", "v1"]
    )

    assert args.edge_model_runner == "v1"


def test_local_engine_benchmark_parser_accepts_verifier_model_runner() -> None:
    module = _load_module()

    args = module.build_parser().parse_args(
        ["--model", "/tmp/model", "--verifier-model-runner", "v1"]
    )

    assert args.verifier_model_runner == "v1"


def test_local_engine_benchmark_parser_accepts_auto_output_json() -> None:
    module = _load_module()

    args = module.build_parser().parse_args(
        ["--model", "/tmp/model", "--output-json"]
    )

    assert args.output_json == "auto"


def test_run_subprocess_for_engine_passes_prompt_len_and_json_only(
    monkeypatch,
) -> None:
    module = _load_module()
    args = module.build_parser().parse_args(
        [
            "--engine",
            "all",
            "--model",
            "/tmp/model",
            "--prompt-len",
            "18",
            "--max-new-tokens",
            "32",
            "--warmup-iters",
            "1",
            "--benchmark-iters",
            "2",
            "--dtype",
            "float16",
            "--native-model-runner",
            "v2",
            "--edge-model-runner",
            "v1",
            "--verifier-model-runner",
            "v1",
            "--trust-remote-code",
            "--no-enforce-eager",
            "--phase-timing",
            "--dump-decode-state",
            "3",
            "--output-json",
        ]
    )
    payload = {
        "engine": "verifier",
        "model": "/tmp/model",
        "prompt_tokens": 18,
        "requested_output_tokens": 32,
        "generated_tokens": 64,
        "warmup_iters": 1,
        "benchmark_iters": 2,
        "total_seconds": 1.0,
        "avg_seconds": 0.5,
        "tokens_per_second": 64.0,
    }
    calls: list[list[str]] = []

    def fake_run(cmd, *, check, capture_output, text):
        assert check is True
        assert capture_output is True
        assert text is True
        calls.append(cmd)
        return SimpleNamespace(stdout=json.dumps(payload))

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    result = module.run_subprocess_for_engine(args, "verifier")

    assert result == module.BenchmarkResult(**payload)
    cmd = calls[0]
    assert cmd[:2] == [sys.executable, str(module.SCRIPT_PATH)]
    assert cmd[cmd.index("--engine") + 1] == "verifier"
    assert cmd[cmd.index("--prompt-len") + 1] == "18"
    assert cmd[cmd.index("--max-new-tokens") + 1] == "32"
    assert cmd[cmd.index("--edge-model-runner") + 1] == "v1"
    assert cmd[cmd.index("--verifier-model-runner") + 1] == "v1"
    assert "--json-only" in cmd
    assert "--trust-remote-code" in cmd
    assert "--no-enforce-eager" in cmd
    assert "--phase-timing" in cmd
    assert cmd[cmd.index("--dump-decode-state") + 1] == "3"
    assert "--output-json" not in cmd


def test_make_verifier_engine_selects_mrv1_components() -> None:
    module = _load_module()
    args = module.build_parser().parse_args(
        ["--model", "/tmp/model", "--verifier-model-runner", "v1"]
    )
    sampler = object()
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(sampler=sampler, num_spec_tokens=0)
    )

    engine = module._make_verifier_engine(
        args=args,
        vllm_config=object(),
        worker=worker,
        scheduler=object(),
    )

    from vllm.dssd.verifier.engine_v1 import VerifierDecodeEngineV1
    from vllm.dssd.verifier.sampler_v1 import DSSDVerifierSamplerV1
    from vllm.dssd.verifier.state_bridge_v1 import VerifierStateBridgeV1

    assert isinstance(engine, VerifierDecodeEngineV1)
    assert isinstance(engine.state_bridge, VerifierStateBridgeV1)
    assert isinstance(engine.verifier_sampler, DSSDVerifierSamplerV1)


def test_make_edge_engine_selects_mrv1_components() -> None:
    module = _load_module()
    args = module.build_parser().parse_args(
        ["--model", "/tmp/model", "--edge-model-runner", "v1"]
    )
    sampler = object()
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(sampler=sampler, vocab_size=10)
    )

    engine = module._make_edge_engine(
        args=args,
        vllm_config=object(),
        worker=worker,
        kv_cache_manager=object(),
    )

    from vllm.dssd.edge.engine_v1 import EdgeDecodeEngineV1
    from vllm.dssd.edge.sampler_v1 import DSSDEdgeDraftSamplerV1
    from vllm.dssd.edge.state_bridge_v1 import EdgeStateBridgeV1

    assert isinstance(engine, EdgeDecodeEngineV1)
    assert isinstance(engine.state_bridge, EdgeStateBridgeV1)
    assert isinstance(engine.draft_sampler, DSSDEdgeDraftSamplerV1)


def test_benchmark_runtime_engine_aggregates_phase_seconds(
    monkeypatch,
) -> None:
    module = _load_module()
    calls: list[str] = []
    runtime = SimpleNamespace(engine=object())

    def fake_profiled_request(
        *,
        engine_name,
        runtime,
        req_id,
        prompt_token_ids,
        sampling_params,
        dump_decode_state_steps,
    ):
        del (
            engine_name,
            runtime,
            prompt_token_ids,
            sampling_params,
            dump_decode_state_steps,
        )
        calls.append(req_id)
        return [10, 11], {
            "open_session": 0.1,
            "prefill_execute": 0.2,
            "bootstrap": 0.3,
            "decode_schedule": 0.2,
            "decode_model": 0.05,
            "decode_postprocess": 0.15,
            "close_session": 0.05,
        }

    monkeypatch.setattr(
        module,
        "_run_profiled_runtime_request",
        fake_profiled_request,
    )
    monkeypatch.setattr(module, "synchronize_if_needed", lambda: None)

    result = module._benchmark_runtime_engine(
        engine_name="edge",
        model="/tmp/model",
        prompt_token_ids=[1, 2, 3],
        warmup_iters=1,
        benchmark_iters=2,
        sampling_params=SimpleNamespace(max_tokens=2),
        runtime=runtime,
        phase_timing=True,
        dump_decode_state_steps=0,
    )

    assert calls == ["warmup-0", "bench-0", "bench-1"]
    assert result.generated_tokens == 4
    assert abs(result.total_seconds - 2.1) < 1e-9
    assert result.phase_seconds == {
        "open_session": 0.2,
        "prefill_execute": 0.4,
        "bootstrap": 0.6,
        "decode_schedule": 0.4,
        "decode_model": 0.1,
        "decode_postprocess": 0.3,
        "close_session": 0.1,
    }


def test_benchmark_runtime_engine_uses_profiled_path_for_decode_dump(
    monkeypatch,
) -> None:
    module = _load_module()
    calls: list[tuple[str, int]] = []
    runtime = SimpleNamespace(engine=object())

    def fake_profiled_request(
        *,
        engine_name,
        runtime,
        req_id,
        prompt_token_ids,
        sampling_params,
        dump_decode_state_steps,
    ):
        del engine_name, runtime, prompt_token_ids, sampling_params
        calls.append((req_id, dump_decode_state_steps))
        phase_seconds = module._new_zero_phase_seconds()
        phase_seconds["decode_model"] = 0.25
        return [10], phase_seconds

    monkeypatch.setattr(
        module,
        "_run_profiled_runtime_request",
        fake_profiled_request,
    )
    monkeypatch.setattr(module, "synchronize_if_needed", lambda: None)

    result = module._benchmark_runtime_engine(
        engine_name="edge",
        model="/tmp/model",
        prompt_token_ids=[1, 2, 3],
        warmup_iters=1,
        benchmark_iters=2,
        sampling_params=SimpleNamespace(max_tokens=1),
        runtime=runtime,
        phase_timing=False,
        dump_decode_state_steps=2,
    )

    assert calls == [("warmup-0", 2), ("bench-0", 2), ("bench-1", 2)]
    assert result.generated_tokens == 2
    assert result.phase_seconds is None


def test_dump_edge_decode_state_supports_v2_runner(capsys) -> None:
    module = _load_module()
    runtime = SimpleNamespace(
        engine=SimpleNamespace(
            model_runner=SimpleNamespace(
                execute_model_state=SimpleNamespace(
                    input_batch=SimpleNamespace(
                        num_reqs=1,
                        req_ids=["req-0"],
                        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
                        seq_lens=torch.tensor([129], dtype=torch.int32),
                        positions=torch.tensor([128], dtype=torch.int64),
                        input_ids=torch.tensor([42], dtype=torch.int32),
                    )
                ),
                req_states=SimpleNamespace(
                    req_id_to_index={"req-0": 0},
                    num_computed_tokens=SimpleNamespace(
                        gpu=torch.tensor([128], dtype=torch.int32)
                    ),
                ),
                block_tables=SimpleNamespace(
                    num_blocks=SimpleNamespace(
                        np=np.array([[9]], dtype=np.int32)
                    )
                ),
            )
        )
    )
    session = SimpleNamespace(
        req_id="req-0",
        total_len=129,
        num_computed_tokens=128,
    )
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=1,
        num_scheduled_tokens={"req-0": 1},
        scheduled_cached_reqs=SimpleNamespace(
            num_computed_tokens=[128],
            num_output_tokens=[1],
            new_block_ids=[([7],)],
        ),
    )

    module._dump_edge_decode_state(
        runtime,
        req_id="bench-0",
        session=session,
        step_index=1,
        scheduler_output=scheduler_output,
    )

    output = capsys.readouterr().out.strip()
    assert output.startswith("EDGE_DECODE_STATE ")
    payload = json.loads(output.split(" ", 1)[1])
    assert payload["runner"] == "v2"
    assert payload["scheduler_total_num_scheduled_tokens"] == 1
    assert payload["input_batch_num_reqs"] == 1
    assert payload["input_batch_req_ids"] == ["req-0"]
    assert payload["input_batch_num_computed_tokens"] == 128
    assert payload["query_start_loc"] == [0, 1]
    assert payload["seq_lens"] == [129]
    assert payload["positions"] == [128]
    assert payload["input_ids"] == [42]
    assert payload["block_table_num_blocks"] == [9]


def test_main_all_uses_subprocesses_only(
    monkeypatch,
    capsys,
) -> None:
    module = _load_module()
    calls: list[str] = []

    def make_result(engine: str, tok_s: float):
        return module.BenchmarkResult(
            engine=engine,
            model="/tmp/model",
            prompt_tokens=18,
            requested_output_tokens=32,
            generated_tokens=64,
            warmup_iters=1,
            benchmark_iters=2,
            total_seconds=1.0,
            avg_seconds=0.5,
            tokens_per_second=tok_s,
        )

    def fake_run_subprocess_for_engine(args, engine):
        del args
        calls.append(engine)
        speed = {"native": 100.0, "edge": 120.0, "verifier": 140.0}[engine]
        return make_result(engine, speed)

    def fail_if_parent_initializes_cuda():
        raise AssertionError("parent should not initialize engines in all mode")

    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark", "--engine", "all", "--model", "/tmp/model", "--json"],
    )
    monkeypatch.setattr(module, "run_subprocess_for_engine", fake_run_subprocess_for_engine)
    monkeypatch.setattr(module, "require_cuda", fail_if_parent_initializes_cuda)

    module.main()

    assert calls == ["native", "edge", "verifier"]
    output = capsys.readouterr().out
    assert '"engine": "native"' in output
    assert '"engine": "edge"' in output
    assert '"engine": "verifier"' in output


def test_main_writes_output_json_when_requested(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    module = _load_module()
    output_path = tmp_path / "local-engines.json"

    def make_result(engine: str, tok_s: float):
        return module.BenchmarkResult(
            engine=engine,
            model="/tmp/model",
            prompt_tokens=18,
            requested_output_tokens=32,
            generated_tokens=64,
            warmup_iters=1,
            benchmark_iters=2,
            total_seconds=1.0,
            avg_seconds=0.5,
            tokens_per_second=tok_s,
        )

    def fake_run_subprocess_for_engine(args, engine):
        del args
        speed = {"native": 100.0, "edge": 120.0, "verifier": 140.0}[engine]
        return make_result(engine, speed)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--engine",
            "all",
            "--model",
            "/tmp/model",
            "--json",
            "--output-json",
            str(output_path),
        ],
    )
    monkeypatch.setattr(module, "run_subprocess_for_engine", fake_run_subprocess_for_engine)

    module.main()

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert [item["engine"] for item in payload] == ["native", "edge", "verifier"]
    stdout = capsys.readouterr().out
    assert '"engine": "native"' in stdout


def test_maybe_write_results_json_auto_uses_default_results_dir(
    monkeypatch,
) -> None:
    module = _load_module()
    fixed_timestamp = "20260417-120000"
    args = module.build_parser().parse_args(
        ["--engine", "edge", "--model", "/tmp/Qwen3-0.6B", "--output-json"]
    )
    results = [
        module.BenchmarkResult(
            engine="edge",
            model="/tmp/Qwen3-0.6B",
            prompt_tokens=128,
            requested_output_tokens=128,
            generated_tokens=128,
            warmup_iters=1,
            benchmark_iters=1,
            total_seconds=1.0,
            avg_seconds=1.0,
            tokens_per_second=128.0,
        )
    ]
    written: dict[str, str] = {}

    monkeypatch.setattr(module.time, "strftime", lambda fmt, tm: fixed_timestamp)
    monkeypatch.setattr(module.time, "gmtime", lambda: object())

    def fake_write_text(self, text, encoding):
        written["path"] = str(self)
        written["text"] = text
        return len(text)

    monkeypatch.setattr(Path, "write_text", fake_write_text)
    monkeypatch.setattr(Path, "mkdir", lambda self, parents, exist_ok: None)

    path = module.maybe_write_results_json(args, results)

    assert path == (
        module.DEFAULT_RESULT_DIR
        / f"{fixed_timestamp}-local-engines-edge-Qwen3-0.6B.json"
    )
    assert json.loads(written["text"])["engine"] == "edge"
