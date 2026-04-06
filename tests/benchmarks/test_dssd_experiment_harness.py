# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import csv
import json
import socket
import sys
import textwrap
from pathlib import Path

import pytest

from vllm.benchmarks.dssd_experiment import (
    DSSDExperimentConfig,
    DSSDExperimentHarness,
)
from vllm.entrypoints.cli.benchmark.main import BenchmarkSubcommand
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.dssd.transport import HTTPDSSDTransport


def _open_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_experiment_config_expands_cartesian_run_cases(tmp_path: Path):
    config = DSSDExperimentConfig.from_dict(
        {
            "output_dir": str(tmp_path / "results"),
            "draft_models": ["draft-a", "draft-b"],
            "target_models": ["target-a"],
            "experiment_modes": ["dssd", "target-baseline"],
            "gammas": [2, 4],
            "network_profiles": [
                {"name": "lan", "latency_ms": 0.0},
                {"name": "wan", "latency_ms": 20.0, "bandwidth_mbps": 100.0},
            ],
            "edge": {
                "host": "127.0.0.1",
                "port": 8000,
                "cuda_visible_devices": "1",
            },
            "verifier": {
                "host": "127.0.0.1",
                "port": 9001,
                "cuda_visible_devices": "0",
            },
            "requests": [
                {
                    "messages": [{"role": "user", "content": "Count to 3."}],
                    "max_tokens": 8,
                }
            ],
        }
    )

    cases = config.expand_cases()

    assert len(cases) == 12
    assert {case.experiment_mode for case in cases} == {"dssd", "target-baseline"}
    assert {case.gamma for case in cases} == {2, 4}
    assert {case.network_profile.name for case in cases} == {"lan", "wan"}
    assert len({case.case_id for case in cases}) == len(cases)
    baseline_cases = [
        case for case in cases if case.experiment_mode == "target-baseline"
    ]
    assert len(baseline_cases) == 4
    assert all(case.draft_model is None for case in baseline_cases)


def test_experiment_config_rejects_dssd_without_target_models(tmp_path: Path):
    with pytest.raises(ValueError, match="target_models"):
        DSSDExperimentConfig.from_dict(
            {
                "output_dir": str(tmp_path / "results"),
                "draft_models": ["draft-a"],
                "experiment_modes": ["dssd"],
                "edge": {"port": 8000},
                "requests": [
                    {
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 4,
                    }
                ],
            }
        )


def test_experiment_config_rejects_dssd_without_draft_models(tmp_path: Path):
    with pytest.raises(ValueError, match="draft_models"):
        DSSDExperimentConfig.from_dict(
            {
                "output_dir": str(tmp_path / "results"),
                "target_models": ["target-a"],
                "experiment_modes": ["dssd"],
                "requests": [
                    {
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 4,
                    }
                ],
            }
        )


def test_harness_builds_expected_commands_and_envs(tmp_path: Path):
    config = DSSDExperimentConfig.from_dict(
        {
            "output_dir": str(tmp_path / "results"),
            "draft_models": ["draft-a"],
            "target_models": ["target-a"],
            "experiment_modes": ["dssd", "target-baseline"],
            "gammas": [4],
            "network_profiles": [{"name": "wan", "latency_ms": 20.0}],
            "edge": {
                "port": 8000,
                "cuda_visible_devices": "1",
                "extra_args": ["--max-num-seqs", "64"],
            },
            "verifier": {
                "port": 9001,
                "cuda_visible_devices": "0",
                "extra_args": ["--max-model-len", "32768"],
            },
            "requests": [
                {
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": 4,
                    "temperature": 0.7,
                }
            ],
        }
    )
    dssd_case = next(
        case for case in config.expand_cases() if case.experiment_mode == "dssd"
    )
    baseline_case = next(
        case
        for case in config.expand_cases()
        if case.experiment_mode == "target-baseline"
    )
    harness = DSSDExperimentHarness(config, python_bin="python3")
    trace_path = tmp_path / "trace.jsonl"

    verifier_cmd = harness.build_verifier_command(dssd_case)
    edge_cmd = harness.build_edge_command(dssd_case, trace_output_path=trace_path)
    baseline_cmd = harness.build_target_baseline_command(baseline_case)

    assert harness.build_verifier_env(dssd_case)["CUDA_VISIBLE_DEVICES"] == "0"
    assert harness.build_edge_env(dssd_case)["CUDA_VISIBLE_DEVICES"] == "1"
    assert harness.build_target_env(baseline_case)["CUDA_VISIBLE_DEVICES"] == "0"
    assert verifier_cmd[:4] == [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
    ]
    assert "--no-async-scheduling" in verifier_cmd
    assert "--no-async-scheduling" in edge_cmd
    assert "--no-async-scheduling" not in baseline_cmd

    verifier_cfg = json.loads(verifier_cmd[verifier_cmd.index("--dssd-config") + 1])
    edge_cfg = json.loads(edge_cmd[edge_cmd.index("--dssd-config") + 1])

    assert verifier_cfg == {"enabled": True, "role": "verifier", "gamma": 4}
    assert edge_cfg["verifier_url"] == "http://127.0.0.1:9001"
    assert edge_cfg["network_simulation"] == {
        "latency_ms": 20.0,
        "bandwidth_mbps": None,
        "jitter_ms": 0.0,
    }
    assert edge_cfg["experiment_result_path"] == str(trace_path)
    assert "--dssd-config" not in baseline_cmd
    assert baseline_cmd[baseline_cmd.index("--model") + 1] == "target-a"


def test_bench_dssd_parser_accepts_experiment_config_flag(tmp_path: Path):
    config_path = tmp_path / "experiment.json"
    config_path.write_text("{}", encoding="utf-8")

    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest="subparser")
    BenchmarkSubcommand().subparser_init(subparsers)

    args = parser.parse_args(
        ["bench", "dssd", "--experiment-config", str(config_path)]
    )

    assert args.subparser == "bench"
    assert args.bench_type == "dssd"
    assert args.experiment_config == str(config_path)


def test_http_dssd_transport_disables_environment_proxies():
    transport = HTTPDSSDTransport("http://127.0.0.1:9001")
    try:
        assert transport._client._trust_env is False
    finally:
        asyncio.run(transport.aclose())


def test_harness_smoke_run_writes_summary_and_trace_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    verifier_port = _open_port()
    edge_port = _open_port()
    baseline_port = _open_port()
    fake_server = tmp_path / "fake_dssd_server.py"
    fake_server.write_text(
        textwrap.dedent(
            """
            import argparse
            import json
            import time
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
            from pathlib import Path

            parser = argparse.ArgumentParser()
            parser.add_argument("--port", type=int, required=True)
            parser.add_argument("--trace-path", type=str, default=None)
            parser.add_argument("--delay-ms", type=float, default=0.0)
            args = parser.parse_args()

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path != "/health":
                        self.send_response(404)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b"ok")

                def do_POST(self):
                    if self.path != "/v1/chat/completions":
                        self.send_response(404)
                        self.end_headers()
                        return
                    length = int(self.headers.get("Content-Length", "0"))
                    self.rfile.read(length)
                    if args.delay_ms > 0:
                        time.sleep(args.delay_ms / 1000.0)
                    if args.trace_path:
                        trace_path = Path(args.trace_path)
                        trace_path.parent.mkdir(parents=True, exist_ok=True)
                        with trace_path.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps({"request": {"request_id": "req-1"}, "rounds": [], "run": {"mode": "dssd"}}) + "\\n")
                    body = {
                        "id": "chatcmpl-fake",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "fake-model",
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": "1 2 3"},
                            "finish_reason": "stop",
                        }],
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 3,
                            "total_tokens": 6,
                        },
                    }
                    encoded = json.dumps(body).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(encoded)))
                    self.end_headers()
                    self.wfile.write(encoded)

                def log_message(self, format, *args):
                    return

            server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
            server.serve_forever()
            """
        ),
        encoding="utf-8",
    )

    config = DSSDExperimentConfig.from_dict(
        {
            "output_dir": str(tmp_path / "results"),
            "draft_models": ["draft-a"],
            "target_models": ["target-a"],
            "experiment_modes": ["dssd", "target-baseline"],
            "gammas": [4],
            "network_profiles": [{"name": "lan", "latency_ms": 0.0}],
            "edge": {"host": "127.0.0.1", "port": edge_port},
            "verifier": {"host": "127.0.0.1", "port": verifier_port},
            "target": {"host": "127.0.0.1", "port": baseline_port},
            "requests": [
                {
                    "request_id": "req-1",
                    "messages": [{"role": "user", "content": "count"}],
                    "max_tokens": 4,
                }
            ],
        }
    )
    harness = DSSDExperimentHarness(config, python_bin=sys.executable)

    def _build_verifier_command(_case):
        return [sys.executable, str(fake_server), "--port", str(verifier_port)]

    def _build_edge_command(_case, *, trace_output_path=None):
        cmd = [
            sys.executable,
            str(fake_server),
            "--port",
            str(edge_port),
            "--delay-ms",
            "10",
        ]
        if trace_output_path is not None:
            cmd.extend(["--trace-path", str(trace_output_path)])
        return cmd

    def _build_target_baseline_command(_case):
        return [
            sys.executable,
            str(fake_server),
            "--port",
            str(baseline_port),
            "--delay-ms",
            "40",
        ]

    monkeypatch.setattr(harness, "build_verifier_command", _build_verifier_command)
    monkeypatch.setattr(harness, "build_edge_command", _build_edge_command)
    monkeypatch.setattr(
        harness, "build_target_baseline_command", _build_target_baseline_command
    )

    results = asyncio.run(harness.run_all())

    assert len(results) == 2
    summary_csv = Path(config.output_dir) / "summary.csv"
    summary_json = Path(config.output_dir) / "summary.json"
    runs_json = Path(config.output_dir) / "runs.json"
    assert summary_csv.exists()
    assert summary_json.exists()
    assert runs_json.exists()

    with summary_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    by_mode = {row["experiment_mode"]: row for row in rows}
    assert by_mode["dssd"]["status"] == "ok"
    assert by_mode["target-baseline"]["status"] == "ok"
    assert by_mode["dssd"]["trace_records"] == "1"
    assert float(by_mode["dssd"]["speedup_vs_target_baseline"]) > 1.0

    manifest = json.loads(runs_json.read_text(encoding="utf-8"))
    assert len(manifest) == 2
    assert {item["status"] for item in manifest} == {"ok"}

    dssd_manifest = next(
        item for item in manifest if item["case"]["experiment_mode"] == "dssd"
    )
    baseline_manifest = next(
        item
        for item in manifest
        if item["case"]["experiment_mode"] == "target-baseline"
    )
    assert baseline_manifest["trace_path"] is None

    trace_path = Path(dssd_manifest["trace_path"])
    assert trace_path.exists()
    trace_rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert trace_rows[0]["request"]["request_id"] == "req-1"
