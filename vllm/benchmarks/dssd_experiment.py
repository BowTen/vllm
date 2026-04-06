# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx


def _ensure_list(value: Any, *, default: list[Any] | None = None) -> list[Any]:
    if value is None:
        return list(default or [])
    if isinstance(value, list):
        return value
    return [value]


def _sanitize_name(value: str | None) -> str:
    if value is None:
        return "none"
    safe = []
    for char in Path(value).name:
        safe.append(char if char.isalnum() else "-")
    compact = "".join(safe).strip("-")
    return compact or "value"


@dataclass(frozen=True)
class ExperimentNetworkProfile:
    name: str
    latency_ms: float = 0.0
    bandwidth_mbps: float | None = None
    jitter_ms: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentNetworkProfile":
        name = str(data.get("name") or "default")
        latency_ms = float(data.get("latency_ms", 0.0))
        jitter_ms = float(data.get("jitter_ms", 0.0))
        bandwidth = data.get("bandwidth_mbps")
        bandwidth_mbps = None if bandwidth is None else float(bandwidth)
        if latency_ms < 0 or jitter_ms < 0:
            raise ValueError("network profile latency_ms and jitter_ms must be >= 0")
        if bandwidth_mbps is not None and bandwidth_mbps <= 0:
            raise ValueError("network profile bandwidth_mbps must be > 0")
        return cls(
            name=name,
            latency_ms=latency_ms,
            bandwidth_mbps=bandwidth_mbps,
            jitter_ms=jitter_ms,
        )


@dataclass(frozen=True)
class ExperimentServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    cuda_visible_devices: str | None = None
    extra_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    startup_timeout_s: float = 300.0
    ready_poll_interval_s: float = 1.0

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
        *,
        default_port: int,
    ) -> "ExperimentServerConfig":
        raw = data or {}
        return cls(
            host=str(raw.get("host", "127.0.0.1")),
            port=int(raw.get("port", default_port)),
            cuda_visible_devices=(
                None
                if raw.get("cuda_visible_devices") is None
                else str(raw["cuda_visible_devices"])
            ),
            extra_args=[str(item) for item in raw.get("extra_args", [])],
            env={str(k): str(v) for k, v in dict(raw.get("env", {})).items()},
            startup_timeout_s=float(raw.get("startup_timeout_s", 300.0)),
            ready_poll_interval_s=float(raw.get("ready_poll_interval_s", 1.0)),
        )


@dataclass(frozen=True)
class ExperimentRequest:
    messages: list[dict[str, Any]]
    max_tokens: int
    request_id: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    stream: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentRequest":
        messages = list(data.get("messages") or [])
        if not messages:
            raise ValueError("experiment request requires non-empty messages")
        max_tokens = int(data.get("max_tokens", 0))
        if max_tokens <= 0:
            raise ValueError("experiment request max_tokens must be > 0")
        top_k = data.get("top_k")
        return cls(
            messages=messages,
            max_tokens=max_tokens,
            request_id=(
                None if data.get("request_id") is None else str(data["request_id"])
            ),
            temperature=(
                None
                if data.get("temperature") is None
                else float(data["temperature"])
            ),
            top_p=None if data.get("top_p") is None else float(data["top_p"]),
            top_k=None if top_k is None else int(top_k),
            min_p=None if data.get("min_p") is None else float(data["min_p"]),
            stream=bool(data.get("stream", False)),
        )

    def to_payload(self, *, model: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": self.messages,
            "max_tokens": self.max_tokens,
            "stream": self.stream,
        }
        if self.request_id is not None:
            payload["request_id"] = self.request_id
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.top_k is not None:
            payload["top_k"] = self.top_k
        if self.min_p is not None:
            payload["min_p"] = self.min_p
        return payload


@dataclass(frozen=True)
class ResolvedExperimentCase:
    case_id: str
    experiment_mode: str
    draft_model: str | None
    target_model: str
    gamma: int
    network_profile: ExperimentNetworkProfile
    requests: list[ExperimentRequest]
    edge: ExperimentServerConfig
    verifier: ExperimentServerConfig | None
    target: ExperimentServerConfig | None


@dataclass(frozen=True)
class DSSDExperimentConfig:
    output_dir: str
    draft_models: list[str]
    target_models: list[str]
    experiment_modes: list[str]
    gammas: list[int]
    network_profiles: list[ExperimentNetworkProfile]
    requests: list[ExperimentRequest]
    edge: ExperimentServerConfig
    verifier: ExperimentServerConfig | None
    target: ExperimentServerConfig | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DSSDExperimentConfig":
        output_dir = data.get("output_dir")
        if output_dir is None:
            raise ValueError("output_dir is required")

        draft_models = [str(item) for item in _ensure_list(data.get("draft_models"))]
        experiment_modes = [
            str(item)
            for item in _ensure_list(
                data.get("experiment_modes"), default=["dssd"]
            )
        ]
        invalid_modes = set(experiment_modes) - {"dssd", "target-baseline"}
        if invalid_modes:
            raise ValueError(f"Unsupported experiment_modes: {sorted(invalid_modes)}")

        if "dssd" in experiment_modes and not draft_models:
            raise ValueError("draft_models is required")

        target_models = [str(item) for item in _ensure_list(data.get("target_models"))]
        if not target_models:
            raise ValueError("target_models is required")

        gammas = [int(item) for item in _ensure_list(data.get("gammas"), default=[4])]
        if any(gamma < 1 for gamma in gammas):
            raise ValueError("gammas must be >= 1")

        raw_profiles = _ensure_list(
            data.get("network_profiles"),
            default=[{"name": "default", "latency_ms": 0.0}],
        )
        network_profiles = [
            ExperimentNetworkProfile.from_dict(dict(item)) for item in raw_profiles
        ]

        requests = _load_requests(data)
        edge = ExperimentServerConfig.from_dict(data.get("edge"), default_port=8000)
        verifier = None
        if "dssd" in experiment_modes:
            verifier = ExperimentServerConfig.from_dict(
                data.get("verifier"),
                default_port=9001,
            )
        target = None
        if "target-baseline" in experiment_modes:
            target_source = data.get("target")
            if target_source is None and verifier is not None:
                target = ExperimentServerConfig(
                    host=verifier.host,
                    port=verifier.port,
                    cuda_visible_devices=verifier.cuda_visible_devices,
                    extra_args=list(verifier.extra_args),
                    env=dict(verifier.env),
                    startup_timeout_s=verifier.startup_timeout_s,
                    ready_poll_interval_s=verifier.ready_poll_interval_s,
                )
            else:
                target = ExperimentServerConfig.from_dict(
                    target_source,
                    default_port=9002,
                )

        return cls(
            output_dir=str(output_dir),
            draft_models=draft_models,
            target_models=target_models,
            experiment_modes=experiment_modes,
            gammas=gammas,
            network_profiles=network_profiles,
            requests=requests,
            edge=edge,
            verifier=verifier,
            target=target,
        )

    @classmethod
    def read_json(cls, path: str | os.PathLike[str]) -> "DSSDExperimentConfig":
        with open(path, encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def expand_cases(self) -> list[ResolvedExperimentCase]:
        cases: list[ResolvedExperimentCase] = []
        if "dssd" in self.experiment_modes:
            for draft_model in self.draft_models:
                for target_model in self.target_models:
                    for gamma in self.gammas:
                        for network_profile in self.network_profiles:
                            cases.append(
                                ResolvedExperimentCase(
                                    case_id="-".join(
                                        [
                                            "dssd",
                                            _sanitize_name(draft_model),
                                            _sanitize_name(target_model),
                                            f"g{gamma}",
                                            _sanitize_name(network_profile.name),
                                        ]
                                    ),
                                    experiment_mode="dssd",
                                    draft_model=draft_model,
                                    target_model=target_model,
                                    gamma=gamma,
                                    network_profile=network_profile,
                                    requests=self.requests,
                                    edge=self.edge,
                                    verifier=self.verifier,
                                    target=self.target,
                                )
                            )
        if "target-baseline" in self.experiment_modes:
            for target_model in self.target_models:
                for gamma in self.gammas:
                    for network_profile in self.network_profiles:
                        cases.append(
                            ResolvedExperimentCase(
                                case_id="-".join(
                                    [
                                        "target-baseline",
                                        _sanitize_name(target_model),
                                        f"g{gamma}",
                                        _sanitize_name(network_profile.name),
                                    ]
                                ),
                                experiment_mode="target-baseline",
                                draft_model=None,
                                target_model=target_model,
                                gamma=gamma,
                                network_profile=network_profile,
                                requests=self.requests,
                                edge=self.edge,
                                verifier=self.verifier,
                                target=self.target,
                            )
                        )
        return cases


def _load_requests(data: dict[str, Any]) -> list[ExperimentRequest]:
    if "requests" in data:
        raw_requests = list(data["requests"])
        return [ExperimentRequest.from_dict(dict(item)) for item in raw_requests]

    requests_path = data.get("requests_path")
    if requests_path is None:
        raise ValueError("Either requests or requests_path is required")

    path = Path(str(requests_path))
    if path.suffix == ".jsonl":
        items = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                stripped = line.strip()
                if stripped:
                    items.append(json.loads(stripped))
    else:
        with path.open(encoding="utf-8") as stream:
            items = json.load(stream)
    return [ExperimentRequest.from_dict(dict(item)) for item in items]


@dataclass(frozen=True)
class ExperimentRequestResult:
    request_id: str
    status: str
    latency_ms: float
    completion_tokens: int
    total_tokens: int
    error: str | None = None


@dataclass(frozen=True)
class ExperimentRunResult:
    case: ResolvedExperimentCase
    status: str
    edge_command: list[str]
    verifier_command: list[str] | None
    request_results: list[ExperimentRequestResult]
    trace_path: str | None
    trace_records: int
    error: str | None = None

    def avg_latency_ms(self) -> float:
        if not self.request_results:
            return 0.0
        return sum(item.latency_ms for item in self.request_results) / len(
            self.request_results
        )

    def summary_row(self, *, speedup_vs_target_baseline: float | None = None) -> dict[str, str]:
        avg_latency = self.avg_latency_ms()
        return {
            "case_id": self.case.case_id,
            "status": self.status,
            "experiment_mode": self.case.experiment_mode,
            "draft_model": self.case.draft_model or "",
            "target_model": self.case.target_model,
            "gamma": str(self.case.gamma),
            "network_profile": self.case.network_profile.name,
            "requests": str(len(self.request_results)),
            "avg_latency_ms": f"{avg_latency:.3f}",
            "speedup_vs_target_baseline": (
                ""
                if speedup_vs_target_baseline is None
                else f"{speedup_vs_target_baseline:.6f}"
            ),
            "trace_records": str(self.trace_records),
            "trace_path": self.trace_path or "",
            "error": self.error or "",
        }

    def record(self) -> dict[str, Any]:
        return {
            "case_id": self.case.case_id,
            "status": self.status,
            "error": self.error,
            "case": {
                "experiment_mode": self.case.experiment_mode,
                "draft_model": self.case.draft_model,
                "target_model": self.case.target_model,
                "gamma": self.case.gamma,
                "network_profile": asdict(self.case.network_profile),
            },
            "edge_command": self.edge_command,
            "verifier_command": self.verifier_command,
            "request_results": [asdict(item) for item in self.request_results],
            "trace_path": self.trace_path,
            "trace_records": self.trace_records,
        }


class ManagedServerProcess:
    def __init__(
        self,
        cmd: list[str],
        *,
        env_overrides: dict[str, str] | None = None,
    ) -> None:
        merged_env = os.environ.copy()
        if env_overrides:
            merged_env.update(env_overrides)
        self._cmd = cmd
        self._env = merged_env
        self.proc: subprocess.Popen[str] | None = None

    def start(self) -> None:
        self.proc = subprocess.Popen(
            self._cmd,
            env=self._env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)


class DSSDExperimentHarness:
    def __init__(
        self,
        config: DSSDExperimentConfig,
        *,
        python_bin: str | None = None,
    ) -> None:
        self.config = config
        self.python_bin = python_bin or sys.executable

    def build_verifier_env(self, case: ResolvedExperimentCase) -> dict[str, str]:
        env = dict(case.verifier.env if case.verifier is not None else {})
        if case.verifier is not None and case.verifier.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = case.verifier.cuda_visible_devices
        return env

    def build_edge_env(self, case: ResolvedExperimentCase) -> dict[str, str]:
        env = dict(case.edge.env)
        if case.edge.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = case.edge.cuda_visible_devices
        return env

    def build_verifier_command(
        self,
        case: ResolvedExperimentCase,
    ) -> list[str] | None:
        if case.experiment_mode != "dssd":
            return None

        cmd = [
            self.python_bin,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            case.target_model,
            "--host",
            case.verifier.host if case.verifier is not None else "127.0.0.1",
            "--port",
            str(case.verifier.port if case.verifier is not None else 9001),
            *(
                case.verifier.extra_args
                if case.verifier is not None
                else []
            ),
        ]
        _append_flag_once(cmd, "--no-async-scheduling")
        cmd.extend(
            [
                "--dssd-config",
                json.dumps(
                    {
                        "enabled": True,
                        "role": "verifier",
                        "gamma": case.gamma,
                    }
                ),
            ]
        )
        return cmd

    def build_edge_command(
        self,
        case: ResolvedExperimentCase,
        *,
        trace_output_path: Path | None,
    ) -> list[str]:
        assert case.draft_model is not None
        cmd = [
            self.python_bin,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            case.draft_model,
            "--host",
            case.edge.host,
            "--port",
            str(case.edge.port),
            *case.edge.extra_args,
        ]
        if case.experiment_mode == "dssd":
            _append_flag_once(cmd, "--no-async-scheduling")

        dssd_config: dict[str, Any] = {
            "enabled": True,
            "role": "edge",
            "gamma": case.gamma,
            "experiment_mode": case.experiment_mode,
            "network_simulation": {
                "latency_ms": case.network_profile.latency_ms,
                "bandwidth_mbps": case.network_profile.bandwidth_mbps,
                "jitter_ms": case.network_profile.jitter_ms,
            },
        }
        if case.experiment_mode == "dssd" and case.verifier is not None:
            dssd_config["verifier_url"] = (
                f"http://{case.verifier.host}:{case.verifier.port}"
            )
            if trace_output_path is not None:
                dssd_config["experiment_result_path"] = str(trace_output_path)
        cmd.extend(["--dssd-config", json.dumps(dssd_config)])
        return cmd

    def build_target_env(self, case: ResolvedExperimentCase) -> dict[str, str]:
        env = dict(case.target.env if case.target is not None else {})
        if case.target is not None and case.target.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = case.target.cuda_visible_devices
        return env

    def build_target_baseline_command(
        self,
        case: ResolvedExperimentCase,
    ) -> list[str]:
        assert case.target is not None
        return [
            self.python_bin,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            case.target_model,
            "--host",
            case.target.host,
            "--port",
            str(case.target.port),
            *case.target.extra_args,
        ]

    async def run_all(self) -> list[ExperimentRunResult]:
        results = []
        for case in self.config.expand_cases():
            results.append(await self.run_case(case))
        self.write_artifacts(results)
        return results

    async def run_case(self, case: ResolvedExperimentCase) -> ExperimentRunResult:
        output_root = Path(self.config.output_dir)
        case_dir = output_root / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        trace_output_path = (
            case_dir / "dssd-traces.jsonl"
            if case.experiment_mode == "dssd"
            else None
        )
        verifier_cmd = self.build_verifier_command(case)
        if case.experiment_mode == "dssd":
            serving_cmd = self.build_edge_command(
                case,
                trace_output_path=trace_output_path,
            )
        else:
            serving_cmd = self.build_target_baseline_command(case)
        verifier_proc = (
            ManagedServerProcess(
                verifier_cmd,
                env_overrides=self.build_verifier_env(case),
            )
            if verifier_cmd is not None
            else None
        )
        serving_proc = ManagedServerProcess(
            serving_cmd,
            env_overrides=(
                self.build_edge_env(case)
                if case.experiment_mode == "dssd"
                else self.build_target_env(case)
            ),
        )

        try:
            if verifier_proc is not None:
                verifier_proc.start()
                assert case.verifier is not None
                await self._wait_for_health(
                    f"http://{case.verifier.host}:{case.verifier.port}/health",
                    timeout_s=case.verifier.startup_timeout_s,
                    poll_interval_s=case.verifier.ready_poll_interval_s,
                )

            serving_proc.start()
            server_cfg = case.edge if case.experiment_mode == "dssd" else case.target
            assert server_cfg is not None
            await self._wait_for_health(
                f"http://{server_cfg.host}:{server_cfg.port}/health",
                timeout_s=server_cfg.startup_timeout_s,
                poll_interval_s=server_cfg.ready_poll_interval_s,
            )

            request_results = []
            for index, request in enumerate(case.requests):
                request_results.append(await self._execute_request(case, request, index))
            trace_records = _count_jsonl_records(trace_output_path)
            return ExperimentRunResult(
                case=case,
                status="ok",
                edge_command=serving_cmd,
                verifier_command=verifier_cmd,
                request_results=request_results,
                trace_path=(
                    str(trace_output_path)
                    if trace_output_path is not None and trace_output_path.exists()
                    else None
                ),
                trace_records=trace_records,
            )
        except Exception as exc:
            return ExperimentRunResult(
                case=case,
                status="error",
                edge_command=serving_cmd,
                verifier_command=verifier_cmd,
                request_results=[],
                trace_path=None,
                trace_records=0,
                error=str(exc),
            )
        finally:
            serving_proc.stop()
            if verifier_proc is not None:
                verifier_proc.stop()

    async def _wait_for_health(
        self,
        url: str,
        *,
        timeout_s: float,
        poll_interval_s: float,
    ) -> None:
        deadline = time.perf_counter() + timeout_s
        async with httpx.AsyncClient(trust_env=False, timeout=5.0) as client:
            while time.perf_counter() < deadline:
                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(poll_interval_s)
        raise TimeoutError(f"Timed out waiting for server health: {url}")

    async def _execute_request(
        self,
        case: ResolvedExperimentCase,
        request: ExperimentRequest,
        index: int,
    ) -> ExperimentRequestResult:
        started = time.perf_counter()
        request_model = case.draft_model if case.experiment_mode == "dssd" else case.target_model
        payload = request.to_payload(model=request_model)
        if request.request_id is None:
            payload["request_id"] = f"{case.case_id}-req-{index}"
        server_cfg = case.edge if case.experiment_mode == "dssd" else case.target
        assert server_cfg is not None
        url = f"http://{server_cfg.host}:{server_cfg.port}/v1/chat/completions"
        async with httpx.AsyncClient(trust_env=False, timeout=120.0) as client:
            response = await client.post(url, json=payload)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if response.status_code != 200:
            return ExperimentRequestResult(
                request_id=str(payload["request_id"]),
                status="error",
                latency_ms=latency_ms,
                completion_tokens=0,
                total_tokens=0,
                error=response.text,
            )

        body = response.json()
        usage = body.get("usage", {})
        return ExperimentRequestResult(
            request_id=str(payload["request_id"]),
            status="ok",
            latency_ms=latency_ms,
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
        )

    def write_artifacts(self, results: list[ExperimentRunResult]) -> None:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        baseline_latency_by_key: dict[tuple[str, int, str], float] = {}
        for result in results:
            if result.case.experiment_mode != "target-baseline" or result.status != "ok":
                continue
            baseline_latency_by_key[
                (
                    result.case.target_model,
                    result.case.gamma,
                    result.case.network_profile.name,
                )
            ] = result.avg_latency_ms()

        summary_rows = []
        for result in results:
            speedup = None
            if result.case.experiment_mode == "dssd" and result.status == "ok":
                baseline_latency = baseline_latency_by_key.get(
                    (
                        result.case.target_model,
                        result.case.gamma,
                        result.case.network_profile.name,
                    )
                )
                if baseline_latency and result.avg_latency_ms() > 0:
                    speedup = baseline_latency / result.avg_latency_ms()
            summary_rows.append(
                result.summary_row(speedup_vs_target_baseline=speedup)
            )
        with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
            json.dump(summary_rows, stream, indent=2, ensure_ascii=False)
        with (output_dir / "runs.json").open("w", encoding="utf-8") as stream:
            json.dump(
                [result.record() for result in results],
                stream,
                indent=2,
                ensure_ascii=False,
            )
        if summary_rows:
            with (output_dir / "summary.csv").open(
                "w",
                encoding="utf-8",
                newline="",
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0].keys()))
                writer.writeheader()
                writer.writerows(summary_rows)


def _count_jsonl_records(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _append_flag_once(cmd: list[str], flag: str) -> None:
    if flag not in cmd:
        cmd.append(flag)


def add_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--experiment-config",
        type=str,
        required=True,
        help="Path to a JSON config file describing the DSSD experiment run(s).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional override for the config file output_dir.",
    )


def main(args: argparse.Namespace) -> None:
    config = DSSDExperimentConfig.read_json(args.experiment_config)
    if args.output_dir is not None:
        config = DSSDExperimentConfig(
            output_dir=args.output_dir,
            draft_models=config.draft_models,
            target_models=config.target_models,
            experiment_modes=config.experiment_modes,
            gammas=config.gammas,
            network_profiles=config.network_profiles,
            requests=config.requests,
            edge=config.edge,
            verifier=config.verifier,
            target=config.target,
        )
    harness = DSSDExperimentHarness(config)
    asyncio.run(harness.run_all())
