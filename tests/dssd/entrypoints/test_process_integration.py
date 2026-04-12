# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def test_edge_and_verifier_entrypoints_run_end_to_end_in_subprocesses(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    ready_file = tmp_path / "verifier-ready.json"
    env = os.environ | {
        "PYTHONPATH": str(repo_root),
        "PYTHONUNBUFFERED": "1",
    }
    verifier_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm.dssd.entrypoints.verifier_server",
            "--service-factory",
            "tests.entrypoints.dssd_fake_factories.build_verifier_service",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--ready-file",
            str(ready_file),
        ],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        server_url = _wait_for_server_url(ready_file, verifier_proc)
        edge_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "vllm.dssd.entrypoints.edge_runner",
                "--service-factory",
                "tests.entrypoints.dssd_fake_factories.build_edge_service",
                "--verifier-url",
                server_url,
                "--req-id",
                "req-1",
                "--prompt-token-ids",
                "1,2,3",
                "--max-tokens",
                "4",
                "--temperature",
                "0.0",
                "--eos-token-id",
                "2",
                "--gamma",
                "2",
            ],
            cwd=repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=30,
        )
        assert edge_result.returncode == 0, edge_result.stderr
        output_line = edge_result.stdout.strip().splitlines()[-1]
        assert json.loads(output_line) == {
            "req_id": "req-1",
            "output_ids": [17, 19, 20, 21],
        }
    finally:
        verifier_proc.send_signal(signal.SIGINT)
        verifier_stdout, verifier_stderr = verifier_proc.communicate(timeout=10)
        assert verifier_proc.returncode == 0, (
            verifier_stdout,
            verifier_stderr,
        )


def test_edge_runner_exits_nonzero_when_verifier_round_fails(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    ready_file = tmp_path / "verifier-ready.json"
    env = os.environ | {
        "PYTHONPATH": str(repo_root),
        "PYTHONUNBUFFERED": "1",
    }
    verifier_proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm.dssd.entrypoints.verifier_server",
            "--service-factory",
            (
                "tests.entrypoints.dssd_fake_factories."
                "build_failing_verify_verifier_service"
            ),
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--ready-file",
            str(ready_file),
        ],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        server_url = _wait_for_server_url(ready_file, verifier_proc)
        edge_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "vllm.dssd.entrypoints.edge_runner",
                "--service-factory",
                "tests.entrypoints.dssd_fake_factories.build_edge_service",
                "--verifier-url",
                server_url,
                "--req-id",
                "req-1",
                "--prompt-token-ids",
                "1,2,3",
                "--max-tokens",
                "4",
                "--temperature",
                "0.0",
                "--eos-token-id",
                "2",
                "--gamma",
                "2",
            ],
            cwd=repo_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=30,
        )
        assert edge_result.returncode != 0
        assert "verify failed" in edge_result.stderr
    finally:
        _stop_process(verifier_proc)


def test_edge_runner_exits_nonzero_when_verifier_is_unreachable() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ | {
        "PYTHONPATH": str(repo_root),
        "PYTHONUNBUFFERED": "1",
    }
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        host, port = sock.getsockname()

    edge_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "vllm.dssd.entrypoints.edge_runner",
            "--service-factory",
            "tests.entrypoints.dssd_fake_factories.build_edge_service",
            "--verifier-url",
            f"http://{host}:{port}",
            "--req-id",
            "req-1",
            "--prompt-token-ids",
            "1,2,3",
            "--max-tokens",
            "4",
            "--temperature",
            "0.0",
            "--eos-token-id",
            "2",
            "--gamma",
            "2",
        ],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=30,
    )

    assert edge_result.returncode != 0
    assert "verifier request failed" in edge_result.stderr


def _wait_for_server_url(
    ready_file: Path,
    proc: subprocess.Popen[str],
    *,
    timeout_s: float = 30.0,
) -> str:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise AssertionError(
                "verifier server exited early\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )
        if ready_file.exists():
            return json.loads(ready_file.read_text())["server_url"]
        time.sleep(0.05)
    raise AssertionError("verifier ready file was not created in time")


def _stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        proc.communicate()
        return
    proc.terminate()
    try:
        proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate(timeout=10)
