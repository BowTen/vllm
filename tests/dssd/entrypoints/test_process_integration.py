# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib import error as urllib_error
from urllib import request as urllib_request

import pytest


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


def _start_edge_server(
    tmp_path: Path,
    *,
    service_factory: str,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.Popen[str], str]:
    repo_root = Path(__file__).resolve().parents[3]
    ready_file = tmp_path / "edge-ready.json"
    env = os.environ | {
        "PYTHONPATH": str(repo_root),
        "PYTHONUNBUFFERED": "1",
        "VLLM_DSSD_EDGE_SERVER_RAW_SAMPLING_PARAMS": "1",
    }
    if extra_env:
        env.update(extra_env)
    edge_server_path = (
        repo_root / "vllm" / "dssd" / "entrypoints" / "edge_server.py"
    )
    proc = subprocess.Popen(
        [
            sys.executable,
            str(edge_server_path),
            "--service-factory",
            service_factory,
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--ready-file",
            str(ready_file),
            "--verifier-url",
            "http://127.0.0.1:18021",
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
    )
    return proc, _wait_for_server_url(ready_file, proc)


def _post_edge_generate(
    server_url: str,
    *,
    req_id: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    temperature: float = 0.0,
) -> dict:
    body = json.dumps(
        {
            "req_id": req_id,
            "prompt_token_ids": prompt_token_ids,
            "sampling_params": {
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            "lora_request": None,
        }
    ).encode("utf-8")
    request = urllib_request.Request(
        url=f"{server_url}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def test_edge_server_process_handles_repeated_sequential_requests(
    tmp_path: Path,
) -> None:
    proc, server_url = _start_edge_server(
        tmp_path,
        service_factory=(
            "tests.entrypoints.dssd_fake_factories."
            "build_persistent_edge_service"
        ),
    )
    try:
        first = _post_edge_generate(
            server_url,
            req_id="req-1",
            prompt_token_ids=[1, 2],
            max_tokens=4,
        )
        second = _post_edge_generate(
            server_url,
            req_id="req-2",
            prompt_token_ids=[3, 4],
            max_tokens=4,
        )
    finally:
        _stop_process(proc)

    assert first == {"req_id": "req-1", "output_ids": [17, 1]}
    assert second == {"req_id": "req-2", "output_ids": [17, 2]}


def test_edge_server_serializes_second_request_until_first_finishes(
    tmp_path: Path,
) -> None:
    started_file = tmp_path / "started"
    release_file = tmp_path / "release"
    proc, server_url = _start_edge_server(
        tmp_path,
        service_factory=(
            "tests.entrypoints.dssd_fake_factories."
            "build_blocking_edge_service"
        ),
        extra_env={
            "DSSD_EDGE_STARTED_FILE": str(started_file),
            "DSSD_EDGE_RELEASE_FILE": str(release_file),
        },
    )
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(
                _post_edge_generate,
                server_url,
                req_id="req-1",
                prompt_token_ids=[1],
                max_tokens=4,
            )
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not started_file.exists():
                time.sleep(0.05)
            assert started_file.exists()

            second_future = executor.submit(
                _post_edge_generate,
                server_url,
                req_id="req-2",
                prompt_token_ids=[2],
                max_tokens=4,
            )
            time.sleep(0.2)
            assert not second_future.done()

            release_file.write_text("ok")

            first = first_future.result(timeout=10)
            second = second_future.result(timeout=10)
    finally:
        _stop_process(proc)

    assert first == {"req_id": "req-1", "output_ids": [9001]}
    assert second == {"req_id": "req-2", "output_ids": [9002]}


def test_edge_server_returns_structured_500_on_generate_failure(
    tmp_path: Path,
) -> None:
    proc, server_url = _start_edge_server(
        tmp_path,
        service_factory=(
            "tests.entrypoints.dssd_fake_factories."
            "build_failing_edge_service"
        ),
    )
    try:
        request = urllib_request.Request(
            url=f"{server_url}/generate",
            data=json.dumps(
                {
                    "req_id": "req-1",
                    "prompt_token_ids": [1],
                    "sampling_params": {
                        "max_tokens": 4,
                        "temperature": 0.0,
                    },
                    "lora_request": None,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib_error.HTTPError) as exc_info:
            urllib_request.urlopen(request, timeout=30)
    finally:
        _stop_process(proc)

    payload = json.loads(exc_info.value.read().decode("utf-8"))
    assert exc_info.value.code == 500
    assert payload["error_type"] == "RuntimeError"
    assert "edge generate failed" in payload["error"]


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
