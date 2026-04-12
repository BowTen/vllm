# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from vllm.dssd.protocol import CloseSessionAck, VerifyRoundRequest
from vllm.sampling_params import SamplingParams


def _start_verifier_server(
    tmp_path: Path,
    *,
    service_factory: str = (
        "tests.entrypoints.dssd_fake_factories.build_verifier_service"
    ),
) -> tuple[subprocess.Popen[str], str]:
    ready_file = tmp_path / "verifier-ready.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm.dssd.entrypoints.verifier_server",
            "--service-factory",
            service_factory,
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--ready-file",
            str(ready_file),
        ],
        cwd=Path(__file__).resolve().parents[3],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            raise AssertionError(
                "verifier server exited early\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )
        if ready_file.exists():
            payload = json.loads(ready_file.read_text())
            return proc, payload["server_url"]
        time.sleep(0.05)
    proc.terminate()
    stdout, stderr = proc.communicate(timeout=5)
    raise AssertionError(
        "verifier server did not become ready\n"
        f"stdout:\n{stdout}\n"
        f"stderr:\n{stderr}"
    )


def _stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate(timeout=5)


def test_http_verifier_transport_round_trips_across_server_process(
    tmp_path: Path,
) -> None:
    from vllm.dssd.transport.http_verifier_transport import HTTPVerifierTransport

    proc, server_url = _start_verifier_server(tmp_path)
    try:
        transport = HTTPVerifierTransport(server_url=server_url)

        opened = transport.open_session(
            req_id="req-1",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=8),
        )
        response = transport.verify_round(
            VerifyRoundRequest(
                req_id="req-1",
                committed_token_id=17,
                draft_token_ids=[19, 20],
                draft_q_values=[0.6, 0.4],
            )
        )
        ack = transport.close_session("req-1")
    finally:
        _stop_process(proc)

    assert opened.bootstrap_token_id == 17
    assert response.accepted_len == 2
    assert response.bonus_token_id == 21
    assert ack == CloseSessionAck(req_id="req-1")


def test_edge_runner_cli_can_generate_via_http_verifier_server(tmp_path: Path) -> None:
    proc, server_url = _start_verifier_server(tmp_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    try:
        result = subprocess.run(
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
                "1,3",
                "--max-tokens",
                "4",
                "--temperature",
                "0.0",
                "--eos-token-id",
                "21",
                "--gamma",
                "2",
            ],
            cwd=Path(__file__).resolve().parents[3],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        _stop_process(proc)

    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload == {"req_id": "req-1", "output_ids": [17, 19, 20, 21]}


def test_http_verifier_transport_surfaces_remote_verify_errors(
    tmp_path: Path,
) -> None:
    from vllm.dssd.transport.http_verifier_transport import HTTPVerifierTransport

    proc, server_url = _start_verifier_server(
        tmp_path,
        service_factory=(
            "tests.entrypoints.dssd_fake_factories."
            "build_failing_verify_verifier_service"
        ),
    )
    try:
        transport = HTTPVerifierTransport(server_url=server_url)
        transport.open_session(
            req_id="req-1",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=8),
        )
        with pytest.raises(RuntimeError, match="verify failed"):
            transport.verify_round(
                VerifyRoundRequest(
                    req_id="req-1",
                    committed_token_id=17,
                    draft_token_ids=[19, 20],
                    draft_q_values=[0.6, 0.4],
                )
            )
        ack = transport.close_session("req-1")
    finally:
        _stop_process(proc)

    assert ack == CloseSessionAck(req_id="req-1")


def test_http_verifier_transport_surfaces_unreachable_server_errors() -> None:
    from vllm.dssd.transport.http_verifier_transport import HTTPVerifierTransport

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        host, port = sock.getsockname()

    transport = HTTPVerifierTransport(
        server_url=f"http://{host}:{port}",
        timeout_s=0.2,
    )
    with pytest.raises(RuntimeError, match="verifier request failed"):
        transport.open_session(
            req_id="req-1",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=8),
        )
