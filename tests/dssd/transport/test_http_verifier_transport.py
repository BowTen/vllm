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
import torch

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


def test_http_verifier_transport_generate_uses_verifier_target_only(
    tmp_path: Path,
) -> None:
    from vllm.dssd.transport.http_verifier_transport import HTTPVerifierTransport

    proc, server_url = _start_verifier_server(tmp_path)
    try:
        transport = HTTPVerifierTransport(server_url=server_url)
        output_ids = transport.generate(
            req_id="target-1",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=4, temperature=0.0),
            lora_request=None,
        )
    finally:
        _stop_process(proc)

    assert output_ids == [31, 32, 33]


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


def test_http_verifier_transport_round_trips_binary_reject_response_across_server_process(
    tmp_path: Path,
) -> None:
    from vllm.dssd.transport.http_verifier_transport import HTTPVerifierTransport

    proc, server_url = _start_verifier_server(
        tmp_path,
        service_factory=(
            "tests.entrypoints.dssd_fake_factories."
            "build_rejecting_verify_verifier_service"
        ),
    )
    try:
        transport = HTTPVerifierTransport(server_url=server_url)
        transport.open_session(
            req_id="req-1",
            prompt_token_ids=[1, 2, 3],
            sampling_params=SamplingParams(max_tokens=8),
        )
        response = transport.verify_round(
            VerifyRoundRequest(
                req_id="req-1",
                committed_token_id=17,
                draft_token_ids=[19],
                draft_q_values=[0.6],
            )
        )
        ack = transport.close_session("req-1")
    finally:
        _stop_process(proc)

    assert response.accepted_len == 0
    assert response.bonus_token_id is None
    assert torch.equal(
        response.rejected_target_logits,
        torch.tensor([0.5, -0.25, 1.25], dtype=torch.float32),
    )
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


def test_http_verifier_transport_uses_serialized_body_sizes_for_network_delay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.dssd.transport.http_verifier_transport import HTTPVerifierTransport

    class RecordingNetwork:
        def __init__(self) -> None:
            self.payload_bytes: list[int] = []

        def simulate_transfer_bytes(self, payload_bytes: int) -> None:
            self.payload_bytes.append(payload_bytes)

    class FakeHTTPResponse:
        def __init__(self, body: bytes) -> None:
            self._body = body
            self.headers = {"Content-Type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return self._body

    raw_response = (
        b'{"req_id":"req-1","accepted_len":1,"bonus_token_id":23,'
        b'"rejected_target_logits":null}'
    )
    captured = {}

    class FakeOpener:
        def open(self, request, timeout):
            captured["request_bytes"] = bytes(request.data)
            return FakeHTTPResponse(raw_response)

    request_network = RecordingNetwork()
    response_network = RecordingNetwork()
    transport = HTTPVerifierTransport(
        server_url="http://127.0.0.1:18021",
        request_network=request_network,
        response_network=response_network,
    )
    monkeypatch.setattr(transport, "_opener", FakeOpener())

    response = transport.verify_round(
        VerifyRoundRequest(
            req_id="req-1",
            committed_token_id=17,
            draft_token_ids=[19, 20],
            draft_q_values=[0.6, 0.4],
        )
    )

    assert response.accepted_len == 1
    assert request_network.payload_bytes == [len(captured["request_bytes"])]
    assert response_network.payload_bytes == [len(raw_response)]


def test_http_verifier_transport_decodes_binary_reject_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.dssd.transport.http_verifier_transport import HTTPVerifierTransport

    logits = torch.tensor([0.5, -0.25, 1.25], dtype=torch.float32)
    metadata = {
        "req_id": "req-1",
        "accepted_len": 0,
        "bonus_token_id": None,
        "dtype": "float32",
        "shape": [3],
    }
    metadata_bytes = json.dumps(metadata).encode("utf-8")
    response_body = (
        len(metadata_bytes).to_bytes(4, "little")
        + metadata_bytes
        + logits.numpy().tobytes()
    )

    class FakeHTTPResponse:
        def __init__(self, body: bytes) -> None:
            self._body = body
            self.headers = {"Content-Type": "application/octet-stream"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return self._body

    class FakeOpener:
        def open(self, request, timeout):
            return FakeHTTPResponse(response_body)

    transport = HTTPVerifierTransport(server_url="http://127.0.0.1:18021")
    monkeypatch.setattr(transport, "_opener", FakeOpener())

    response = transport.verify_round(
        VerifyRoundRequest(
            req_id="req-1",
            committed_token_id=17,
            draft_token_ids=[19],
            draft_q_values=[0.6],
        )
    )

    assert response.accepted_len == 0
    assert response.bonus_token_id is None
    assert torch.equal(response.rejected_target_logits, logits)


def test_http_verifier_transport_rejects_malformed_binary_reject_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.dssd.transport.http_verifier_transport import HTTPVerifierTransport

    class FakeHTTPResponse:
        def __init__(self, body: bytes) -> None:
            self._body = body
            self.headers = {"Content-Type": "application/octet-stream"}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return self._body

    class FakeOpener:
        def open(self, request, timeout):
            return FakeHTTPResponse(b"\x08\x00\x00")

    transport = HTTPVerifierTransport(server_url="http://127.0.0.1:18021")
    monkeypatch.setattr(transport, "_opener", FakeOpener())

    with pytest.raises(RuntimeError, match="malformed binary verify_round response"):
        transport.verify_round(
            VerifyRoundRequest(
                req_id="req-1",
                committed_token_id=17,
                draft_token_ids=[19],
                draft_q_values=[0.6],
            )
        )
