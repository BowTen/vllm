# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import time

import pytest
import torch

from vllm.dssd.protocol import (
    CloseSessionAck,
    OpenSessionResponse,
    VerifyRoundResponse,
)


def test_fake_network_transfer_time_accounts_for_latency_and_bandwidth() -> None:
    from vllm.dssd.transport.fake_network import FakeNetwork

    network = FakeNetwork(fixed_latency_ms=2.0, bandwidth_bytes_per_s=1000.0)

    assert network.transfer_time_s(payload_bytes=500) == pytest.approx(0.502)


def test_fake_network_estimates_reject_payload_larger_than_bonus() -> None:
    from vllm.dssd.transport.fake_network import FakeNetwork

    network = FakeNetwork(fixed_latency_ms=0.0, bandwidth_bytes_per_s=1e9)
    bonus_bytes = network.estimate_payload_bytes(
        VerifyRoundResponse(req_id="r", accepted_len=1, bonus_token_id=7)
    )
    reject_bytes = network.estimate_payload_bytes(
        VerifyRoundResponse(
            req_id="r",
            accepted_len=0,
            rejected_target_logits=torch.zeros(128, dtype=torch.float32),
        )
    )

    assert reject_bytes > bonus_bytes


def test_fake_network_simulate_transfer_sleeps_for_estimated_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.dssd.transport.fake_network import FakeNetwork

    network = FakeNetwork(fixed_latency_ms=1.5, bandwidth_bytes_per_s=1000.0)
    response = OpenSessionResponse(req_id="r", bootstrap_token_id=9)
    sleep_calls: list[float] = []

    monkeypatch.setattr(time, "sleep", sleep_calls.append)

    network.simulate_transfer(response)

    expected = network.transfer_time_s(
        payload_bytes=network.estimate_payload_bytes(response)
    )
    assert sleep_calls == [pytest.approx(expected)]


def test_fake_network_simulate_transfer_bytes_uses_explicit_payload_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.dssd.transport.fake_network import FakeNetwork

    network = FakeNetwork(fixed_latency_ms=1.5, bandwidth_bytes_per_s=1000.0)
    sleep_calls: list[float] = []

    monkeypatch.setattr(time, "sleep", sleep_calls.append)

    network.simulate_transfer_bytes(7)

    assert sleep_calls == [pytest.approx(0.0085)]


def test_fake_network_calibrated_mode_only_sleeps_for_target_minus_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.dssd.transport.fake_network import FakeNetwork, LinkTimingModel

    network = FakeNetwork(
        fixed_latency_ms=10.0,
        bandwidth_bytes_per_s=1000.0,
        local_link_model=LinkTimingModel(
            fixed_latency_ms=4.0,
            bandwidth_bytes_per_s=2000.0,
        ),
        warning_label="request",
    )
    sleep_calls: list[float] = []

    monkeypatch.setattr(time, "sleep", sleep_calls.append)

    network.simulate_transfer_bytes(100)

    assert sleep_calls == [pytest.approx(0.056)]


def test_fake_network_warns_once_when_target_link_is_faster_than_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.dssd.transport import fake_network as fake_network_module
    from vllm.dssd.transport.fake_network import FakeNetwork, LinkTimingModel

    network = FakeNetwork(
        fixed_latency_ms=1.0,
        bandwidth_bytes_per_s=1000.0,
        local_link_model=LinkTimingModel(
            fixed_latency_ms=5.0,
            bandwidth_bytes_per_s=500.0,
        ),
        warning_label="response",
    )
    sleep_calls: list[float] = []
    warnings: list[str] = []

    monkeypatch.setattr(time, "sleep", sleep_calls.append)
    monkeypatch.setattr(
        fake_network_module.logger,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )

    network.simulate_transfer_bytes(100)
    network.simulate_transfer_bytes(200)

    assert sleep_calls == []
    assert len(warnings) == 1
    assert "response target link is faster than measured local link" in warnings[0]
    assert "payload_bytes=100" in warnings[0]


def test_fake_network_supports_close_ack_payload() -> None:
    from vllm.dssd.transport.fake_network import FakeNetwork

    network = FakeNetwork(fixed_latency_ms=0.0, bandwidth_bytes_per_s=1e9)
    assert network.estimate_payload_bytes(CloseSessionAck(req_id="req-1")) > 0
