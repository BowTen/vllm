# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.sampling_params import SamplingParams


def test_edge_generate_request_round_trips_sampling_params() -> None:
    from vllm.dssd.transport.http_utils import (
        edge_generate_request_from_payload,
        edge_generate_request_to_payload,
    )

    payload = edge_generate_request_to_payload(
        req_id="req-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=4, temperature=0.3),
        lora_request=None,
    )

    decoded = edge_generate_request_from_payload(payload)

    assert decoded["req_id"] == "req-1"
    assert decoded["prompt_token_ids"] == [1, 2, 3]
    assert decoded["sampling_params"].max_tokens == 4
    assert decoded["sampling_params"].temperature == 0.3
    assert decoded["lora_request"] is None


def test_edge_generate_response_round_trips_output_ids() -> None:
    from vllm.dssd.transport.http_utils import (
        edge_generate_response_from_payload,
        edge_generate_response_to_payload,
    )

    payload = edge_generate_response_to_payload(
        req_id="req-7",
        output_ids=[17, 19, 20, 21],
    )

    decoded = edge_generate_response_from_payload(payload)

    assert decoded == {
        "req_id": "req-7",
        "output_ids": [17, 19, 20, 21],
    }
