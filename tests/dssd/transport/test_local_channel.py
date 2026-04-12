# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations


def test_local_channel_round_trip_preserves_message_object() -> None:
    from vllm.dssd.transport.local_channel import LocalChannel

    channel = LocalChannel()
    message = {"req_id": "req-1", "kind": "open"}

    channel.send(message)

    assert channel.recv() is message
