# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DSSDRequestMetrics:
    request_id: str
    uplink_bytes: int = 0
    downlink_bytes: int = 0
    accepted_tokens: int = 0
    rejected_rounds: int = 0

    def record_uplink(self, size: int) -> None:
        self.uplink_bytes += size

    def record_downlink(self, size: int) -> None:
        self.downlink_bytes += size

    def record_accept(self, token_count: int) -> None:
        self.accepted_tokens += token_count

    def record_reject(self) -> None:
        self.rejected_rounds += 1
