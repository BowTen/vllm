# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DSSDRoundMetrics:
    seq_no: int
    accepted_count: int
    reject_index: int | None = None
    draft_latency_ms: float | None = None
    verify_latency_ms: float | None = None
    round_trip_latency_ms: float | None = None

    def export(self) -> dict[str, int | float | None]:
        return {
            "seq_no": self.seq_no,
            "accepted_count": self.accepted_count,
            "reject_index": self.reject_index,
            "draft_latency_ms": self.draft_latency_ms,
            "verify_latency_ms": self.verify_latency_ms,
            "round_trip_latency_ms": self.round_trip_latency_ms,
        }


@dataclass
class DSSDRequestMetrics:
    request_id: str
    uplink_bytes: int = 0
    downlink_bytes: int = 0
    accepted_tokens: int = 0
    rejected_rounds: int = 0
    rounds: list[DSSDRoundMetrics] = field(default_factory=list)
    total_latency_ms: float | None = None
    draft_latency_total_ms: float = 0.0
    verify_latency_total_ms: float = 0.0
    round_trip_latency_total_ms: float = 0.0

    def record_uplink(self, size: int) -> None:
        self.uplink_bytes += size

    def record_downlink(self, size: int) -> None:
        self.downlink_bytes += size

    def record_accept(self, token_count: int) -> None:
        self.accepted_tokens += token_count

    def record_reject(self) -> None:
        self.rejected_rounds += 1

    def record_round(
        self,
        *,
        seq_no: int,
        accepted_count: int,
        reject_index: int | None = None,
        draft_latency_ms: float | None = None,
        verify_latency_ms: float | None = None,
        round_trip_latency_ms: float | None = None,
    ) -> None:
        self.record_accept(accepted_count)
        if reject_index is not None:
            self.record_reject()
        if draft_latency_ms is not None:
            self.draft_latency_total_ms += draft_latency_ms
        if verify_latency_ms is not None:
            self.verify_latency_total_ms += verify_latency_ms
        if round_trip_latency_ms is not None:
            self.round_trip_latency_total_ms += round_trip_latency_ms
        self.rounds.append(
            DSSDRoundMetrics(
                seq_no=seq_no,
                accepted_count=accepted_count,
                reject_index=reject_index,
                draft_latency_ms=draft_latency_ms,
                verify_latency_ms=verify_latency_ms,
                round_trip_latency_ms=round_trip_latency_ms,
            )
        )

    def finalize_request(self, *, total_latency_ms: float) -> None:
        self.total_latency_ms = total_latency_ms

    def export(
        self,
        *,
        run_metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return {
            "request": {
                "request_id": self.request_id,
                "uplink_bytes": self.uplink_bytes,
                "downlink_bytes": self.downlink_bytes,
                "accepted_tokens": self.accepted_tokens,
                "rejected_rounds": self.rejected_rounds,
                "latency_ms": {
                    "total": self.total_latency_ms,
                    "draft_total": self.draft_latency_total_ms,
                    "verify_total": self.verify_latency_total_ms,
                    "round_trip_total": self.round_trip_latency_total_ms,
                },
            },
            "rounds": [round_metrics.export() for round_metrics in self.rounds],
            "run": dict(run_metadata or {}),
        }
