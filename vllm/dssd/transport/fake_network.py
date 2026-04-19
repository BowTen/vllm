from __future__ import annotations

from dataclasses import fields, is_dataclass
import time


class FakeNetwork:
    def __init__(
        self,
        *,
        fixed_latency_ms: float,
        bandwidth_bytes_per_s: float | None,
    ) -> None:
        if fixed_latency_ms < 0:
            raise ValueError("fixed_latency_ms must be non-negative")
        if bandwidth_bytes_per_s is not None and bandwidth_bytes_per_s <= 0:
            raise ValueError("bandwidth_bytes_per_s must be positive")
        self.fixed_latency_ms = float(fixed_latency_ms)
        self.bandwidth_bytes_per_s = (
            None if bandwidth_bytes_per_s is None else float(bandwidth_bytes_per_s)
        )

    def transfer_time_s(self, payload_bytes: int) -> float:
        transfer_s = self.fixed_latency_ms / 1000.0
        if self.bandwidth_bytes_per_s is not None:
            transfer_s += int(payload_bytes) / self.bandwidth_bytes_per_s
        return transfer_s

    def simulate_transfer(self, payload: object) -> None:
        time.sleep(
            self.transfer_time_s(
                payload_bytes=self.estimate_payload_bytes(payload),
            )
        )

    def estimate_payload_bytes(self, payload: object) -> int:
        if payload is None:
            return 1
        if isinstance(payload, bool):
            return 1
        if isinstance(payload, int):
            return 8
        if isinstance(payload, float):
            return 8
        if isinstance(payload, str):
            return max(len(payload.encode("utf-8")), 1)
        if hasattr(payload, "numel") and hasattr(payload, "element_size"):
            return max(int(payload.numel()) * int(payload.element_size()), 1)
        if isinstance(payload, (list, tuple)):
            return max(sum(self.estimate_payload_bytes(item) for item in payload), 1)
        if isinstance(payload, dict):
            total = 0
            for key, value in payload.items():
                total += self.estimate_payload_bytes(key)
                total += self.estimate_payload_bytes(value)
            return max(total, 1)
        if is_dataclass(payload):
            total = 0
            for field in fields(payload):
                total += self.estimate_payload_bytes(getattr(payload, field.name))
            return max(total, 1)
        return 1
