from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields, is_dataclass
import logging
import time


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LinkTimingModel:
    fixed_latency_ms: float
    bandwidth_bytes_per_s: float | None

    def transfer_time_s(self, payload_bytes: int) -> float:
        transfer_s = self.fixed_latency_ms / 1000.0
        if self.bandwidth_bytes_per_s is not None:
            transfer_s += int(payload_bytes) / self.bandwidth_bytes_per_s
        return transfer_s


class FakeNetwork:
    def __init__(
        self,
        *,
        fixed_latency_ms: float,
        bandwidth_bytes_per_s: float | None,
        local_link_model: LinkTimingModel | None = None,
        warning_label: str | None = None,
    ) -> None:
        if fixed_latency_ms < 0:
            raise ValueError("fixed_latency_ms must be non-negative")
        if bandwidth_bytes_per_s is not None and bandwidth_bytes_per_s <= 0:
            raise ValueError("bandwidth_bytes_per_s must be positive")
        self.fixed_latency_ms = float(fixed_latency_ms)
        self.bandwidth_bytes_per_s = (
            None if bandwidth_bytes_per_s is None else float(bandwidth_bytes_per_s)
        )
        self.local_link_model = local_link_model
        self.warning_label = warning_label or "network"
        self._warned_local_faster = False

    def transfer_time_s(self, payload_bytes: int) -> float:
        transfer_s = self.fixed_latency_ms / 1000.0
        if self.bandwidth_bytes_per_s is not None:
            transfer_s += int(payload_bytes) / self.bandwidth_bytes_per_s
        return transfer_s

    def simulate_transfer(
        self,
        payload: object,
        *,
        payload_bytes: int | None = None,
    ) -> None:
        self.simulate_transfer_bytes(
            self.estimate_payload_bytes(payload)
            if payload_bytes is None
            else int(payload_bytes)
        )

    def simulate_transfer_bytes(self, payload_bytes: int) -> None:
        delay_s = self.transfer_time_s(payload_bytes=payload_bytes)
        if self.local_link_model is not None:
            local_s = self.local_link_model.transfer_time_s(payload_bytes)
            delay_s -= local_s
            if delay_s <= 0.0:
                self._warn_target_faster_than_local_once(
                    payload_bytes=payload_bytes,
                    target_s=self.transfer_time_s(payload_bytes=payload_bytes),
                    local_s=local_s,
                )
                return
        time.sleep(delay_s)

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

    def _warn_target_faster_than_local_once(
        self,
        *,
        payload_bytes: int,
        target_s: float,
        local_s: float,
    ) -> None:
        if self._warned_local_faster:
            return
        logger.warning(
            "%s target link is faster than measured local link; "
            "skipping extra delay for payload_bytes=%s "
            "(target_ms=%.3f, local_ms=%.3f); suppressing further warnings",
            self.warning_label,
            payload_bytes,
            target_s * 1000.0,
            local_s * 1000.0,
        )
        self._warned_local_faster = True
