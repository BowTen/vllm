# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VerifierRoundBatchKey:
    gamma: int
    sampling_signature: str


@dataclass(frozen=True)
class VerifierRoundBatchItem:
    verifier_session_id: str
    batch_key: VerifierRoundBatchKey
    payload: Any


@dataclass
class VerifierRoundBatch:
    batch_key: VerifierRoundBatchKey
    items: list[VerifierRoundBatchItem] = field(default_factory=list)


class VerifierRoundBatcher:
    def __init__(self) -> None:
        self._pending: list[VerifierRoundBatchItem] = []

    def add(self, item: VerifierRoundBatchItem) -> None:
        self._pending.append(item)

    def flush(self) -> list[VerifierRoundBatch]:
        groups: dict[VerifierRoundBatchKey, VerifierRoundBatch] = {}
        for item in self._pending:
            group = groups.get(item.batch_key)
            if group is None:
                group = VerifierRoundBatch(batch_key=item.batch_key)
                groups[item.batch_key] = group
            group.items.append(item)
        self._pending.clear()
        return list(groups.values())
