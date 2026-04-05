# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections import defaultdict


class VerifierRoundBatcher:
    def __init__(self) -> None:
        self._pending: list[dict] = []

    def add(self, item: dict) -> None:
        self._pending.append(item)

    def flush(self) -> list[list[dict]]:
        groups: dict[tuple[object, object], list[dict]] = defaultdict(list)
        for item in self._pending:
            key = (item["gamma"], item["sampling_signature"])
            groups[key].append(item)
        self._pending.clear()
        return list(groups.values())
