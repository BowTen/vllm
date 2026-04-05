# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any


class DSSDSessionStore:

    def __init__(self) -> None:
        self._sessions: dict[str, Any] = {}

    def put(self, session_id: str, state: Any) -> None:
        self._sessions[session_id] = state

    def get(self, session_id: str) -> Any | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
