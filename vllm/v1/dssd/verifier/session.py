# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any


@dataclass
class DSSDVerifierSessionState:
    verifier_session_id: str
    sampling_params_fingerprint: str
    seq_no: int = -1
    committed_token_ids: list[int] = field(default_factory=list)
    last_response_cache: dict[int, Any] = field(default_factory=dict)
    last_activity_at: float = field(default_factory=time)


class DSSDVerifierSessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, DSSDVerifierSessionState] = {}

    def create_session(
        self, verifier_session_id: str, *, sampling_params_fingerprint: str
    ) -> None:
        self._sessions[verifier_session_id] = DSSDVerifierSessionState(
            verifier_session_id=verifier_session_id,
            sampling_params_fingerprint=sampling_params_fingerprint,
        )

    def cache_response(self, verifier_session_id: str, *, seq_no: int, response: Any):
        self._sessions[verifier_session_id].last_response_cache[seq_no] = response

    def get_cached_response(self, verifier_session_id: str, *, seq_no: int) -> Any:
        return self._sessions[verifier_session_id].last_response_cache.get(seq_no)

    def update_seq_no(self, verifier_session_id: str, seq_no: int) -> None:
        self._sessions[verifier_session_id].seq_no = seq_no

    def ensure_next_seq_no(self, verifier_session_id: str, seq_no: int) -> None:
        current = self._sessions[verifier_session_id].seq_no
        if seq_no > current + 1:
            raise ValueError("out-of-order verify_round seq_no")
