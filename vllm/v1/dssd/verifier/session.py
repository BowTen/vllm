# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import random
from dataclasses import dataclass, field
from time import time
from typing import Any


@dataclass
class DSSDVerifierSessionState:
    verifier_session_id: str
    binding_id: str
    sampling_params_fingerprint: str
    seq_no: int = -1
    committed_token_ids: list[int] = field(default_factory=list)
    last_response_cache: dict[int, Any] = field(default_factory=dict)
    rng: random.Random = field(default_factory=random.Random, repr=False)
    last_activity_at: float = field(default_factory=time)


class DSSDVerifierSessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, DSSDVerifierSessionState] = {}

    def create_session(
        self,
        verifier_session_id: str,
        *,
        binding_id: str,
        sampling_params_fingerprint: str,
        prompt_token_ids: list[int] | None = None,
    ) -> None:
        self._sessions[verifier_session_id] = DSSDVerifierSessionState(
            verifier_session_id=verifier_session_id,
            binding_id=binding_id,
            sampling_params_fingerprint=sampling_params_fingerprint,
            committed_token_ids=list(prompt_token_ids or []),
            rng=random.Random(verifier_session_id),
        )

    def get_session(self, verifier_session_id: str) -> DSSDVerifierSessionState:
        try:
            return self._sessions[verifier_session_id]
        except KeyError as exc:
            raise ValueError("unknown verifier session") from exc

    def delete_session(self, verifier_session_id: str) -> bool:
        return self._sessions.pop(verifier_session_id, None) is not None

    def cache_response(self, verifier_session_id: str, *, seq_no: int, response: Any):
        self.get_session(verifier_session_id).last_response_cache[seq_no] = response

    def get_cached_response(self, verifier_session_id: str, *, seq_no: int) -> Any:
        return self.get_session(verifier_session_id).last_response_cache.get(seq_no)

    def update_seq_no(self, verifier_session_id: str, seq_no: int) -> None:
        session = self.get_session(verifier_session_id)
        session.seq_no = seq_no
        session.last_activity_at = time()

    def ensure_next_seq_no(self, verifier_session_id: str, seq_no: int) -> None:
        session = self.get_session(verifier_session_id)
        current = session.seq_no
        if seq_no <= current and seq_no not in session.last_response_cache:
            raise ValueError("stale verify_round seq_no")
        if seq_no > current + 1:
            raise ValueError("out-of-order verify_round seq_no")

    def append_prefix_delta(
        self, verifier_session_id: str, prefix_delta_token_ids: list[int]
    ) -> None:
        session = self.get_session(verifier_session_id)
        if prefix_delta_token_ids:
            session.committed_token_ids.extend(prefix_delta_token_ids)
        session.last_activity_at = time()

    def commit_tokens(
        self, verifier_session_id: str, token_ids: list[int]
    ) -> None:
        session = self.get_session(verifier_session_id)
        if token_ids:
            session.committed_token_ids.extend(token_ids)
        session.last_activity_at = time()
