# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.v1.dssd.engine.session_store import DSSDSessionStore
from vllm.v1.dssd.protocol import VerifyRoundRequest


class DSSDSessionRunner:

    def __init__(self) -> None:
        self.edge_sessions = DSSDSessionStore()
        self.verifier_sessions = DSSDSessionStore()

    def dssd_verify_round(self, request: VerifyRoundRequest) -> dict[str, object]:
        return {"method": "dssd_verify_round", "request": request}
