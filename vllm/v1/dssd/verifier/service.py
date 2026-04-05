# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.v1.dssd.verifier.session import DSSDVerifierSessionManager


class DSSDVerifierService:
    def __init__(self, engine_client, vllm_config) -> None:
        self.engine_client = engine_client
        self.vllm_config = vllm_config
        self.session_manager = DSSDVerifierSessionManager()

    async def bind_verifier(self, *_args, **_kwargs) -> dict[str, bool]:
        return {"ok": True}
