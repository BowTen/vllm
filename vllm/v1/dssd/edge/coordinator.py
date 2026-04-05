# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.v1.dssd.protocol import VerifyRoundResponse


class DSSDRoundCoordinator:
    def __init__(self, edge_engine, transport) -> None:
        self.edge_engine = edge_engine
        self.transport = transport

    def _build_committed_tokens(
        self,
        draft_token_ids: list[int],
        response: VerifyRoundResponse,
        *,
        resampled_token: int | None,
    ) -> list[int]:
        committed = list(draft_token_ids[: response.accepted_count])
        if response.all_accepted:
            if response.bonus_token_id is not None:
                committed.append(response.bonus_token_id)
            return committed
        if resampled_token is not None:
            committed.append(resampled_token)
        return committed
