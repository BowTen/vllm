# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.v1.dssd.edge.coordinator import DSSDRoundCoordinator


class DSSDEdgeServingChat(OpenAIServingChat):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.round_coordinator = DSSDRoundCoordinator(
            edge_engine=self.engine_client,
            transport=None,
        )
