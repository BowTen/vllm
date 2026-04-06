# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.v1.dssd.transport import HTTPDSSDTransport
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.v1.dssd.edge.coordinator import DSSDRoundCoordinator


class DSSDEdgeServingChat(OpenAIServingChat):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        dssd_config = getattr(self.engine_client.vllm_config, "dssd_config", None)
        transport = None
        gamma = 1
        if dssd_config is not None:
            gamma = dssd_config.gamma
            if dssd_config.verifier_url:
                transport = HTTPDSSDTransport(
                    dssd_config.verifier_url,
                    network_simulation=dssd_config.network_simulation,
                )
        self.round_coordinator = DSSDRoundCoordinator(
            edge_engine=self.engine_client,
            transport=transport,
            gamma=gamma,
        )

    async def create_chat_completion(self, request, raw_request=None):
        if request.stream:
            return self.round_coordinator.create_chat_completion_stream(
                request=request,
                raw_request=raw_request,
                serving=self,
            )
        return await self.round_coordinator.create_chat_completion(
            request=request,
            raw_request=raw_request,
            serving=self,
        )
