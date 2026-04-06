# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.v1.dssd.edge.coordinator import DSSDRoundCoordinator
from vllm.v1.dssd.transport import HTTPDSSDTransport


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


class DSSDExperimentBaselineServingChat(OpenAIServingChat):
    async def create_chat_completion(self, request, raw_request=None):
        response = await super().create_chat_completion(request, raw_request)
        self._publish_experiment_result(request=request, raw_request=raw_request)
        return response

    def _publish_experiment_result(self, *, request, raw_request) -> None:
        if raw_request is None:
            return
        state = getattr(raw_request, "state", None)
        if state is None:
            return
        setattr(
            state,
            "dssd_experiment_result",
            {
                "request": {
                    "request_id": self._request_id(request, raw_request),
                },
                "rounds": [],
                "run": {
                    "mode": "baseline",
                    "edge_model_id": self._model_name(),
                    "verifier_model_id": None,
                    "sampling": self._sampling_metadata(request),
                    "network": self._network_metadata(),
                },
            },
        )

    def _request_id(self, request, raw_request) -> str:
        request_id = getattr(request, "request_id", None)
        if request_id:
            return str(request_id)
        state = getattr(raw_request, "state", None)
        metadata = getattr(state, "request_metadata", None)
        request_id = getattr(metadata, "request_id", None)
        if request_id:
            return str(request_id)
        return "baseline"

    def _model_name(self) -> str:
        models = getattr(self, "models", None)
        if models is not None and hasattr(models, "model_name"):
            return models.model_name(None)
        model_config = getattr(self.engine_client, "model_config", None)
        return getattr(model_config, "model", "dssd-baseline")

    def _sampling_metadata(self, request) -> dict[str, float | int | None]:
        default_sampling_params = getattr(self, "default_sampling_params", {}) or {}
        return {
            "temperature": getattr(request, "temperature", None)
            if getattr(request, "temperature", None) is not None
            else default_sampling_params.get("temperature"),
            "top_p": getattr(request, "top_p", None)
            if getattr(request, "top_p", None) is not None
            else default_sampling_params.get("top_p"),
            "top_k": getattr(request, "top_k", None)
            if getattr(request, "top_k", None) is not None
            else default_sampling_params.get("top_k"),
            "min_p": getattr(request, "min_p", None)
            if getattr(request, "min_p", None) is not None
            else default_sampling_params.get("min_p"),
            "max_tokens": getattr(request, "max_completion_tokens", None)
            or getattr(request, "max_tokens", None)
            or default_sampling_params.get("max_tokens"),
        }

    def _network_metadata(self) -> dict[str, float | None]:
        dssd_config = getattr(self.engine_client.vllm_config, "dssd_config", None)
        network_simulation = getattr(dssd_config, "network_simulation", None)
        if network_simulation is None:
            return {
                "latency_ms": None,
                "bandwidth_mbps": None,
                "jitter_ms": None,
            }
        return {
            "latency_ms": getattr(network_simulation, "latency_ms", None),
            "bandwidth_mbps": getattr(network_simulation, "bandwidth_mbps", None),
            "jitter_ms": getattr(network_simulation, "jitter_ms", None),
        }
