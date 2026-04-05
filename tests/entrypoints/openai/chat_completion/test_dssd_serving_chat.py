# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

ROOT = Path(__file__).resolve().parents[4]
VLLM_DIR = ROOT / "vllm"
CHAT_DIR = VLLM_DIR / "entrypoints" / "openai" / "chat_completion"
GENERATE_DIR = VLLM_DIR / "entrypoints" / "openai" / "generate"


def _install_package_stub(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


def _install_module_stub(name: str, **attrs) -> None:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class _DummyServingBase:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs
        self.engine_client = args[0] if args else kwargs.get("engine_client")

    def warmup(self) -> None:
        return None

    async def create_chat_completion(self, *args, **kwargs):
        return ("base", args, kwargs)

    def create_error_response(self, message: str, **kwargs):
        return {"message": message, **kwargs}


class _DummyModelConfig:
    model = "dummy-model"
    max_model_len = 1024
    hf_config = SimpleNamespace(model_type="dummy")
    hf_text_config = SimpleNamespace(model_type="dummy")
    generation_config = "vllm"
    override_generation_config = {}

    def get_diff_sampling_param(self):
        return {}


class _DummyVllmConfig:
    def __init__(
        self,
        role: str = "edge",
        verifier_url: str | None = "http://127.0.0.1:9001",
    ) -> None:
        self.dssd_config = SimpleNamespace(
            enabled=True,
            role=role,
            gamma=4,
            verifier_url=verifier_url,
            network_simulation=SimpleNamespace(
                latency_ms=0.0,
                bandwidth_mbps=None,
                jitter_ms=0.0,
            ),
        )


_install_package_stub("vllm", VLLM_DIR)
_install_package_stub("vllm.entrypoints", VLLM_DIR / "entrypoints")
_install_package_stub("vllm.entrypoints.openai", VLLM_DIR / "entrypoints" / "openai")
_install_package_stub(
    "vllm.entrypoints.openai.chat_completion", CHAT_DIR
)
_install_package_stub("vllm.entrypoints.openai.generate", GENERATE_DIR)
_install_package_stub("vllm.v1", VLLM_DIR / "v1")
_install_package_stub("vllm.v1.dssd", VLLM_DIR / "v1" / "dssd")
_install_package_stub("vllm.v1.dssd.edge", VLLM_DIR / "v1" / "dssd" / "edge")

_install_module_stub(
    "vllm.entrypoints.openai.chat_completion.serving",
    OpenAIServingChat=_DummyServingBase,
)
_install_module_stub(
    "vllm.entrypoints.openai.completion.serving",
    OpenAIServingCompletion=_DummyServingBase,
)
_install_module_stub(
    "vllm.entrypoints.openai.responses.serving",
    OpenAIServingResponses=_DummyServingBase,
)
_install_module_stub(
    "vllm.entrypoints.serve.disagg.serving",
    ServingTokens=_DummyServingBase,
)
_install_module_stub(
    "vllm.entrypoints.serve.render.serving",
    OpenAIServingRender=_DummyServingBase,
)
_install_module_stub(
    "vllm.entrypoints.anthropic.serving",
    AnthropicServingMessages=_DummyServingBase,
)
_install_module_stub(
    "vllm.entrypoints.chat_utils",
    load_chat_template=lambda _template: "resolved-template",
)
_install_module_stub(
    "vllm.entrypoints.mcp.tool_server",
    DemoToolServer=_DummyServingBase,
    MCPToolServer=_DummyServingBase,
    ToolServer=object,
)

dssd_coordinator_module = _load_module(
    "vllm.v1.dssd.edge.coordinator",
    VLLM_DIR / "v1" / "dssd" / "edge" / "coordinator.py",
)
dssd_serving_module = _load_module(
    "vllm.entrypoints.openai.chat_completion.dssd_serving",
    CHAT_DIR / "dssd_serving.py",
)
api_router_module = _load_module(
    "vllm.entrypoints.openai.generate.api_router",
    GENERATE_DIR / "api_router.py",
)

DSSDEdgeServingChat = dssd_serving_module.DSSDEdgeServingChat
DSSDRoundCoordinator = dssd_coordinator_module.DSSDRoundCoordinator
init_generate_state = api_router_module.init_generate_state


def test_dssd_serving_chat_initializes_round_coordinator():
    engine_client = SimpleNamespace(vllm_config=_DummyVllmConfig())
    models = SimpleNamespace(model_name=lambda *_args, **_kwargs: "dummy-model")
    serving = DSSDEdgeServingChat(
        engine_client,
        models,
        "assistant",
        openai_serving_render=SimpleNamespace(),
        request_logger=None,
        chat_template=None,
        chat_template_content_format="string",
    )

    assert isinstance(serving.round_coordinator, DSSDRoundCoordinator)
    assert serving.round_coordinator.edge_engine is engine_client


def test_dssd_serving_chat_delegates_generation_to_round_coordinator():
    engine_client = SimpleNamespace(vllm_config=_DummyVllmConfig())
    models = SimpleNamespace(model_name=lambda *_args, **_kwargs: "dummy-model")
    serving = DSSDEdgeServingChat(
        engine_client,
        models,
        "assistant",
        openai_serving_render=SimpleNamespace(),
        request_logger=None,
        chat_template=None,
        chat_template_content_format="string",
    )
    request = SimpleNamespace(stream=False)
    raw_request = SimpleNamespace()
    sentinel = {"path": "dssd"}
    serving.round_coordinator = SimpleNamespace(
        create_chat_completion=AsyncMock(return_value=sentinel),
    )

    import asyncio

    result = asyncio.run(serving.create_chat_completion(request, raw_request))

    assert result == sentinel
    serving.round_coordinator.create_chat_completion.assert_awaited_once_with(
        request=request,
        raw_request=raw_request,
        serving=serving,
    )


def test_init_generate_state_uses_dssd_serving_chat_for_edge_role():
    engine_client = SimpleNamespace(
        vllm_config=_DummyVllmConfig(role="edge"),
        model_config=_DummyModelConfig(),
        renderer=SimpleNamespace(),
        io_processor=SimpleNamespace(),
        errored=False,
        dead_error=RuntimeError("dead"),
    )
    state = SimpleNamespace(
        openai_serving_models=SimpleNamespace(registry="registry"),
        openai_serving_render=None,
        openai_serving_chat=None,
    )
    args = SimpleNamespace(
        tool_server=None,
        chat_template=None,
        chat_template_content_format="string",
        exclude_tools_when_tool_choice_none=False,
        enable_auto_tool_choice=False,
        tool_call_parser=None,
        default_chat_template_kwargs=None,
        return_tokens_as_token_ids=False,
        response_role="assistant",
        structured_outputs_config=SimpleNamespace(reasoning_parser=""),
        enable_prompt_tokens_details=False,
        enable_force_include_usage=False,
        enable_log_outputs=False,
        enable_log_deltas=True,
        trust_request_chat_template=False,
        log_error_stack=False,
        tokens_only=False,
    )

    import asyncio

    asyncio.run(
        init_generate_state(
            engine_client,
            state,
            args,
            request_logger=None,
            supported_tasks=("generate",),
        )
    )

    assert isinstance(state.openai_serving_chat, DSSDEdgeServingChat)
