# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from argparse import Namespace
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[3]
CLI_ARGS_PATH = ROOT / "vllm" / "entrypoints" / "openai" / "cli_args.py"


def _config_decorator(cls=None, **_kwargs):
    def decorate(target_cls):
        return dataclass(target_cls)

    if cls is None:
        return decorate
    return decorate(cls)


def _stub_module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _load_cli_args_module(monkeypatch: pytest.MonkeyPatch):
    logger = types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
    )

    class _AsyncEngineArgs:
        @staticmethod
        def add_cli_args(parser):
            return parser

    @dataclass
    class _LoRAModulePath:
        name: str
        path: str
        base_model_name: str | None = None

    class _ToolParserManager:
        @staticmethod
        def list_registered():
            return []

    class _FlexibleArgumentParser(argparse.ArgumentParser):
        pass

    modules = {
        "vllm": _stub_module("vllm"),
        "vllm.envs": _stub_module(
            "vllm.envs",
            VLLM_LOGGING_CONFIG_PATH=None,
            VLLM_SERVER_DEV_MODE=False,
        ),
        "vllm.config": _stub_module("vllm.config", config=_config_decorator),
        "vllm.engine": _stub_module("vllm.engine"),
        "vllm.engine.arg_utils": _stub_module(
            "vllm.engine.arg_utils",
            AsyncEngineArgs=_AsyncEngineArgs,
            optional_type=lambda value_type: value_type,
        ),
        "vllm.entrypoints": _stub_module("vllm.entrypoints"),
        "vllm.entrypoints.chat_utils": _stub_module(
            "vllm.entrypoints.chat_utils",
            ChatTemplateContentFormatOption=str,
            validate_chat_template=lambda _template: None,
        ),
        "vllm.entrypoints.constants": _stub_module(
            "vllm.entrypoints.constants",
            H11_MAX_HEADER_COUNT_DEFAULT=256,
            H11_MAX_INCOMPLETE_EVENT_SIZE_DEFAULT=4 * 1024 * 1024,
        ),
        "vllm.entrypoints.openai": _stub_module("vllm.entrypoints.openai"),
        "vllm.entrypoints.openai.models": _stub_module(
            "vllm.entrypoints.openai.models"
        ),
        "vllm.entrypoints.openai.models.protocol": _stub_module(
            "vllm.entrypoints.openai.models.protocol",
            LoRAModulePath=_LoRAModulePath,
        ),
        "vllm.logger": _stub_module("vllm.logger", init_logger=lambda _name: logger),
        "vllm.tool_parsers": _stub_module(
            "vllm.tool_parsers",
            ToolParserManager=_ToolParserManager,
        ),
        "vllm.utils": _stub_module("vllm.utils"),
        "vllm.utils.argparse_utils": _stub_module(
            "vllm.utils.argparse_utils",
            FlexibleArgumentParser=_FlexibleArgumentParser,
        ),
    }

    for package_name in (
        "vllm",
        "vllm.engine",
        "vllm.entrypoints",
        "vllm.entrypoints.openai",
        "vllm.entrypoints.openai.models",
        "vllm.utils",
    ):
        modules[package_name].__path__ = []  # type: ignore[attr-defined]

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location(
        "vllm.entrypoints.openai.cli_args",
        CLI_ARGS_PATH,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "vllm.entrypoints.openai.cli_args", module)
    spec.loader.exec_module(module)
    return module


def test_validate_parsed_serve_args_rejects_edge_without_verifier_url(
    monkeypatch: pytest.MonkeyPatch,
):
    cli_args_mod = _load_cli_args_module(monkeypatch)

    args = Namespace(
        subparser="serve",
        chat_template=None,
        enable_auto_tool_choice=False,
        tool_call_parser=None,
        enable_log_outputs=False,
        enable_log_requests=False,
        dssd_config={"enabled": True, "role": "edge", "gamma": 4},
    )

    with pytest.raises(TypeError, match="verifier_url"):
        cli_args_mod.validate_parsed_serve_args(args)


def test_validate_parsed_serve_args_allows_verifier_role_without_verifier_url(
    monkeypatch: pytest.MonkeyPatch,
):
    cli_args_mod = _load_cli_args_module(monkeypatch)

    args = Namespace(
        subparser="serve",
        chat_template=None,
        enable_auto_tool_choice=False,
        tool_call_parser=None,
        enable_log_outputs=False,
        enable_log_requests=False,
        dssd_config={"enabled": True, "role": "verifier", "gamma": 4},
    )

    cli_args_mod.validate_parsed_serve_args(args)
