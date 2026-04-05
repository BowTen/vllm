# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
from argparse import Namespace
from collections import defaultdict
import copy
from dataclasses import dataclass, field
import importlib.util
import json
from pathlib import Path
import re
import sys
import types
from typing import Literal

import pytest


ROOT = Path(__file__).resolve().parents[3]
CLI_ARGS_PATH = ROOT / "vllm" / "entrypoints" / "openai" / "cli_args.py"
ARG_UTILS_PATH = ROOT / "vllm" / "engine" / "arg_utils.py"
DSSD_CONFIG_PATH = ROOT / "vllm" / "config" / "dssd.py"
VLLM_CONFIG_PATH = ROOT / "vllm" / "config" / "vllm.py"


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


def _make_logger():
    return types.SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        info_once=lambda *args, **kwargs: None,
        warning_once=lambda *args, **kwargs: None,
    )


def _default_class_attr(name: str):
    if name in {"model", "tokenizer"}:
        return "stub-model"
    if name in {"distributed_executor_backend", "data_parallel_backend"}:
        return "mp"
    if name in {"worker_cls"}:
        return "worker"
    if name in {"prefix_caching_hash_algo"}:
        return "sha256"
    if name.startswith(
        (
            "enable_",
            "disable_",
            "trust_",
            "skip_",
            "fully_",
            "calculate_",
            "collect_",
        )
    ) or name.endswith("_only"):
        return False
    if (
        name.endswith("_size")
        or name.endswith("_rank")
        or name.endswith("_port")
        or name.endswith("_seconds")
        or name.endswith("_interval")
        or name.endswith("_step")
        or name.endswith("_workers")
        or name.endswith("_len")
        or name.startswith("max_")
    ):
        return 1
    if name.endswith("_utilization"):
        return 0.9
    return None


class _ConfigMeta(type):
    def __getattr__(cls, name: str):
        return _default_class_attr(name)


class _BaseConfig(metaclass=_ConfigMeta):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __deepcopy__(self, _memo):
        clone = type(self)()
        clone.__dict__.update(copy.deepcopy(self.__dict__))
        return clone


def _load_dssd_module(monkeypatch: pytest.MonkeyPatch):
    vllm_pkg = _stub_module("vllm")
    vllm_pkg.__path__ = []  # type: ignore[attr-defined]
    config_pkg = _stub_module("vllm.config")
    config_pkg.__path__ = []  # type: ignore[attr-defined]
    utils_mod = _stub_module("vllm.config.utils", config=_config_decorator)

    monkeypatch.setitem(sys.modules, "vllm", vllm_pkg)
    monkeypatch.setitem(sys.modules, "vllm.config", config_pkg)
    monkeypatch.setitem(sys.modules, "vllm.config.utils", utils_mod)

    spec = importlib.util.spec_from_file_location(
        "vllm.config.dssd",
        DSSD_CONFIG_PATH,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "vllm.config.dssd", module)
    spec.loader.exec_module(module)
    return module


def _install_cli_stubs(monkeypatch: pytest.MonkeyPatch):
    logger = _make_logger()

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


def _load_cli_args_module(monkeypatch: pytest.MonkeyPatch):
    _install_cli_stubs(monkeypatch)

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


def _install_vllm_config_stubs(
    monkeypatch: pytest.MonkeyPatch,
    dssd_mod: types.ModuleType,
):
    logger = _make_logger()

    def _field_wrapper(*args, **kwargs):
        supported_kwargs = {}
        if "default" in kwargs:
            supported_kwargs["default"] = kwargs["default"]
        if "default_factory" in kwargs:
            supported_kwargs["default_factory"] = kwargs["default_factory"]
        if "init" in kwargs:
            supported_kwargs["init"] = kwargs["init"]
        return field(*args, **supported_kwargs)

    def _model_validator(**_kwargs):
        def decorate(func):
            return func

        return decorate

    class AttentionConfig(_BaseConfig):
        backend = None

        @classmethod
        def validate_backend_before(cls, value):
            return value

    class CacheConfig(_BaseConfig):
        block_size = 16
        mamba_block_size = None
        kv_offloading_size = None
        kv_offloading_backend = None
        calculate_kv_scales = False
        kv_sharing_fast_prefill = False
        mamba_cache_dtype = None
        mamba_ssm_cache_dtype = None
        mamba_cache_mode = None
        num_gpu_blocks_override = None
        prefix_caching_hash_algo = "sha256"
        enable_prefix_caching = True

    class CompilationConfig(_BaseConfig):
        cudagraph_capture_sizes = None
        max_cudagraph_capture_size = None
        enable_sp = False
        use_inductor_graph_partition = False

        def is_custom_op_enabled(self, _name: str) -> bool:
            return False

    class CompilationMode:
        pass

    class CUDAGraphMode:
        NONE = "none"
        PIECEWISE = "piecewise"
        FULL_AND_PIECEWISE = "full_and_piecewise"

    class DeviceConfig(_BaseConfig):
        device = "cpu"

    class LoadConfig(_BaseConfig):
        use_tqdm_on_load = False
        pt_load_map_location = None

    class KernelConfig(_BaseConfig):
        enable_flashinfer_autotune = None
        moe_backend = "auto"

    class ObservabilityConfig(_BaseConfig):
        show_hidden_metrics_for_version = None
        otlp_traces_endpoint = None
        collect_detailed_traces = None
        kv_cache_metrics = False
        kv_cache_metrics_sample = 0.0
        cudagraph_metrics = False
        enable_layerwise_nvtx_tracing = False
        enable_mfu_metrics = False
        enable_logging_iteration_details = False
        enable_mm_processor_stats = False

    class OffloadConfig(_BaseConfig):
        pass

    class ParallelConfig(_BaseConfig):
        tensor_parallel_size = 1
        pipeline_parallel_size = 1
        data_parallel_size = 1
        distributed_executor_backend = "mp"
        data_parallel_master_ip = "127.0.0.1"
        data_parallel_rpc_port = 1234
        worker_cls = "worker"
        worker_extension_cls = ""

    class ProfilerConfig(_BaseConfig):
        pass

    class SchedulerConfig(_BaseConfig):
        async_scheduling = False
        max_num_batched_tokens = 16
        long_prefill_token_threshold = 0
        disable_chunked_mm_input = False

        @classmethod
        def default_factory(cls):
            return cls()

    class StructuredOutputsConfig(_BaseConfig):
        reasoning_parser = ""
        reasoning_parser_plugin = None

    class SupportsHash:
        pass

    def _replace(instance, **kwargs):
        values = dict(instance.__dict__)
        values.update(kwargs)
        return type(instance)(**values)

    modules = {
        "pydantic": _stub_module(
            "pydantic",
            ConfigDict=lambda **kwargs: kwargs,
            Field=_field_wrapper,
            model_validator=_model_validator,
        ),
        "torch": _stub_module("torch", dtype=object),
        "vllm": _stub_module("vllm"),
        "vllm.config": _stub_module("vllm.config"),
        "vllm.config.attention": _stub_module(
            "vllm.config.attention",
            AttentionConfig=AttentionConfig,
        ),
        "vllm.config.cache": _stub_module("vllm.config.cache", CacheConfig=CacheConfig),
        "vllm.config.compilation": _stub_module(
            "vllm.config.compilation",
            CompilationConfig=CompilationConfig,
            CompilationMode=CompilationMode,
            CUDAGraphMode=CUDAGraphMode,
        ),
        "vllm.config.device": _stub_module(
            "vllm.config.device",
            DeviceConfig=DeviceConfig,
        ),
        "vllm.config.dssd": dssd_mod,
        "vllm.config.ec_transfer": _stub_module(
            "vllm.config.ec_transfer",
            ECTransferConfig=type("ECTransferConfig", (_BaseConfig,), {}),
        ),
        "vllm.config.kernel": _stub_module(
            "vllm.config.kernel",
            KernelConfig=KernelConfig,
        ),
        "vllm.config.kv_events": _stub_module(
            "vllm.config.kv_events",
            KVEventsConfig=type("KVEventsConfig", (_BaseConfig,), {}),
        ),
        "vllm.config.kv_transfer": _stub_module(
            "vllm.config.kv_transfer",
            KVTransferConfig=type("KVTransferConfig", (_BaseConfig,), {}),
        ),
        "vllm.config.load": _stub_module("vllm.config.load", LoadConfig=LoadConfig),
        "vllm.config.lora": _stub_module(
            "vllm.config.lora",
            LoRAConfig=type("LoRAConfig", (_BaseConfig,), {}),
        ),
        "vllm.config.model": _stub_module(
            "vllm.config.model",
            ModelConfig=type("ModelConfig", (_BaseConfig,), {}),
        ),
        "vllm.config.observability": _stub_module(
            "vllm.config.observability",
            ObservabilityConfig=ObservabilityConfig,
        ),
        "vllm.config.offload": _stub_module(
            "vllm.config.offload",
            OffloadConfig=OffloadConfig,
        ),
        "vllm.config.parallel": _stub_module(
            "vllm.config.parallel",
            ParallelConfig=ParallelConfig,
        ),
        "vllm.config.profiler": _stub_module(
            "vllm.config.profiler",
            ProfilerConfig=ProfilerConfig,
        ),
        "vllm.config.scheduler": _stub_module(
            "vllm.config.scheduler",
            SchedulerConfig=SchedulerConfig,
        ),
        "vllm.config.speculative": _stub_module(
            "vllm.config.speculative",
            EagleModelTypes=Literal["eagle"],
            NgramGPUTypes=Literal["ngram"],
            SpeculativeConfig=type("SpeculativeConfig", (_BaseConfig,), {}),
        ),
        "vllm.config.structured_outputs": _stub_module(
            "vllm.config.structured_outputs",
            StructuredOutputsConfig=StructuredOutputsConfig,
        ),
        "vllm.config.utils": _stub_module(
            "vllm.config.utils",
            SupportsHash=SupportsHash,
            config=_config_decorator,
            replace=_replace,
        ),
        "vllm.config.weight_transfer": _stub_module(
            "vllm.config.weight_transfer",
            WeightTransferConfig=type("WeightTransferConfig", (_BaseConfig,), {}),
        ),
        "vllm.envs": _stub_module("vllm.envs"),
        "vllm.logger": _stub_module(
            "vllm.logger",
            enable_trace_function_call=lambda *args, **kwargs: None,
            init_logger=lambda _name: logger,
        ),
        "vllm.transformers_utils": _stub_module("vllm.transformers_utils"),
        "vllm.transformers_utils.runai_utils": _stub_module(
            "vllm.transformers_utils.runai_utils",
            is_runai_obj_uri=lambda _uri: False,
        ),
        "vllm.utils": _stub_module("vllm.utils", random_uuid=lambda: "uuid"),
        "vllm.utils.hashing": _stub_module(
            "vllm.utils.hashing",
            safe_hash=lambda *_args, **_kwargs: "hash",
        ),
    }

    for package_name in (
        "vllm",
        "vllm.config",
        "vllm.transformers_utils",
    ):
        modules[package_name].__path__ = []  # type: ignore[attr-defined]

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _load_vllm_config_module(monkeypatch: pytest.MonkeyPatch):
    dssd_mod = _load_dssd_module(monkeypatch)
    _install_vllm_config_stubs(monkeypatch, dssd_mod)

    spec = importlib.util.spec_from_file_location(
        "vllm.config.vllm",
        VLLM_CONFIG_PATH,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "vllm.config.vllm", module)
    spec.loader.exec_module(module)
    return dssd_mod, module


def _install_arg_utils_stubs(
    monkeypatch: pytest.MonkeyPatch,
    dssd_mod: types.ModuleType,
):
    logger = _make_logger()

    class AttentionConfig(_BaseConfig):
        backend = None

        @classmethod
        def validate_backend_before(cls, value):
            return value

    class CacheConfig(_BaseConfig):
        block_size = 16
        gpu_memory_utilization = 0.9
        kv_cache_memory_bytes = None
        calculate_kv_scales = False
        mamba_cache_dtype = None
        mamba_ssm_cache_dtype = None
        mamba_block_size = None
        mamba_cache_mode = None
        kv_sharing_fast_prefill = False
        num_gpu_blocks_override = None
        kv_offloading_size = None
        kv_offloading_backend = None
        prefix_caching_hash_algo = "sha256"

    class CompilationConfig(_BaseConfig):
        cudagraph_capture_sizes = None
        max_cudagraph_capture_size = None

        def is_custom_op_enabled(self, _name: str) -> bool:
            return False

    class DeviceConfig(_BaseConfig):
        device = "cpu"

    class KernelConfig(_BaseConfig):
        enable_flashinfer_autotune = None
        moe_backend = "auto"

    class LoadConfig(_BaseConfig):
        use_tqdm_on_load = False
        pt_load_map_location = None
        model_loader_extra_config = {}
        ignore_patterns = []

    class ModelConfig(_BaseConfig):
        model = "stub-model"
        model_weights = None
        tokenizer = None
        runner = "generate"
        convert = "auto"
        tokenizer_mode = "auto"
        trust_remote_code = False
        allowed_local_media_path = ""
        allowed_media_domains = None
        dtype = "auto"
        seed = 0
        revision = None
        code_revision = None
        hf_token = None
        hf_overrides = None
        tokenizer_revision = None
        max_model_len = 32
        quantization = None
        allow_deprecated_quantization = False
        enforce_eager = False
        enable_return_routed_experts = False
        max_logprobs = 0
        logprobs_mode = None
        disable_sliding_window = False
        disable_cascade_attn = False
        skip_tokenizer_init = False
        enable_prompt_embeds = False
        served_model_name = None
        language_model_only = False
        limit_mm_per_prompt = None
        enable_mm_embeds = False
        interleave_mm_strings = False
        media_io_kwargs = {}
        skip_mm_profiling = False
        config_format = None
        mm_processor_kwargs = None
        mm_processor_cache_gb = 0
        mm_processor_cache_type = None
        mm_shm_cache_max_object_size_mb = 0
        mm_encoder_only = False
        mm_encoder_tp_mode = None
        mm_encoder_attn_backend = None
        pooler_config = None
        generation_config = "auto"
        override_generation_config = {}
        enable_sleep_mode = False
        model_impl = None
        override_attention_dtype = None
        logits_processors = None
        video_pruning_rate = None
        io_processor_plugin = None

    class MultiModalConfig(_BaseConfig):
        enable_mm_embeds = False
        interleave_mm_strings = False
        media_io_kwargs = {}
        mm_processor_kwargs = None
        mm_processor_cache_gb = 0
        mm_processor_cache_type = None
        mm_shm_cache_max_object_size_mb = 0
        mm_encoder_only = False
        mm_encoder_tp_mode = None
        mm_encoder_attn_backend = None
        skip_mm_profiling = False
        video_pruning_rate = None

    class ObservabilityConfig(_BaseConfig):
        show_hidden_metrics_for_version = None
        otlp_traces_endpoint = None
        collect_detailed_traces = None
        kv_cache_metrics = False
        kv_cache_metrics_sample = 0.0
        cudagraph_metrics = False
        enable_layerwise_nvtx_tracing = False
        enable_mfu_metrics = False
        enable_logging_iteration_details = False
        enable_mm_processor_stats = False

    class OffloadConfig(_BaseConfig):
        pass

    class ParallelConfig(_BaseConfig):
        worker_cls = "worker"
        worker_extension_cls = ""
        data_parallel_master_ip = "127.0.0.1"
        data_parallel_rpc_port = 1234
        distributed_executor_backend = "mp"
        ray_workers_use_nsight = False

    class PoolerConfig(_BaseConfig):
        pass

    class PrefetchOffloadConfig(_BaseConfig):
        pass

    class ProfilerConfig(_BaseConfig):
        pass

    class SchedulerConfig(_BaseConfig):
        disable_chunked_mm_input = False
        disable_hybrid_kv_cache_manager = False
        async_scheduling = False
        stream_interval = 1
        max_num_partial_prefills = 1
        max_long_partial_prefills = 1
        policy = "fcfs"
        scheduler_cls = None

    class StructuredOutputsConfig(_BaseConfig):
        reasoning_parser = ""
        reasoning_parser_plugin = None

    class UVAOffloadConfig(_BaseConfig):
        pass

    class WeightTransferConfig(_BaseConfig):
        pass

    class _VllmConfig:
        structured_outputs_config = StructuredOutputsConfig()
        compilation_config = CompilationConfig()
        attention_config = AttentionConfig()
        kernel_config = KernelConfig()
        profiler_config = ProfilerConfig()
        additional_config = {}
        weight_transfer_config = None
        dssd_config = None
        optimization_level = 2
        performance_mode = "balanced"

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.dssd_config = kwargs.get("dssd_config")
            if self.dssd_config is not None:
                self.dssd_config.validate()

    class _TypeAdapter:
        def __init__(self, cls):
            self.cls = cls

        def validate_json(self, value: str):
            parsed = __import__("json").loads(value)
            return self.cls(**parsed)

    class _ValidationError(Exception):
        pass

    class _FieldInfo:
        pass

    def _get_field(cls, name: str):
        default = getattr(cls, name, None)
        if isinstance(default, (dict, list, set)) or (
            default is not None
            and not isinstance(default, (str, int, float, bool, tuple))
        ):
            return field(default_factory=lambda value=default: copy.deepcopy(value))
        return field(default=default)

    current_platform = types.SimpleNamespace(
        pre_register_and_update=lambda: None,
        device_type="cpu",
    )

    modules = {
        "huggingface_hub": _stub_module(
            "huggingface_hub",
            constants=types.SimpleNamespace(HF_HUB_OFFLINE=False),
        ),
        "regex": re,
        "pydantic": _stub_module(
            "pydantic",
            TypeAdapter=_TypeAdapter,
            ValidationError=_ValidationError,
        ),
        "pydantic.fields": _stub_module("pydantic.fields", FieldInfo=_FieldInfo),
        "vllm": _stub_module("vllm"),
        "vllm.envs": _stub_module(
            "vllm.envs",
            VLLM_ENABLE_V1_MULTIPROCESSING=True,
            VLLM_RAY_DP_PACK_STRATEGY="pack",
            validate_environ=lambda _fail: None,
        ),
        "vllm.config": _stub_module(
            "vllm.config",
            AttentionConfig=AttentionConfig,
            CacheConfig=CacheConfig,
            CompilationConfig=CompilationConfig,
            ConfigType=type,
            DeviceConfig=DeviceConfig,
            DSSDConfig=dssd_mod.DSSDConfig,
            ECTransferConfig=type("ECTransferConfig", (_BaseConfig,), {}),
            EPLBConfig=type("EPLBConfig", (_BaseConfig,), {}),
            KernelConfig=KernelConfig,
            KVEventsConfig=type("KVEventsConfig", (_BaseConfig,), {}),
            KVTransferConfig=type("KVTransferConfig", (_BaseConfig,), {}),
            LoadConfig=LoadConfig,
            LoRAConfig=type("LoRAConfig", (_BaseConfig,), {}),
            ModelConfig=ModelConfig,
            MultiModalConfig=MultiModalConfig,
            ObservabilityConfig=ObservabilityConfig,
            OffloadConfig=OffloadConfig,
            ParallelConfig=ParallelConfig,
            PoolerConfig=PoolerConfig,
            PrefetchOffloadConfig=PrefetchOffloadConfig,
            ProfilerConfig=ProfilerConfig,
            SchedulerConfig=SchedulerConfig,
            SpeculativeConfig=type("SpeculativeConfig", (_BaseConfig,), {}),
            StructuredOutputsConfig=StructuredOutputsConfig,
            UVAOffloadConfig=UVAOffloadConfig,
            VllmConfig=_VllmConfig,
            WeightTransferConfig=WeightTransferConfig,
            get_attr_docs=lambda _cls: {},
        ),
        "vllm.config.cache": _stub_module(
            "vllm.config.cache",
            CacheDType=str,
            KVOffloadingBackend=str,
            MambaCacheMode=str,
            MambaDType=str,
            PrefixCachingHashAlgo=str,
        ),
        "vllm.config.device": _stub_module("vllm.config.device", Device=str),
        "vllm.config.kernel": _stub_module("vllm.config.kernel", MoEBackend=str),
        "vllm.config.lora": _stub_module("vllm.config.lora", MaxLoRARanks=int),
        "vllm.config.model": _stub_module(
            "vllm.config.model",
            ConvertOption=str,
            HfOverrides=dict,
            LogprobsMode=str,
            ModelDType=str,
            RunnerOption=str,
            TokenizerMode=str,
        ),
        "vllm.config.multimodal": _stub_module(
            "vllm.config.multimodal",
            MMCacheType=str,
            MMEncoderTPMode=str,
        ),
        "vllm.config.observability": _stub_module(
            "vllm.config.observability",
            DetailedTraceModules=str,
        ),
        "vllm.config.parallel": _stub_module(
            "vllm.config.parallel",
            All2AllBackend=str,
            DataParallelBackend=str,
            DCPCommBackend=str,
            DistributedExecutorBackend=str,
            ExpertPlacementStrategy=str,
        ),
        "vllm.config.scheduler": _stub_module(
            "vllm.config.scheduler",
            SchedulerPolicy=str,
        ),
        "vllm.config.utils": _stub_module("vllm.config.utils", get_field=_get_field),
        "vllm.config.vllm": _stub_module(
            "vllm.config.vllm",
            OptimizationLevel=int,
            PerformanceMode=str,
        ),
        "vllm.engine": _stub_module("vllm.engine"),
        "vllm.logger": _stub_module(
            "vllm.logger",
            init_logger=lambda _name: logger,
            suppress_logging=lambda: types.SimpleNamespace(
                __enter__=lambda self: None,
                __exit__=lambda self, exc_type, exc, tb: False,
            ),
        ),
        "vllm.platforms": _stub_module(
            "vllm.platforms",
            CpuArchEnum=str,
            current_platform=current_platform,
        ),
        "vllm.plugins": _stub_module(
            "vllm.plugins",
            load_general_plugins=lambda: None,
        ),
        "vllm.ray": _stub_module("vllm.ray"),
        "vllm.ray.lazy_utils": _stub_module(
            "vllm.ray.lazy_utils",
            is_in_ray_actor=lambda: False,
            is_ray_initialized=lambda: False,
        ),
        "vllm.transformers_utils": _stub_module("vllm.transformers_utils"),
        "vllm.transformers_utils.config": _stub_module(
            "vllm.transformers_utils.config",
            is_interleaved=lambda _config: False,
            maybe_override_with_speculators=lambda **kwargs: (
                kwargs["model"],
                kwargs["tokenizer"],
                kwargs["vllm_speculative_config"],
            ),
        ),
        "vllm.transformers_utils.gguf_utils": _stub_module(
            "vllm.transformers_utils.gguf_utils",
            is_gguf=lambda _model: False,
        ),
        "vllm.transformers_utils.repo_utils": _stub_module(
            "vllm.transformers_utils.repo_utils",
            get_model_path=lambda model, _revision: model,
        ),
        "vllm.transformers_utils.utils": _stub_module(
            "vllm.transformers_utils.utils",
            is_cloud_storage=lambda _model: False,
        ),
        "vllm.utils": _stub_module("vllm.utils"),
        "vllm.utils.argparse_utils": _stub_module(
            "vllm.utils.argparse_utils",
            FlexibleArgumentParser=argparse.ArgumentParser,
        ),
        "vllm.utils.mem_constants": _stub_module(
            "vllm.utils.mem_constants",
            GiB_bytes=1 << 30,
        ),
        "vllm.utils.network_utils": _stub_module(
            "vllm.utils.network_utils",
            get_ip=lambda: "127.0.0.1",
        ),
        "vllm.utils.torch_utils": _stub_module(
            "vllm.utils.torch_utils",
            resolve_kv_cache_dtype_string=lambda dtype, _model_config: dtype,
        ),
        "vllm.v1": _stub_module("vllm.v1"),
        "vllm.v1.attention": _stub_module("vllm.v1.attention"),
        "vllm.v1.attention.backends": _stub_module("vllm.v1.attention.backends"),
        "vllm.v1.attention.backends.registry": _stub_module(
            "vllm.v1.attention.backends.registry",
            AttentionBackendEnum=str,
        ),
        "vllm.v1.sample": _stub_module("vllm.v1.sample"),
        "vllm.v1.sample.logits_processor": _stub_module(
            "vllm.v1.sample.logits_processor",
            LogitsProcessor=object,
        ),
    }

    for package_name in (
        "vllm",
        "vllm.engine",
        "vllm.ray",
        "vllm.transformers_utils",
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.backends",
        "vllm.v1.sample",
    ):
        modules[package_name].__path__ = []  # type: ignore[attr-defined]

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _load_arg_utils_module(monkeypatch: pytest.MonkeyPatch):
    dssd_mod = _load_dssd_module(monkeypatch)
    _install_arg_utils_stubs(monkeypatch, dssd_mod)

    spec = importlib.util.spec_from_file_location(
        "vllm.engine.arg_utils",
        ARG_UTILS_PATH,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", module)
    spec.loader.exec_module(module)
    return dssd_mod, module


def _prepare_engine_args_for_create_config(engine_args, arg_utils_mod):
    class _ModelConfigObj:
        model = "stub-model"
        model_weights = None
        tokenizer = None
        hf_text_config = object()
        is_attention_free = False
        skip_tokenizer_init = False
        is_moe = False
        max_model_len = 32
        runner_type = "generate"
        is_multimodal_model = False
        is_encoder_decoder = False
        quantization = None

        def get_sliding_window(self):
            return None

    arg_utils_mod.EngineArgs._check_feature_supported = lambda self: None
    arg_utils_mod.EngineArgs._set_default_chunked_prefill_and_prefix_caching_args = (
        lambda self, _model_config: (
            setattr(self, "enable_prefix_caching", True),
            setattr(self, "enable_chunked_prefill", False),
        )
    )
    arg_utils_mod.EngineArgs._set_default_max_num_seqs_and_batched_tokens_args = (
        lambda self, _usage_context, _model_config: (
            setattr(self, "max_num_seqs", 4),
            setattr(self, "max_num_batched_tokens", 16),
        )
    )
    arg_utils_mod.EngineArgs.create_model_config = lambda self: _ModelConfigObj()
    arg_utils_mod.EngineArgs.create_speculative_config = (
        lambda self, target_model_config, target_parallel_config: None
    )
    arg_utils_mod.EngineArgs.create_load_config = lambda self: arg_utils_mod.LoadConfig()

    engine_args.model = "stub-model"
    engine_args.tokenizer = None
    engine_args.revision = None
    engine_args.trust_remote_code = False
    engine_args.speculative_config = None
    engine_args.fail_on_environ_validation = False
    engine_args.kv_cache_dtype = "auto"
    engine_args.block_size = 16
    engine_args.gpu_memory_utilization = 0.9
    engine_args.kv_cache_memory_bytes = None
    engine_args.num_gpu_blocks_override = None
    engine_args.prefix_caching_hash_algo = "sha256"
    engine_args.calculate_kv_scales = False
    engine_args.kv_sharing_fast_prefill = False
    engine_args.mamba_cache_dtype = None
    engine_args.mamba_ssm_cache_dtype = None
    engine_args.mamba_block_size = None
    engine_args.mamba_cache_mode = None
    engine_args.kv_offloading_size = None
    engine_args.kv_offloading_backend = None
    engine_args.nnodes = 1
    engine_args.data_parallel_hybrid_lb = False
    engine_args.data_parallel_external_lb = False
    engine_args.data_parallel_backend = "mp"
    engine_args.data_parallel_rank = None
    engine_args.data_parallel_size_local = None
    engine_args.data_parallel_address = None
    engine_args.data_parallel_rpc_port = None
    engine_args.master_addr = None
    engine_args.master_port = 1234
    engine_args.pipeline_parallel_size = 1
    engine_args.tensor_parallel_size = 1
    engine_args.prefill_context_parallel_size = 1
    engine_args.data_parallel_size = 1
    engine_args.node_rank = 0
    engine_args.distributed_timeout_seconds = 1
    engine_args.enable_expert_parallel = False
    engine_args.all2all_backend = None
    engine_args.enable_elastic_ep = False
    engine_args.enable_dbo = False
    engine_args.ubatch_size = 1
    engine_args.dbo_decode_token_threshold = 1
    engine_args.dbo_prefill_token_threshold = 1
    engine_args.disable_nccl_for_dp_synchronization = False
    engine_args.enable_eplb = False
    engine_args.eplb_config = None
    engine_args.expert_placement_strategy = None
    engine_args.max_parallel_loading_workers = None
    engine_args.disable_custom_all_reduce = False
    engine_args.ray_workers_use_nsight = False
    engine_args.distributed_executor_backend = "mp"
    engine_args.worker_cls = "worker"
    engine_args.worker_extension_cls = ""
    engine_args.decode_context_parallel_size = 1
    engine_args.dcp_comm_backend = None
    engine_args.dcp_kv_cache_interleave_size = 1
    engine_args.cp_kv_cache_interleave_size = 1
    engine_args._api_process_count = 1
    engine_args._api_process_rank = 0
    engine_args.max_num_partial_prefills = 1
    engine_args.max_long_partial_prefills = 1
    engine_args.disable_chunked_mm_input = False
    engine_args.disable_hybrid_kv_cache_manager = False
    engine_args.async_scheduling = False
    engine_args.stream_interval = 1
    engine_args.default_mm_loras = None
    engine_args.enable_lora = False
    engine_args.attention_config = arg_utils_mod.AttentionConfig()
    engine_args.attention_backend = None
    engine_args.kernel_config = arg_utils_mod.KernelConfig()
    engine_args.enable_flashinfer_autotune = None
    engine_args.moe_backend = "auto"
    engine_args.reasoning_parser = ""
    engine_args.reasoning_parser_plugin = None
    engine_args.structured_outputs_config = arg_utils_mod.StructuredOutputsConfig()
    engine_args.show_hidden_metrics_for_version = None
    engine_args.otlp_traces_endpoint = None
    engine_args.collect_detailed_traces = None
    engine_args.kv_cache_metrics = False
    engine_args.kv_cache_metrics_sample = 0.0
    engine_args.cudagraph_metrics = False
    engine_args.enable_layerwise_nvtx_tracing = False
    engine_args.enable_mfu_metrics = False
    engine_args.enable_mm_processor_stats = False
    engine_args.enable_logging_iteration_details = False
    engine_args.compilation_config = arg_utils_mod.CompilationConfig()
    engine_args.cudagraph_capture_sizes = None
    engine_args.max_cudagraph_capture_size = None
    engine_args.offload_backend = None
    engine_args.cpu_offload_gb = 0
    engine_args.cpu_offload_params = False
    engine_args.offload_group_size = 1
    engine_args.offload_num_in_group = 1
    engine_args.offload_prefetch_step = 1
    engine_args.offload_params = False
    engine_args.gdn_prefill_backend = None
    engine_args.additional_config = {}
    engine_args.kv_transfer_config = None
    engine_args.kv_events_config = None
    engine_args.ec_transfer_config = None
    engine_args.profiler_config = arg_utils_mod.ProfilerConfig()
    engine_args.optimization_level = 2
    engine_args.performance_mode = "balanced"
    engine_args.weight_transfer_config = None
    engine_args.shutdown_timeout = 0
    engine_args.tokens_only = False


def _patch_get_kwargs_for_parser_surface(
    monkeypatch: pytest.MonkeyPatch,
    arg_utils_mod,
):
    def _default_kwargs():
        return {"default": None, "help": "", "type": str}

    def _get_kwargs(_cls):
        kwargs = defaultdict(_default_kwargs)
        kwargs["model"] = {"default": "stub-model", "help": "", "type": str}
        kwargs["served_model_name"] = {"default": None, "help": "", "type": str}
        kwargs["middleware"] = {
            "default": [],
            "help": "",
            "type": str,
            "nargs": "+",
        }
        kwargs["allowed_origins"] = {
            "default": ["*"],
            "help": "",
            "type": str,
            "nargs": "+",
        }
        kwargs["allowed_methods"] = {
            "default": ["*"],
            "help": "",
            "type": str,
            "nargs": "+",
        }
        kwargs["allowed_headers"] = {
            "default": ["*"],
            "help": "",
            "type": str,
            "nargs": "+",
        }
        kwargs["api_key"] = {
            "default": None,
            "help": "",
            "type": str,
            "nargs": "+",
        }
        kwargs["lora_modules"] = {
            "default": None,
            "help": "",
            "type": str,
            "nargs": "+",
        }
        kwargs["collect_detailed_traces"] = {
            "default": None,
            "help": "",
            "type": str,
            "nargs": "+",
            "choices": ["worker"],
        }
        kwargs["speculative_config"] = {
            "default": None,
            "help": "",
            "type": json.loads,
        }
        kwargs["additional_config"] = {
            "default": {},
            "help": "",
            "type": json.loads,
        }
        kwargs["dssd_config"] = {
            "default": None,
            "help": "",
            "type": json.loads,
        }
        return kwargs

    monkeypatch.setattr(arg_utils_mod, "get_kwargs", _get_kwargs)


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


def test_parser_exposes_dssd_config_and_preserves_wiring(
    monkeypatch: pytest.MonkeyPatch,
):
    dssd_mod, arg_utils_mod = _load_arg_utils_module(monkeypatch)
    _patch_get_kwargs_for_parser_surface(monkeypatch, arg_utils_mod)

    parser = arg_utils_mod.EngineArgs.add_cli_args(
        arg_utils_mod.FlexibleArgumentParser()
    )
    args = parser.parse_args(
        [
            "--dssd-config",
            (
                '{"enabled": true, "role": "edge", "gamma": 4, '
                '"verifier_url": "http://127.0.0.1:9001"}'
            ),
        ]
    )

    assert args.dssd_config["role"] == "edge"

    engine_args = arg_utils_mod.EngineArgs.from_cli_args(args)
    assert isinstance(engine_args.dssd_config, dssd_mod.DSSDConfig)

    _prepare_engine_args_for_create_config(engine_args, arg_utils_mod)
    vllm_config = engine_args.create_engine_config()

    assert isinstance(vllm_config.dssd_config, dssd_mod.DSSDConfig)
    assert vllm_config.dssd_config.verifier_url == "http://127.0.0.1:9001"


def test_engine_args_from_cli_args_and_create_engine_config_preserve_dssd_config(
    monkeypatch: pytest.MonkeyPatch,
):
    dssd_mod, arg_utils_mod = _load_arg_utils_module(monkeypatch)

    engine_args = arg_utils_mod.EngineArgs.from_cli_args(
        Namespace(
            dssd_config={
                "enabled": True,
                "role": "edge",
                "gamma": 4,
                "verifier_url": "http://127.0.0.1:9001",
            }
        )
    )
    assert isinstance(engine_args.dssd_config, dssd_mod.DSSDConfig)

    _prepare_engine_args_for_create_config(engine_args, arg_utils_mod)
    vllm_config = engine_args.create_engine_config()

    assert isinstance(vllm_config.dssd_config, dssd_mod.DSSDConfig)
    assert vllm_config.dssd_config.role == "edge"
    assert vllm_config.dssd_config.verifier_url == "http://127.0.0.1:9001"


def test_engine_args_create_engine_config_rejects_invalid_edge_dssd_config(
    monkeypatch: pytest.MonkeyPatch,
):
    _dssd_mod, arg_utils_mod = _load_arg_utils_module(monkeypatch)

    engine_args = arg_utils_mod.EngineArgs.from_cli_args(
        Namespace(
            dssd_config={
                "enabled": True,
                "role": "edge",
                "gamma": 4,
            }
        )
    )

    _prepare_engine_args_for_create_config(engine_args, arg_utils_mod)

    with pytest.raises(ValueError, match="verifier_url"):
        engine_args.create_engine_config()


def test_vllm_config_rejects_invalid_edge_dssd_config_on_init(
    monkeypatch: pytest.MonkeyPatch,
):
    dssd_mod, vllm_mod = _load_vllm_config_module(monkeypatch)

    vllm_mod.VllmConfig.try_verify_and_update_config = lambda self: None

    with pytest.raises(ValueError, match="verifier_url"):
        vllm_mod.VllmConfig(
            dssd_config=dssd_mod.DSSDConfig(
                enabled=True,
                role="edge",
                gamma=4,
            )
        )
