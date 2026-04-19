#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark local single-request generation for native, edge, and verifier.

This benchmark uses the same prompt token IDs and timing envelope for all
engines:
- batch size is fixed to 1
- a request is timed from enqueue/open_session to the last generated token
- throughput is computed from generated tokens in that request
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeVar

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_RESULT_DIR = REPO_ROOT / "benchmarks" / "dssd" / "results"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

T = TypeVar("T")

import torch
from transformers import AutoTokenizer

from vllm import LLM, SamplingParams, TokensPrompt, envs
from vllm.config import set_current_vllm_config
from vllm.dssd.edge import (
    DSSDEdgeDraftSampler,
    DSSDEdgeDraftSamplerV1,
    EdgeDecodeEngine,
    EdgeDecodeEngineV1,
    EdgeSchedulerAdapter,
    EdgeStateBridge,
    EdgeStateBridgeV1,
)
from vllm.dssd.verifier import (
    DSSDVerifierSampler,
    DSSDVerifierSamplerV1,
    VerifierDecodeEngine,
    VerifierDecodeEngineV1,
    VerifierStateBridge,
    VerifierStateBridgeV1,
)
from vllm.dssd.verifier.types import VerifierSession
from vllm.distributed import cleanup_dist_env_and_memory
import vllm.platforms as platforms
from vllm.platforms.cuda import CudaPlatform
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.worker.gpu_worker import Worker


@dataclass
class BenchmarkResult:
    engine: str
    model: str
    prompt_tokens: int
    requested_output_tokens: int
    generated_tokens: int
    warmup_iters: int
    benchmark_iters: int
    total_seconds: float
    avg_seconds: float
    tokens_per_second: float
    phase_seconds: dict[str, float] | None = None


class BenchmarkVerifierSchedulerAdapter:
    """Verifier scheduler wrapper with explicit request block hashing."""

    def __init__(self, kv_cache_manager, block_hasher) -> None:
        from vllm.dssd.verifier.scheduler import VerifierSchedulerAdapter

        self._delegate = VerifierSchedulerAdapter(kv_cache_manager=kv_cache_manager)
        self.kv_cache_manager = kv_cache_manager
        self._block_hasher = block_hasher

    def allocate_blocks(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params,
        lora_request=None,
    ) -> tuple[list[int], ...]:
        from vllm.v1.request import Request

        request = Request(
            request_id=req_id,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=sampling_params,
            pooling_params=None,
            lora_request=lora_request,
            block_hasher=self._block_hasher,
        )
        blocks = self.kv_cache_manager.allocate_slots(
            request=request,
            num_new_tokens=request.num_tokens - request.num_computed_tokens,
        )
        if blocks is None:
            raise RuntimeError("prefill cannot allocate KV blocks for verifier")
        self._delegate._pending_new_block_ids_to_zero = (  # noqa: SLF001
            self._delegate._drain_new_block_ids_to_zero(blocks)  # noqa: SLF001
        )
        block_ids = blocks.get_block_ids(allow_none=True)
        return ([],) if block_ids is None else block_ids

    def _make_request(self, session):
        from vllm.v1.request import Request

        request = Request(
            request_id=session.req_id,
            prompt_token_ids=list(session.prompt_token_ids),
            sampling_params=session.sampling_params,
            pooling_params=None,
            lora_request=session.lora_request,
            block_hasher=self._block_hasher,
        )
        finalized_output = session.token_ids[session.prompt_len:]
        if finalized_output:
            request.append_output_token_ids(finalized_output)
        request.num_computed_tokens = session.num_computed_tokens
        return request

    def __getattr__(self, name: str):
        return getattr(self._delegate, name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        choices=("native", "edge", "verifier", "all"),
        default="all",
        help="Which engine to benchmark.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model path or HF model name.",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=128,
        help="Prompt token length used for prefill.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of generated tokens to benchmark.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=1,
        help="Number of warmup iterations before timing.",
    )
    parser.add_argument(
        "--benchmark-iters",
        type=int,
        default=3,
        help="Number of timed iterations.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Model dtype passed to vLLM.",
    )
    parser.add_argument(
        "--native-model-runner",
        choices=("v1", "v2"),
        default="v1",
        help="Model runner used by the native engine benchmark.",
    )
    parser.add_argument(
        "--edge-model-runner",
        choices=("v1", "v2"),
        default="v1",
        help="Model runner used by the edge engine benchmark.",
    )
    parser.add_argument(
        "--verifier-model-runner",
        choices=("v1", "v2"),
        default="v1",
        help="Model runner used by the verifier engine benchmark.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for model/KV cache.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=0,
        help="Max model length. Defaults to prompt_len + max_new_tokens + 32.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. 0.0 means greedy.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Sampling top-p.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Sampling top-k.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to tokenizer/model loading.",
    )
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable eager mode. Use --no-enforce-eager to allow compile/cudagraph.",
    )
    parser.add_argument(
        "--honor-eos",
        action="store_true",
        help="Stop on EOS instead of forcing max-new-tokens.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON in addition to the human-readable summary.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--phase-timing",
        action="store_true",
        help=(
            "Collect coarse phase timing for edge/verifier local generation. "
            "Native remains total-only."
        ),
    )
    parser.add_argument(
        "--dump-decode-state",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Dump scheduler/model-runner decode state for the first N decode "
            "steps of each runtime-engine request."
        ),
    )
    parser.add_argument(
        "--output-json",
        nargs="?",
        const="auto",
        default=None,
        metavar="PATH",
        help=(
            "Write benchmark JSON to PATH. If passed without PATH, write to "
            "benchmarks/dssd/results/<timestamp>-local-engines-*.json."
        ),
    )
    return parser


def _resolved_max_model_len(args: argparse.Namespace) -> int:
    if args.max_model_len > 0:
        return args.max_model_len
    return args.prompt_len + args.max_new_tokens + 32


def _json_payload(results: list[BenchmarkResult]) -> dict[str, Any] | list[dict[str, Any]]:
    if len(results) == 1:
        return asdict(results[0])
    return [asdict(result) for result in results]


def _slugify_path_component(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return slug or "model"


def _default_output_json_path(
    args: argparse.Namespace,
    results: list[BenchmarkResult],
) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    model_name = _slugify_path_component(Path(results[0].model).name)
    engine_name = args.engine
    return (
        DEFAULT_RESULT_DIR
        / f"{timestamp}-local-engines-{engine_name}-{model_name}.json"
    )


def maybe_write_results_json(
    args: argparse.Namespace,
    results: list[BenchmarkResult],
) -> Path | None:
    if args.output_json is None:
        return None

    output_path = (
        _default_output_json_path(args, results)
        if args.output_json == "auto"
        else Path(args.output_json).expanduser()
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_payload(results), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def build_sampling_params(args: argparse.Namespace) -> SamplingParams:
    return SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        ignore_eos=not args.honor_eos,
        detokenize=False,
    )


def _build_prompt_token_ids(
    model: str,
    prompt_len: int,
    *,
    trust_remote_code: bool,
) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=trust_remote_code,
        local_files_only=Path(model).exists(),
    )
    seed_text = "Benchmark prompt. " * max(prompt_len, 64)
    token_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not token_ids:
        raise RuntimeError("tokenizer returned no prompt tokens")

    prompt_token_ids: list[int] = []
    while len(prompt_token_ids) < prompt_len:
        prompt_token_ids.extend(token_ids)
    return prompt_token_ids[:prompt_len]


def synchronize_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed_phase(fn) -> tuple[T, float]:
    synchronize_if_needed()
    t0 = time.perf_counter()
    result = fn()
    synchronize_if_needed()
    return result, time.perf_counter() - t0


def _new_phase_seconds() -> dict[str, float]:
    return {
        "open_session": 0.0,
        "prefill_execute": 0.0,
        "bootstrap": 0.0,
        "decode_schedule": 0.0,
        "decode_model": 0.0,
        "decode_postprocess": 0.0,
        "close_session": 0.0,
    }


def _new_zero_phase_seconds() -> dict[str, float]:
    return {phase_name: 0.0 for phase_name in _new_phase_seconds()}


def _summarize_new_block_ids(
    new_block_ids: list[tuple[list[int], ...] | None],
) -> list[list[int] | None]:
    summary: list[list[int] | None] = []
    for req_block_ids in new_block_ids:
        if req_block_ids is None:
            summary.append(None)
            continue
        summary.append([len(group_ids) for group_ids in req_block_ids])
    return summary


def _dump_edge_decode_state(
    runtime: "EdgeRuntime",
    *,
    req_id: str,
    session,
    step_index: int,
    scheduler_output,
) -> None:
    engine = runtime.engine
    if engine is None:
        raise RuntimeError("edge runtime engine is not initialized")

    model_runner = engine.model_runner
    state = model_runner.execute_model_state
    if state is None:
        raise RuntimeError("edge decode state is unavailable before sampling")

    total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

    payload = {
        "kind": "edge_decode_state",
        "req_id": req_id,
        "step_index": step_index,
        "session_total_len": session.total_len,
        "session_num_computed_tokens": session.num_computed_tokens,
        "scheduler_num_scheduled_tokens": dict(scheduler_output.num_scheduled_tokens),
        "scheduler_total_num_scheduled_tokens": total_num_scheduled_tokens,
        "scheduler_cached_num_computed_tokens": list(
            scheduler_output.scheduled_cached_reqs.num_computed_tokens
        ),
        "scheduler_cached_num_output_tokens": list(
            scheduler_output.scheduled_cached_reqs.num_output_tokens
        ),
        "scheduler_cached_new_block_id_counts": _summarize_new_block_ids(
            scheduler_output.scheduled_cached_reqs.new_block_ids
        ),
    }

    if hasattr(state, "input_batch"):
        input_batch = state.input_batch
        req_state_index = model_runner.req_states.req_id_to_index[session.req_id]
        num_reqs = input_batch.num_reqs
        payload.update(
            {
                "runner": "v2",
                "input_batch_num_reqs": num_reqs,
                "input_batch_req_ids": list(input_batch.req_ids),
                "input_batch_num_computed_tokens": int(
                    model_runner.req_states.num_computed_tokens.gpu[
                        req_state_index
                    ].item()
                ),
                "query_start_loc": input_batch.query_start_loc[
                    : num_reqs + 1
                ].cpu().tolist(),
                "seq_lens": input_batch.seq_lens[:num_reqs].cpu().tolist(),
                "positions": input_batch.positions[
                    :total_num_scheduled_tokens
                ].cpu().tolist(),
                "input_ids": input_batch.input_ids[
                    :total_num_scheduled_tokens
                ].cpu().tolist(),
                "block_table_num_blocks": model_runner.block_tables.num_blocks.np[
                    :, req_state_index
                ].tolist(),
            }
        )
    else:
        input_batch = model_runner.input_batch
        req_index = input_batch.req_id_to_index[session.req_id]
        num_reqs = input_batch.num_reqs
        payload.update(
            {
                "runner": "v1",
                "input_batch_num_reqs": num_reqs,
                "input_batch_req_ids": list(input_batch.req_ids),
                "input_batch_num_computed_tokens": int(
                    input_batch.num_computed_tokens_cpu[req_index]
                ),
                "query_start_loc": model_runner.query_start_loc.cpu[
                    : num_reqs + 1
                ].tolist(),
                "seq_lens": model_runner.seq_lens.cpu[:num_reqs].tolist(),
                "positions": model_runner.positions.cpu[
                    :total_num_scheduled_tokens
                ].tolist(),
                "input_ids": model_runner.input_ids.cpu[
                    :total_num_scheduled_tokens
                ].tolist(),
                "block_table_num_blocks": input_batch.block_table.num_blocks.np[
                    :, req_index
                ].tolist(),
            }
        )
    print("EDGE_DECODE_STATE " + json.dumps(payload, ensure_ascii=False))


def ensure_cuda_platform_if_needed() -> None:
    platform = platforms.current_platform
    if torch.cuda.is_available() and platform.device_type == "cpu":
        platforms.current_platform = CudaPlatform()
        for module in tuple(sys.modules.values()):
            if module is None or module is platforms:
                continue
            if hasattr(module, "current_platform"):
                try:
                    setattr(module, "current_platform", platforms.current_platform)
                except Exception:
                    pass
    if torch.cuda.is_available():
        try:
            setattr(
                platforms.current_platform,
                "opaque_attention_op",
                lambda *args, **kwargs: False,
            )
        except Exception:
            pass


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")


def _build_vllm_config(args: argparse.Namespace):
    from vllm import EngineArgs
    from vllm.usage.usage_lib import UsageContext

    ensure_cuda_platform_if_needed()
    engine_args = EngineArgs(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=_resolved_max_model_len(args),
        enforce_eager=args.enforce_eager,
        enable_prefix_caching=False,
    )
    vllm_config = engine_args.create_engine_config(UsageContext.LLM_CLASS)
    if args.enforce_eager:
        vllm_config.compilation_config.custom_ops = ["none"]
        vllm_config.compilation_config.pass_config.fuse_norm_quant = False
        vllm_config.compilation_config.pass_config.fuse_act_quant = False
    return vllm_config


class EdgeRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.edge_model_runner = args.edge_model_runner
        self.vllm_config = _build_vllm_config(args)
        self._tempfile: tempfile.NamedTemporaryFile[str] | None = None
        self.worker: Worker | None = None
        self.engine: EdgeDecodeEngine | EdgeDecodeEngineV1 | None = None
        self._old_use_v2_model_runner = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")

    def __enter__(self) -> "EdgeRuntime":
        require_cuda()
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = (
            "1" if self.edge_model_runner == "v2" else "0"
        )
        envs.disable_envs_cache()
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        self._tempfile = tempfile.NamedTemporaryFile(delete=False)
        distributed_init_method = f"file://{self._tempfile.name}"
        self.worker = Worker(
            vllm_config=self.vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method=distributed_init_method,
        )
        with set_current_vllm_config(self.vllm_config):
            self.worker.init_device()
            self.worker.load_model()
            platforms.current_platform.update_block_size_for_backend(
                self.vllm_config
            )
            kv_cache_specs = [self.worker.get_kv_cache_spec()]
            available_memory = [self.worker.determine_available_memory()]
            kv_cache_configs = get_kv_cache_configs(
                self.vllm_config,
                kv_cache_specs,
                available_memory,
            )
            scheduler_kv_cache_config = generate_scheduler_kv_cache_config(
                kv_cache_configs
            )
            self.vllm_config.cache_config.num_gpu_blocks = (
                scheduler_kv_cache_config.num_blocks
            )
            if scheduler_kv_cache_config.kv_cache_groups:
                self.vllm_config.cache_config.block_size = min(
                    group.kv_cache_spec.block_size
                    for group in scheduler_kv_cache_config.kv_cache_groups
                )
            self.vllm_config.validate_block_size()
            self.worker.initialize_from_config(kv_cache_configs[0])
            self.worker.compile_or_warm_up_model()

        kv_cache_manager = KVCacheManager(
            kv_cache_config=scheduler_kv_cache_config,
            max_model_len=self.vllm_config.model_config.max_model_len,
            hash_block_size=self.vllm_config.cache_config.block_size,
            enable_caching=self.vllm_config.cache_config.enable_prefix_caching,
            use_eagle=False,
            log_stats=False,
            enable_kv_cache_events=False,
            dcp_world_size=self.vllm_config.parallel_config.decode_context_parallel_size,
            pcp_world_size=self.vllm_config.parallel_config.prefill_context_parallel_size,
        )
        self.engine = _make_edge_engine(
            args=self.args,
            vllm_config=self.vllm_config,
            worker=self.worker,
            kv_cache_manager=kv_cache_manager,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.worker is not None:
            self.worker.shutdown()
        cleanup_dist_env_and_memory()
        if self._tempfile is not None:
            try:
                Path(self._tempfile.name).unlink(missing_ok=True)
            except OSError:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self._old_use_v2_model_runner is None:
            os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
        else:
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = self._old_use_v2_model_runner
        envs.disable_envs_cache()


def _make_edge_engine(
    *,
    args: argparse.Namespace,
    vllm_config,
    worker: Worker,
    kv_cache_manager: KVCacheManager,
) -> EdgeDecodeEngine | EdgeDecodeEngineV1:
    sampler = worker.model_runner.sampler
    if sampler is None:
        raise RuntimeError("worker model runner sampler is not initialized")

    scheduler = EdgeSchedulerAdapter(kv_cache_manager=kv_cache_manager)
    if args.edge_model_runner == "v1":
        return EdgeDecodeEngineV1(
            vllm_config=vllm_config,
            worker=worker,
            scheduler=scheduler,
            state_bridge=EdgeStateBridgeV1(),
            draft_sampler=DSSDEdgeDraftSamplerV1(sampler),
        )

    return EdgeDecodeEngine(
        vllm_config=vllm_config,
        worker=worker,
        scheduler=scheduler,
        state_bridge=EdgeStateBridge(),
        draft_sampler=DSSDEdgeDraftSampler(sampler),
    )


def _make_verifier_engine(
    *,
    args: argparse.Namespace,
    vllm_config,
    worker: Worker,
    scheduler,
) -> VerifierDecodeEngine | VerifierDecodeEngineV1:
    sampler = worker.model_runner.sampler
    if sampler is None:
        raise RuntimeError("worker model runner sampler is not initialized")

    if args.verifier_model_runner == "v1":
        return VerifierDecodeEngineV1(
            vllm_config=vllm_config,
            worker=worker,
            scheduler=scheduler,
            state_bridge=VerifierStateBridgeV1(),
            verifier_sampler=DSSDVerifierSamplerV1(sampler),
        )

    return VerifierDecodeEngine(
        vllm_config=vllm_config,
        worker=worker,
        scheduler=scheduler,
        state_bridge=VerifierStateBridge(),
        verifier_sampler=DSSDVerifierSampler(
            sampler=sampler,
            num_speculative_steps=worker.model_runner.num_speculative_steps,
        ),
    )


class VerifierRuntime:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.verifier_model_runner = args.verifier_model_runner
        self.vllm_config = _build_vllm_config(args)
        self._tempfile: tempfile.NamedTemporaryFile[str] | None = None
        self.worker: Worker | None = None
        self.engine: VerifierDecodeEngine | VerifierDecodeEngineV1 | None = None
        self._old_use_v2_model_runner = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")

    def __enter__(self) -> "VerifierRuntime":
        require_cuda()
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = (
            "1" if self.verifier_model_runner == "v2" else "0"
        )
        envs.disable_envs_cache()
        self._tempfile = tempfile.NamedTemporaryFile(delete=False)
        distributed_init_method = f"file://{self._tempfile.name}"
        self.worker = Worker(
            vllm_config=self.vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method=distributed_init_method,
            is_driver_worker=True,
        )
        with set_current_vllm_config(self.vllm_config):
            self.worker.init_device()
            self.worker.load_model()
            available_memory = [self.worker.determine_available_memory()]
            kv_cache_configs = get_kv_cache_configs(
                self.vllm_config,
                [self.worker.get_kv_cache_spec()],
                available_memory,
            )
            scheduler_kv_cache_config = generate_scheduler_kv_cache_config(
                kv_cache_configs
            )
            self.worker.initialize_from_config(kv_cache_configs[0])
            self.worker.compile_or_warm_up_model()

        kv_cache_manager = KVCacheManager(
            kv_cache_config=scheduler_kv_cache_config,
            max_model_len=self.vllm_config.model_config.max_model_len,
            hash_block_size=self.vllm_config.cache_config.block_size,
            enable_caching=self.vllm_config.cache_config.enable_prefix_caching,
            use_eagle=False,
            log_stats=False,
            enable_kv_cache_events=False,
            dcp_world_size=self.vllm_config.parallel_config.decode_context_parallel_size,
            pcp_world_size=self.vllm_config.parallel_config.prefill_context_parallel_size,
        )
        hash_fn = get_hash_fn_by_name(
            self.vllm_config.cache_config.prefix_caching_hash_algo
        )
        init_none_hash(hash_fn)
        block_hasher = get_request_block_hasher(
            self.vllm_config.cache_config.block_size,
            hash_fn,
        )
        self.engine = _make_verifier_engine(
            args=self.args,
            vllm_config=self.vllm_config,
            worker=self.worker,
            scheduler=BenchmarkVerifierSchedulerAdapter(
                kv_cache_manager=kv_cache_manager,
                block_hasher=block_hasher,
            ),
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.worker is not None:
            self.worker.shutdown()
        cleanup_dist_env_and_memory()
        if self._tempfile is not None:
            try:
                Path(self._tempfile.name).unlink(missing_ok=True)
            except OSError:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self._old_use_v2_model_runner is None:
            os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
        else:
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = self._old_use_v2_model_runner
        envs.disable_envs_cache()


def _run_profiled_edge_request(
    runtime: EdgeRuntime,
    *,
    req_id: str,
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
    dump_decode_state_steps: int = 0,
) -> tuple[list[int], dict[str, float]]:
    engine = runtime.engine
    if engine is None:
        raise RuntimeError("edge runtime engine is not initialized")

    phase_seconds = _new_phase_seconds()
    output_token_ids: list[int] = []
    session = None
    try:
        session, phase_seconds["open_session"] = _timed_phase(
            lambda: engine.open_session(
                req_id=req_id,
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
            )
        )

        max_tokens = sampling_params.max_tokens or 0
        if max_tokens <= 0:
            return output_token_ids, phase_seconds

        processed_logits = torch.empty(
            (1, engine.vocab_size),
            dtype=torch.float32,
            device=engine.model_runner.device,
        )

        def _prefill_execute() -> None:
            engine._execute(engine.scheduler.build_prefill_step(session))
            session.num_computed_tokens = session.prompt_len
            session.total_len = session.prompt_len

        _, phase_seconds["prefill_execute"] = _timed_phase(_prefill_execute)
        bootstrap_result, phase_seconds["bootstrap"] = _timed_phase(
            lambda: engine._sample_with_draft_sampler(session, processed_logits)
        )
        next_token_id, _q_value = bootstrap_result
        del _q_value
        output_token_ids = [next_token_id]

        while len(output_token_ids) < max_tokens:
            decode_step_index = len(output_token_ids)

            def _decode_schedule():
                engine.state_bridge.prepare_next_decode(
                    session,
                    output_token_ids[-1],
                    engine.model_runner,
                )
                return engine.scheduler.build_decode_step(session)

            scheduler_output, decode_schedule_elapsed = _timed_phase(
                _decode_schedule
            )
            phase_seconds["decode_schedule"] += decode_schedule_elapsed
            _, decode_model_elapsed = _timed_phase(
                lambda: engine._execute(scheduler_output)
            )
            phase_seconds["decode_model"] += decode_model_elapsed
            if decode_step_index <= dump_decode_state_steps:
                _dump_edge_decode_state(
                    runtime,
                    req_id=req_id,
                    session=session,
                    step_index=decode_step_index,
                    scheduler_output=scheduler_output,
                )
            decode_result, decode_postprocess_elapsed = _timed_phase(
                lambda: engine._sample_with_draft_sampler(session, processed_logits)
            )
            phase_seconds["decode_postprocess"] += decode_postprocess_elapsed
            next_token_id, _q_value = decode_result
            del _q_value
            output_token_ids.append(next_token_id)
            if engine._should_stop_local_generation(
                sampling_params,
                next_token_id,
            ):
                break

        return output_token_ids, phase_seconds
    finally:
        if session is not None:
            _, phase_seconds["close_session"] = _timed_phase(
                lambda: engine.close_session(session)
            )


def _run_profiled_verifier_request(
    runtime: VerifierRuntime,
    *,
    req_id: str,
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
    dump_decode_state_steps: int = 0,
) -> tuple[list[int], dict[str, float]]:
    del dump_decode_state_steps
    if runtime.verifier_model_runner == "v1":
        return _run_profiled_verifier_v1_request(
            runtime,
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )

    engine = runtime.engine
    if engine is None:
        raise RuntimeError("verifier runtime engine is not initialized")

    phase_seconds = _new_phase_seconds()
    output_token_ids: list[int] = []
    session = None
    try:
        def _open_session() -> VerifierSession:
            return VerifierSession(
                req_id=req_id,
                prompt_token_ids=list(prompt_token_ids),
                sampling_params=sampling_params,
                block_ids=engine.scheduler.allocate_blocks(
                    req_id=req_id,
                    prompt_token_ids=prompt_token_ids,
                    sampling_params=sampling_params,
                ),
                prompt_len=len(prompt_token_ids),
                token_ids=list(prompt_token_ids),
            )

        session, phase_seconds["open_session"] = _timed_phase(_open_session)

        max_tokens = sampling_params.max_tokens or 0
        if max_tokens <= 0:
            return output_token_ids, phase_seconds

        _, phase_seconds["prefill_execute"] = _timed_phase(
            lambda: engine._execute(engine.scheduler.build_open_session_step(session))
        )

        def _bootstrap() -> int:
            state = engine.model_runner.take_execute_model_state()
            hidden_states = engine._take_hidden_states(state.hidden_states)
            sampler_output = engine.model_runner.sample_without_postprocess(
                hidden_states,
                state.input_batch,
                grammar_output=None,
            )
            if sampler_output.sampled_token_ids.numel() == 0:
                raise RuntimeError("bootstrap stage did not sample any token")

            bootstrap_token_id = int(sampler_output.sampled_token_ids[0, 0].item())
            engine._postprocess_prefill_only(state.input_batch)
            engine.state_bridge.finish_prefill_without_commit(session)
            engine.sessions[req_id] = session
            engine.state_bridge.inject_local_token(
                session,
                bootstrap_token_id,
                engine.model_runner,
                computed_delta=1,
            )
            return bootstrap_token_id

        next_token_id, phase_seconds["bootstrap"] = _timed_phase(_bootstrap)
        output_token_ids = [next_token_id]

        while len(output_token_ids) < max_tokens:
            def _decode_schedule():
                engine.state_bridge.prepare_local_decode(
                    session,
                    output_token_ids[-1],
                    engine.model_runner,
                )
                return engine.scheduler.build_decode_step(session)

            scheduler_output, decode_schedule_elapsed = _timed_phase(
                _decode_schedule
            )
            phase_seconds["decode_schedule"] += decode_schedule_elapsed
            _, decode_model_elapsed = _timed_phase(
                lambda: engine._execute(scheduler_output)
            )
            phase_seconds["decode_model"] += decode_model_elapsed
            next_token_id, decode_postprocess_elapsed = _timed_phase(
                lambda: engine._sample_local_token(session)
            )
            phase_seconds["decode_postprocess"] += decode_postprocess_elapsed
            output_token_ids.append(next_token_id)
            if engine._should_stop_local_generation(
                sampling_params,
                next_token_id,
            ):
                break

        return output_token_ids, phase_seconds
    finally:
        if session is not None:
            _, phase_seconds["close_session"] = _timed_phase(
                lambda: engine.close_session(session)
            )


def _run_profiled_verifier_v1_request(
    runtime: VerifierRuntime,
    *,
    req_id: str,
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
) -> tuple[list[int], dict[str, float]]:
    engine = runtime.engine
    if engine is None:
        raise RuntimeError("verifier runtime engine is not initialized")

    phase_seconds = _new_phase_seconds()
    output_token_ids: list[int] = []
    session = None
    try:
        def _open_session() -> VerifierSession:
            return VerifierSession(
                req_id=req_id,
                prompt_token_ids=list(prompt_token_ids),
                sampling_params=sampling_params,
                block_ids=engine.scheduler.allocate_blocks(
                    req_id=req_id,
                    prompt_token_ids=prompt_token_ids,
                    sampling_params=sampling_params,
                ),
                prompt_len=len(prompt_token_ids),
                token_ids=list(prompt_token_ids),
            )

        session, phase_seconds["open_session"] = _timed_phase(_open_session)

        max_tokens = sampling_params.max_tokens or 0
        if max_tokens <= 0:
            return output_token_ids, phase_seconds

        _, phase_seconds["prefill_execute"] = _timed_phase(
            lambda: engine._execute(engine.scheduler.build_open_session_step(session))
        )

        def _bootstrap() -> int:
            state = engine.model_runner.take_execute_model_state()
            bootstrap_token_id = engine.verifier_sampler.sample_bootstrap(
                logits=state.logits,
                sampling_metadata=engine.model_runner.input_batch.sampling_metadata,
            )
            engine.state_bridge.finish_prefill_without_commit(
                session,
                engine.model_runner,
            )
            engine.sessions[req_id] = session
            engine.state_bridge.inject_local_token(
                session,
                bootstrap_token_id,
                engine.model_runner,
                computed_delta=1,
            )
            return bootstrap_token_id

        next_token_id, phase_seconds["bootstrap"] = _timed_phase(_bootstrap)
        output_token_ids = [next_token_id]

        while len(output_token_ids) < max_tokens:
            def _decode_schedule():
                engine.state_bridge.prepare_local_decode(
                    session,
                    output_token_ids[-1],
                    engine.model_runner,
                )
                return engine.scheduler.build_decode_step(session)

            scheduler_output, decode_schedule_elapsed = _timed_phase(
                _decode_schedule
            )
            phase_seconds["decode_schedule"] += decode_schedule_elapsed
            _, decode_model_elapsed = _timed_phase(
                lambda: engine._execute(scheduler_output)
            )
            phase_seconds["decode_model"] += decode_model_elapsed

            def _decode_postprocess() -> int:
                state = engine.model_runner.take_execute_model_state()
                token_id = engine.verifier_sampler.sample_bootstrap(
                    logits=state.logits,
                    sampling_metadata=(
                        engine.model_runner.input_batch.sampling_metadata
                    ),
                )
                engine.state_bridge.commit_local_token(
                    session,
                    token_id,
                    engine.model_runner,
                )
                return token_id

            next_token_id, decode_postprocess_elapsed = _timed_phase(
                _decode_postprocess
            )
            phase_seconds["decode_postprocess"] += decode_postprocess_elapsed
            output_token_ids.append(next_token_id)
            if engine._should_stop_local_generation(
                sampling_params,
                next_token_id,
            ):
                break

        return output_token_ids, phase_seconds
    finally:
        if session is not None:
            _, phase_seconds["close_session"] = _timed_phase(
                lambda: engine.close_session(session)
            )


def _run_profiled_runtime_request(
    *,
    engine_name: str,
    runtime,
    req_id: str,
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
    dump_decode_state_steps: int = 0,
) -> tuple[list[int], dict[str, float]]:
    if engine_name == "edge":
        return _run_profiled_edge_request(
            runtime,
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            dump_decode_state_steps=dump_decode_state_steps,
        )
    if engine_name == "verifier":
        return _run_profiled_verifier_request(
            runtime,
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            dump_decode_state_steps=dump_decode_state_steps,
        )
    raise ValueError(f"phase timing is not supported for engine: {engine_name}")


def benchmark_native(
    args: argparse.Namespace,
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
) -> BenchmarkResult:
    require_cuda()
    old_value = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    if args.native_model_runner == "v2":
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    else:
        os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
    envs.disable_envs_cache()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    llm = None
    prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
    try:
        llm = LLM(
            model=args.model,
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
            tensor_parallel_size=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=_resolved_max_model_len(args),
            enforce_eager=args.enforce_eager,
            enable_prefix_caching=False,
        )
        for warmup_idx in range(args.warmup_iters):
            del warmup_idx
            llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
        synchronize_if_needed()

        total_tokens = 0
        total_seconds = 0.0
        for _ in range(args.benchmark_iters):
            synchronize_if_needed()
            t0 = time.perf_counter()
            outputs = llm.generate([prompt], sampling_params=sampling_params, use_tqdm=False)
            synchronize_if_needed()
            elapsed = time.perf_counter() - t0
            total_seconds += elapsed
            total_tokens += len(outputs[0].outputs[0].token_ids)

        return BenchmarkResult(
            engine="native",
            model=args.model,
            prompt_tokens=len(prompt_token_ids),
            requested_output_tokens=args.max_new_tokens,
            generated_tokens=total_tokens,
            warmup_iters=args.warmup_iters,
            benchmark_iters=args.benchmark_iters,
            total_seconds=total_seconds,
            avg_seconds=total_seconds / max(args.benchmark_iters, 1),
            tokens_per_second=total_tokens / total_seconds,
        )
    finally:
        if llm is not None:
            del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if old_value is None:
            os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
        else:
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = old_value
        envs.disable_envs_cache()


def _benchmark_runtime_engine(
    *,
    engine_name: str,
    model: str,
    prompt_token_ids: list[int],
    warmup_iters: int,
    benchmark_iters: int,
    sampling_params: SamplingParams,
    runtime,
    phase_timing: bool,
    dump_decode_state_steps: int,
) -> BenchmarkResult:
    use_profiled_runtime = phase_timing or dump_decode_state_steps > 0
    for warmup_idx in range(warmup_iters):
        if use_profiled_runtime:
            _run_profiled_runtime_request(
                engine_name=engine_name,
                runtime=runtime,
                req_id=f"warmup-{warmup_idx}",
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
                dump_decode_state_steps=dump_decode_state_steps,
            )
        else:
            runtime.engine.generate_local(
                req_id=f"warmup-{warmup_idx}",
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
            )
    synchronize_if_needed()

    total_tokens = 0
    total_seconds = 0.0
    phase_seconds = _new_phase_seconds() if phase_timing else None
    for iter_idx in range(benchmark_iters):
        if use_profiled_runtime:
            output_token_ids, request_phase_seconds = _run_profiled_runtime_request(
                engine_name=engine_name,
                runtime=runtime,
                req_id=f"bench-{iter_idx}",
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
                dump_decode_state_steps=dump_decode_state_steps,
            )
            elapsed = sum(request_phase_seconds.values())
            if phase_timing:
                assert phase_seconds is not None
                for phase_name, phase_elapsed in request_phase_seconds.items():
                    phase_seconds[phase_name] += phase_elapsed
        else:
            synchronize_if_needed()
            t0 = time.perf_counter()
            output_token_ids = runtime.engine.generate_local(
                req_id=f"bench-{iter_idx}",
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
            )
            synchronize_if_needed()
            elapsed = time.perf_counter() - t0
        total_seconds += elapsed
        total_tokens += len(output_token_ids)

    return BenchmarkResult(
        engine=engine_name,
        model=model,
        prompt_tokens=len(prompt_token_ids),
        requested_output_tokens=sampling_params.max_tokens or 0,
        generated_tokens=total_tokens,
        warmup_iters=warmup_iters,
        benchmark_iters=benchmark_iters,
        total_seconds=total_seconds,
        avg_seconds=total_seconds / max(benchmark_iters, 1),
        tokens_per_second=total_tokens / total_seconds,
        phase_seconds=phase_seconds,
    )


def benchmark_edge(
    args: argparse.Namespace,
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
) -> BenchmarkResult:
    with EdgeRuntime(args) as runtime:
        return _benchmark_runtime_engine(
            engine_name="edge",
            model=args.model,
            prompt_token_ids=prompt_token_ids,
            warmup_iters=args.warmup_iters,
            benchmark_iters=args.benchmark_iters,
            sampling_params=sampling_params,
            runtime=runtime,
            phase_timing=args.phase_timing,
            dump_decode_state_steps=args.dump_decode_state,
        )


def benchmark_verifier(
    args: argparse.Namespace,
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
) -> BenchmarkResult:
    with VerifierRuntime(args) as runtime:
        return _benchmark_runtime_engine(
            engine_name="verifier",
            model=args.model,
            prompt_token_ids=prompt_token_ids,
            warmup_iters=args.warmup_iters,
            benchmark_iters=args.benchmark_iters,
            sampling_params=sampling_params,
            runtime=runtime,
            phase_timing=args.phase_timing,
            dump_decode_state_steps=args.dump_decode_state,
        )


def _loads_trailing_json(stdout: str) -> dict[str, Any]:
    text = stdout.strip()
    start = text.rfind("{")
    decoder = json.JSONDecoder()
    while start >= 0:
        try:
            payload, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.rfind("{", 0, start)
            continue
        if text[end:].strip() == "" and isinstance(payload, dict):
            return payload
        start = text.rfind("{", 0, start)
    raise RuntimeError(
        "could not parse benchmark JSON payload from subprocess stdout\n"
        f"stdout tail:\n{text[-2000:]}"
    )


def run_subprocess_for_engine(
    args: argparse.Namespace,
    engine: str,
) -> BenchmarkResult:
    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--engine",
        engine,
        "--model",
        args.model,
        "--prompt-len",
        str(args.prompt_len),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--warmup-iters",
        str(args.warmup_iters),
        "--benchmark-iters",
        str(args.benchmark_iters),
        "--dtype",
        args.dtype,
        "--native-model-runner",
        args.native_model_runner,
        "--edge-model-runner",
        args.edge_model_runner,
        "--verifier-model-runner",
        args.verifier_model_runner,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--json-only",
    ]
    if args.trust_remote_code:
        cmd.append("--trust-remote-code")
    cmd.append("--enforce-eager" if args.enforce_eager else "--no-enforce-eager")
    if args.honor_eos:
        cmd.append("--honor-eos")
    if args.phase_timing:
        cmd.append("--phase-timing")
    if args.dump_decode_state > 0:
        cmd.extend(["--dump-decode-state", str(args.dump_decode_state)])
    completed = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = _loads_trailing_json(completed.stdout)
    return BenchmarkResult(**payload)


def print_results(results: list[BenchmarkResult], *, json_only: bool) -> None:
    if json_only:
        print(json.dumps(_json_payload(results), ensure_ascii=False))
        return

    print(f"model: {results[0].model}")
    print(
        f"prompt_tokens: {results[0].prompt_tokens}, "
        f"max_new_tokens: {results[0].requested_output_tokens}, "
        f"warmup: {results[0].warmup_iters}, "
        f"iters: {results[0].benchmark_iters}"
    )
    print()
    header = (
        f"{'engine':<10} {'avg_s':>10} {'total_s':>10} "
        f"{'gen_toks':>10} {'tok/s':>12}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.engine:<10} {result.avg_seconds:>10.4f} "
            f"{result.total_seconds:>10.4f} {result.generated_tokens:>10d} "
            f"{result.tokens_per_second:>12.2f}"
        )
        if result.phase_seconds is not None:
            avg_phase_parts = [
                f"{phase_name}={phase_elapsed / max(result.benchmark_iters, 1):.4f}s"
                for phase_name, phase_elapsed in result.phase_seconds.items()
            ]
            print(f"  phases(avg_s): {' '.join(avg_phase_parts)}")
    if any(result.engine == "native" for result in results):
        native = next(result for result in results if result.engine == "native")
        for result in results:
            if result.engine == "native":
                continue
            print(
                f"speedup {result.engine}/native: "
                f"{result.tokens_per_second / native.tokens_per_second:.4f}x"
            )


def _run_single_engine(args: argparse.Namespace) -> BenchmarkResult:
    require_cuda()
    prompt_token_ids = _build_prompt_token_ids(
        args.model,
        args.prompt_len,
        trust_remote_code=args.trust_remote_code,
    )
    sampling_params = build_sampling_params(args)
    if args.engine == "native":
        return benchmark_native(args, prompt_token_ids, sampling_params)
    if args.engine == "edge":
        return benchmark_edge(args, prompt_token_ids, sampling_params)
    return benchmark_verifier(args, prompt_token_ids, sampling_params)


def main() -> None:
    args = build_parser().parse_args()
    if args.engine == "all":
        results = [
            run_subprocess_for_engine(args, "native"),
            run_subprocess_for_engine(args, "edge"),
            run_subprocess_for_engine(args, "verifier"),
        ]
    else:
        results = [_run_single_engine(args)]
    maybe_write_results_json(args, results)
    print_results(results, json_only=args.json or args.json_only)


if __name__ == "__main__":
    main()
