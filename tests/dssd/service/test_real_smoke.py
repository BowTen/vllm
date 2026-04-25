# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import subprocess
import tempfile
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm import envs
from vllm.config import set_current_vllm_config
from vllm.dssd.edge import (
    DSSDEdgeDraftSampler,
    DSSDEdgeDraftSamplerV1,
    EdgeDecodeEngine,
    EdgeDecodeEngineV1,
    EdgeSchedulerAdapter,
    EdgeStateBridge,
    EdgeStateBridgeV1,
    EdgeSession,
)
from vllm.dssd.protocol import VerifyRoundRequest
from vllm.dssd.verifier import (
    DSSDVerifierSampler,
    VerifierDecodeEngine,
    VerifierSchedulerAdapter,
    VerifierStateBridge,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_configs,
)
from vllm.v1.worker.gpu_worker import Worker


_REMOTE_SMOKE_MODEL = "hmellor/tiny-random-LlamaForCausalLM"
_LOCAL_SMOKE_MODEL_CANDIDATES = (
    _REMOTE_SMOKE_MODEL,
    "facebook/opt-125m",
)
_REAL_SMOKE_KV_CACHE_MEMORY_BYTES = 128 * 1024 * 1024
_REAL_SMOKE_GPU_MEMORY_UTILIZATION = 0.01
_REAL_SMOKE_MAX_NUM_BATCHED_TOKENS = 64
_REAL_SMOKE_MAX_NUM_SEQS = 2
_REAL_SMOKE_MIN_FREE_MIB = 2048
_REAL_SPLIT_SMOKE_MIN_FREE_MIB = 3072


class _RealRuntimeUnavailable(RuntimeError):
    pass


def _current_free_gpu_memory_mib() -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    first_line = output.splitlines()[0].strip()
    if not first_line:
        return None
    try:
        return int(first_line)
    except ValueError:
        return None


def _skip_if_insufficient_free_gpu_memory(min_free_mib: int, label: str) -> None:
    free_mib = _current_free_gpu_memory_mib()
    if free_mib is None:
        return
    if free_mib < min_free_mib:
        pytest.skip(
            f"insufficient GPU memory for {label}: "
            f"{free_mib} MiB free < {min_free_mib} MiB required"
        )


def _snapshot_has_required_files(snapshot_dir: Path) -> bool:
    if not snapshot_dir.is_dir() or not (snapshot_dir / "config.json").is_file():
        return False

    has_weights = any(snapshot_dir.glob("*.safetensors")) or any(
        snapshot_dir.glob("*.bin")
    )
    has_tokenizer = (
        (snapshot_dir / "tokenizer.json").is_file()
        or (snapshot_dir / "tokenizer.model").is_file()
        or (
            (snapshot_dir / "vocab.json").is_file()
            and (snapshot_dir / "merges.txt").is_file()
        )
    )
    return has_weights and has_tokenizer


def _find_local_snapshot(model_id: str) -> str | None:
    repo_cache = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{model_id.replace('/', '--')}"
    )
    snapshots_dir = repo_cache / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    ref_path = repo_cache / "refs" / "main"
    if ref_path.is_file():
        preferred_snapshot = snapshots_dir / ref_path.read_text().strip()
        if _snapshot_has_required_files(preferred_snapshot):
            return str(preferred_snapshot)

    for snapshot_dir in sorted(snapshots_dir.iterdir()):
        if _snapshot_has_required_files(snapshot_dir):
            return str(snapshot_dir)

    return None


def _resolve_smoke_model() -> str:
    for model_id in _LOCAL_SMOKE_MODEL_CANDIDATES:
        local_snapshot = _find_local_snapshot(model_id)
        if local_snapshot is not None:
            return local_snapshot

    return _REMOTE_SMOKE_MODEL


def _init_real_runtime():
    from vllm.distributed import cleanup_dist_env_and_memory
    import torch

    worker = None
    runtime = None
    for attempt in range(3):
        engine_args = EngineArgs(
            model=_resolve_smoke_model(),
            enforce_eager=True,
            async_scheduling=False,
            max_model_len=64,
            gpu_memory_utilization=_REAL_SMOKE_GPU_MEMORY_UTILIZATION,
            kv_cache_memory_bytes=_REAL_SMOKE_KV_CACHE_MEMORY_BYTES,
            max_num_batched_tokens=_REAL_SMOKE_MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=_REAL_SMOKE_MAX_NUM_SEQS,
        )
        vllm_config = engine_args.create_engine_config()
        with tempfile.NamedTemporaryFile() as tmp_file, set_current_vllm_config(
            vllm_config
        ):
            worker = Worker(
                vllm_config=vllm_config,
                local_rank=0,
                rank=0,
                distributed_init_method=f"file://{tmp_file.name}",
                is_driver_worker=True,
            )
            try:
                worker.init_device()
                worker.load_model()

                available_memory = [worker.determine_available_memory()]
                kv_cache_configs = get_kv_cache_configs(
                    vllm_config,
                    [worker.get_kv_cache_spec()],
                    available_memory,
                )
                scheduler_kv_cache_config = generate_scheduler_kv_cache_config(
                    kv_cache_configs
                )
                worker.initialize_from_config(kv_cache_configs[0])

                kv_cache_manager = KVCacheManager(
                    kv_cache_config=scheduler_kv_cache_config,
                    max_model_len=vllm_config.model_config.max_model_len,
                    hash_block_size=vllm_config.cache_config.block_size,
                )
                runtime = SimpleNamespace(
                    worker=worker,
                    vllm_config=vllm_config,
                    kv_cache_manager=kv_cache_manager,
                )
                break
            except torch.AcceleratorError as exc:
                message = str(exc)
                if "out of memory" not in message.lower():
                    if worker is not None:
                        worker.shutdown()
                    cleanup_dist_env_and_memory()
                    raise
                if worker is not None:
                    worker.shutdown()
                    worker = None
                cleanup_dist_env_and_memory()
                raise _RealRuntimeUnavailable(message) from exc
            except ValueError as exc:
                message = str(exc)
                if "less than desired GPU memory utilization" in message:
                    if worker is not None:
                        worker.shutdown()
                        worker = None
                    cleanup_dist_env_and_memory()
                    raise _RealRuntimeUnavailable(message) from exc
                if (
                    "No available memory for the cache blocks" not in message
                    or attempt == 2
                ):
                    if worker is not None:
                        worker.shutdown()
                    cleanup_dist_env_and_memory()
                    raise
                worker.shutdown()
                worker = None
                cleanup_dist_env_and_memory()
                continue
            except Exception:
                if worker is not None:
                    worker.shutdown()
                    worker = None
                cleanup_dist_env_and_memory()
                raise

    if runtime is None:
        raise RuntimeError("failed to initialize real DSSD runtime")
    return runtime


@pytest.fixture
def real_dssd_runtime():
    if not pytest.importorskip("torch").cuda.is_available():
        pytest.skip("requires cuda")
    if not current_platform.is_cuda():
        pytest.skip("requires a CUDA-resolved vLLM runtime")
    _skip_if_insufficient_free_gpu_memory(
        _REAL_SMOKE_MIN_FREE_MIB,
        "real smoke",
    )

    old_use_v2_model_runner = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    envs.disable_envs_cache()

    from vllm.distributed import cleanup_dist_env_and_memory

    runtime = None
    try:
        try:
            runtime = _init_real_runtime()
        except _RealRuntimeUnavailable as exc:
            pytest.skip(f"insufficient GPU memory for real smoke: {exc}")
        yield runtime
    finally:
        if runtime is not None:
            runtime.worker.shutdown()
        cleanup_dist_env_and_memory()
        if old_use_v2_model_runner is None:
            os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
        else:
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = old_use_v2_model_runner
        envs.disable_envs_cache()


@pytest.fixture
def real_dssd_runtime_v1():
    if not pytest.importorskip("torch").cuda.is_available():
        pytest.skip("requires cuda")
    if not current_platform.is_cuda():
        pytest.skip("requires a CUDA-resolved vLLM runtime")
    _skip_if_insufficient_free_gpu_memory(
        _REAL_SMOKE_MIN_FREE_MIB,
        "real smoke",
    )

    old_use_v2_model_runner = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    envs.disable_envs_cache()

    from vllm.distributed import cleanup_dist_env_and_memory

    runtime = None
    try:
        try:
            runtime = _init_real_runtime()
        except _RealRuntimeUnavailable as exc:
            pytest.skip(f"insufficient GPU memory for real smoke: {exc}")
        yield runtime
    finally:
        if runtime is not None:
            runtime.worker.shutdown()
        cleanup_dist_env_and_memory()
        if old_use_v2_model_runner is None:
            os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
        else:
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = old_use_v2_model_runner
        envs.disable_envs_cache()


@pytest.fixture
def real_dssd_runtime_v1_edge_v2_verifier():
    if not pytest.importorskip("torch").cuda.is_available():
        pytest.skip("requires cuda")
    if not current_platform.is_cuda():
        pytest.skip("requires a CUDA-resolved vLLM runtime")
    _skip_if_insufficient_free_gpu_memory(
        _REAL_SMOKE_MIN_FREE_MIB,
        "real smoke",
    )

    old_use_v2_model_runner = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    from vllm.distributed import cleanup_dist_env_and_memory

    edge_runtime = None
    verifier_runtime = None
    try:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        envs.disable_envs_cache()

        try:
            edge_runtime = _init_real_runtime()

            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
            envs.disable_envs_cache()
            verifier_runtime = _init_real_runtime()
        except _RealRuntimeUnavailable as exc:
            if verifier_runtime is not None:
                verifier_runtime.worker.shutdown()
                verifier_runtime = None
            if edge_runtime is not None:
                edge_runtime.worker.shutdown()
                edge_runtime = None
            pytest.skip(f"insufficient GPU memory for real smoke: {exc}")

        yield SimpleNamespace(
            edge_runtime=edge_runtime,
            verifier_runtime=verifier_runtime,
        )
    finally:
        if verifier_runtime is not None:
            verifier_runtime.worker.shutdown()
        if edge_runtime is not None:
            edge_runtime.worker.shutdown()
        cleanup_dist_env_and_memory()
        if old_use_v2_model_runner is None:
            os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
        else:
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = old_use_v2_model_runner
        envs.disable_envs_cache()


@pytest.fixture
def real_split_dssd_runtime():
    if not pytest.importorskip("torch").cuda.is_available():
        pytest.skip("requires cuda")
    if not current_platform.is_cuda():
        pytest.skip("requires a CUDA-resolved vLLM runtime")
    _skip_if_insufficient_free_gpu_memory(
        _REAL_SPLIT_SMOKE_MIN_FREE_MIB,
        "split real smoke",
    )

    old_use_v2_model_runner = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    envs.disable_envs_cache()

    from vllm.distributed import cleanup_dist_env_and_memory

    edge_runtime = None
    verifier_runtime = None
    try:
        try:
            edge_runtime = _init_real_runtime()
            verifier_runtime = _init_real_runtime()
        except _RealRuntimeUnavailable as exc:
            if verifier_runtime is not None:
                verifier_runtime.worker.shutdown()
                verifier_runtime = None
            if edge_runtime is not None:
                edge_runtime.worker.shutdown()
                edge_runtime = None
            pytest.skip(f"insufficient GPU memory for split real smoke: {exc}")
        yield SimpleNamespace(
            edge_runtime=edge_runtime,
            verifier_runtime=verifier_runtime,
        )
    finally:
        if verifier_runtime is not None:
            verifier_runtime.worker.shutdown()
        if edge_runtime is not None:
            edge_runtime.worker.shutdown()
        cleanup_dist_env_and_memory()
        if old_use_v2_model_runner is None:
            os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
        else:
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = old_use_v2_model_runner
        envs.disable_envs_cache()


def test_real_dssd_runtime_v1_edge_v2_verifier_fixture_orders_init_and_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.distributed as distributed

    trace: list[str] = []

    class _FakeWorker:
        def __init__(self, label: str) -> None:
            self.label = label
            self.model_runner = SimpleNamespace()
            if label == "1":
                self.model_runner.vocab_size = None

        def shutdown(self) -> None:
            trace.append(f"shutdown:{self.label}")

    def fake_importorskip(module_name: str):
        assert module_name == "torch"
        return SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True),
        )

    def fake_init_real_runtime():
        label = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
        trace.append(f"init:{label}")
        return SimpleNamespace(
            worker=_FakeWorker(label or "unset"),
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(
                    get_vocab_size=lambda: 7,
                ),
            ),
        )

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "sentinel")
    monkeypatch.setattr(pytest, "importorskip", fake_importorskip)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        sys.modules[__name__],
        "_skip_if_insufficient_free_gpu_memory",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        "_init_real_runtime",
        fake_init_real_runtime,
    )
    monkeypatch.setattr(
        envs,
        "disable_envs_cache",
        lambda: trace.append(
            f"disable_cache:{os.environ.get('VLLM_USE_V2_MODEL_RUNNER')}"
        ),
    )
    monkeypatch.setattr(
        distributed,
        "cleanup_dist_env_and_memory",
        lambda: trace.append("cleanup"),
    )

    fixture_gen = real_dssd_runtime_v1_edge_v2_verifier.__wrapped__()
    runtime_pair = next(fixture_gen)
    assert runtime_pair.edge_runtime.vllm_config.model_config.get_vocab_size() == 7
    assert not hasattr(runtime_pair.edge_runtime.worker.model_runner, "vocab_size")
    assert runtime_pair.verifier_runtime.worker.model_runner.vocab_size is None

    with pytest.raises(StopIteration):
        next(fixture_gen)

    assert trace == [
        "disable_cache:0",
        "init:0",
        "disable_cache:1",
        "init:1",
        "shutdown:1",
        "shutdown:0",
        "cleanup",
        "disable_cache:sentinel",
    ]
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "sentinel"


def test_real_service_transport_smoke_target_only(real_dssd_runtime) -> None:
    from vllm.dssd.service import DSSDVerifierService
    from vllm.dssd.transport import FakeNetwork, InProcessVerifierTransport

    sampler = real_dssd_runtime.worker.model_runner.sampler
    assert sampler is not None

    verifier_engine = VerifierDecodeEngine(
        vllm_config=real_dssd_runtime.vllm_config,
        worker=real_dssd_runtime.worker,
        scheduler=VerifierSchedulerAdapter(
            kv_cache_manager=real_dssd_runtime.kv_cache_manager
        ),
        state_bridge=VerifierStateBridge(),
        verifier_sampler=DSSDVerifierSampler(
            sampler=sampler,
            num_speculative_steps=real_dssd_runtime.worker.model_runner.num_speculative_steps,
        ),
    )
    edge_engine = EdgeDecodeEngine(
        vllm_config=real_dssd_runtime.vllm_config,
        worker=real_dssd_runtime.worker,
        scheduler=EdgeSchedulerAdapter(
            kv_cache_manager=real_dssd_runtime.kv_cache_manager
        ),
        state_bridge=EdgeStateBridge(),
        draft_sampler=DSSDEdgeDraftSampler(sampler),
    )

    verifier_service = DSSDVerifierService(decode_engine=verifier_engine)
    transport = InProcessVerifierTransport(
        verifier_service=verifier_service,
        request_network=FakeNetwork(
            fixed_latency_ms=0.0,
            bandwidth_bytes_per_s=1e9,
        ),
        response_network=FakeNetwork(
            fixed_latency_ms=0.0,
            bandwidth_bytes_per_s=1e9,
        ),
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        ignore_eos=True,
    )
    req_id = "req-1"
    prompt_token_ids = [1, 2, 3]

    opened = transport.open_session(
        req_id=req_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )
    edge_session = EdgeSession(
        req_id=req_id,
        prompt_token_ids=list(prompt_token_ids),
        sampling_params=sampling_params,
        block_ids=([],),
        prompt_len=len(prompt_token_ids),
        num_computed_tokens=len(prompt_token_ids),
        total_len=len(prompt_token_ids),
        token_ids=list(prompt_token_ids),
    )

    bootstrap_token_id = edge_engine.commit_external_token(
        edge_session,
        opened.bootstrap_token_id,
    )
    round_state = edge_engine.draft(
        edge_session,
        first_token_id=bootstrap_token_id,
        gamma=0,
    )
    response = transport.verify_round(
        VerifyRoundRequest(
            req_id=req_id,
            committed_token_id=bootstrap_token_id,
            draft_token_ids=list(round_state.draft_token_ids),
            draft_q_values=list(round_state.draft_q_values),
        )
    )

    assert response.accepted_len == 0
    assert response.bonus_token_id is not None

    committed_token_id = edge_engine.commit_external_token(
        edge_session,
        response.bonus_token_id,
    )
    assert committed_token_id == response.bonus_token_id
    assert edge_session.committed_output_ids() == [
        bootstrap_token_id,
        response.bonus_token_id,
    ]

    transport.close_session(req_id)


def _build_real_components(
    real_dssd_runtime,
    *,
    verifier_gamma: int = 0,
    remote_req_id_factory=None,
):
    return _build_real_components_from_runtimes(
        edge_runtime=real_dssd_runtime,
        verifier_runtime=real_dssd_runtime,
        verifier_gamma=verifier_gamma,
        remote_req_id_factory=remote_req_id_factory,
    )


def _build_real_components_from_runtimes(
    *,
    edge_runtime,
    verifier_runtime,
    edge_model_runner_version: str = "v2",
    verifier_gamma: int = 0,
    remote_req_id_factory=None,
):
    from vllm.dssd.service import DSSDVerifierService
    from vllm.dssd.transport import FakeNetwork, InProcessVerifierTransport

    edge_sampler = edge_runtime.worker.model_runner.sampler
    verifier_sampler = verifier_runtime.worker.model_runner.sampler
    assert edge_sampler is not None
    assert verifier_sampler is not None

    verifier_engine = VerifierDecodeEngine(
        vllm_config=verifier_runtime.vllm_config,
        worker=verifier_runtime.worker,
        scheduler=VerifierSchedulerAdapter(
            kv_cache_manager=verifier_runtime.kv_cache_manager
        ),
        state_bridge=VerifierStateBridge(),
        verifier_sampler=DSSDVerifierSampler(
            sampler=verifier_sampler,
            num_speculative_steps=verifier_gamma,
        ),
    )
    if edge_model_runner_version == "v1":
        edge_engine_cls = EdgeDecodeEngineV1
        state_bridge = EdgeStateBridgeV1()
        draft_sampler = DSSDEdgeDraftSamplerV1(edge_sampler)
    elif edge_model_runner_version == "v2":
        edge_engine_cls = EdgeDecodeEngine
        state_bridge = EdgeStateBridge()
        draft_sampler = DSSDEdgeDraftSampler(edge_sampler)
    else:
        raise ValueError(
            "edge_model_runner_version must be one of {'v1', 'v2'}"
        )

    edge_engine = edge_engine_cls(
        vllm_config=edge_runtime.vllm_config,
        worker=edge_runtime.worker,
        scheduler=EdgeSchedulerAdapter(
            kv_cache_manager=edge_runtime.kv_cache_manager
        ),
        state_bridge=state_bridge,
        draft_sampler=draft_sampler,
    )
    verifier_service = DSSDVerifierService(decode_engine=verifier_engine)
    transport = InProcessVerifierTransport(
        verifier_service=verifier_service,
        request_network=FakeNetwork(
            fixed_latency_ms=0.0,
            bandwidth_bytes_per_s=1e9,
        ),
        response_network=FakeNetwork(
            fixed_latency_ms=0.0,
            bandwidth_bytes_per_s=1e9,
        ),
        remote_req_id_factory=remote_req_id_factory,
    )
    return edge_engine, verifier_engine, transport


def _make_edge_shadow_session(
    *,
    req_id: str,
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
) -> EdgeSession:
    return EdgeSession(
        req_id=req_id,
        prompt_token_ids=list(prompt_token_ids),
        sampling_params=sampling_params,
        block_ids=([],),
        prompt_len=len(prompt_token_ids),
        num_computed_tokens=len(prompt_token_ids),
        total_len=len(prompt_token_ids),
        token_ids=list(prompt_token_ids),
    )


def _decode_via_transport_target_only(
    *,
    transport,
    edge_engine: EdgeDecodeEngine,
    req_id: str,
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
    output_tokens: int,
) -> tuple[list[int], EdgeSession]:
    opened = transport.open_session(
        req_id=req_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )
    edge_session = edge_engine.open_session(
        req_id=req_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )
    token_ids = [
        edge_engine.prefill(
            edge_session,
            bootstrap_token_id=opened.bootstrap_token_id,
        )
    ]

    try:
        while len(token_ids) < output_tokens:
            round_state = edge_engine.draft(
                edge_session,
                first_token_id=token_ids[-1],
                gamma=0,
            )
            response = transport.verify_round(
                VerifyRoundRequest(
                    req_id=req_id,
                    committed_token_id=token_ids[-1],
                    draft_token_ids=list(round_state.draft_token_ids),
                    draft_q_values=list(round_state.draft_q_values),
                )
            )
            assert response.accepted_len == 0
            assert response.bonus_token_id is not None
            token_ids.append(
                edge_engine.commit_external_token(
                    edge_session,
                    response.bonus_token_id,
                )
            )
    finally:
        if req_id in edge_engine.sessions:
            edge_engine.close_session(edge_session)

    return token_ids, edge_session


def _decode_via_direct_verifier_target_only(
    *,
    verifier_engine: VerifierDecodeEngine,
    req_id: str,
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
    output_tokens: int,
) -> list[int]:
    opened = verifier_engine.open_session(
        req_id=req_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )
    session = verifier_engine.sessions[req_id]
    token_ids = [opened.bootstrap_token_id]

    while len(token_ids) < output_tokens:
        response = verifier_engine.verify_round(
            session,
            VerifyRoundRequest(
                req_id=req_id,
                committed_token_id=token_ids[-1],
                draft_token_ids=[],
                draft_q_values=[],
            ),
        )
        assert response.accepted_len == 0
        assert response.bonus_token_id is not None
        token_ids.append(response.bonus_token_id)

    verifier_engine.close_session(session)
    return token_ids


def test_real_service_transport_target_only_matches_direct_verifier(
    real_dssd_runtime,
) -> None:
    edge_engine, verifier_engine, transport = _build_real_components(
        real_dssd_runtime,
        remote_req_id_factory=lambda req_id: f"{req_id}::verifier",
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=5,
        ignore_eos=True,
    )
    prompt_token_ids = [1, 2, 3]

    transport_token_ids, edge_session = _decode_via_transport_target_only(
        transport=transport,
        edge_engine=edge_engine,
        req_id="transport-req",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        output_tokens=5,
    )
    direct_token_ids = _decode_via_direct_verifier_target_only(
        verifier_engine=verifier_engine,
        req_id="direct-req",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        output_tokens=5,
    )

    assert transport_token_ids == direct_token_ids
    assert edge_session.committed_output_ids() == direct_token_ids

    transport.close_session("transport-req")


def test_real_service_transport_smoke_external_draft_round(
    real_dssd_runtime,
) -> None:
    edge_engine, _verifier_engine, transport = _build_real_components(
        real_dssd_runtime,
        verifier_gamma=1,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        ignore_eos=True,
    )
    req_id = "external-draft-req"
    prompt_token_ids = [1, 2, 3]

    opened = transport.open_session(
        req_id=req_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )
    edge_session = _make_edge_shadow_session(
        req_id=req_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )
    bootstrap_token_id = edge_engine.commit_external_token(
        edge_session,
        opened.bootstrap_token_id,
    )
    round_state = edge_engine.draft(
        edge_session,
        first_token_id=bootstrap_token_id,
        gamma=1,
    )
    response = transport.verify_round(
        VerifyRoundRequest(
            req_id=req_id,
            committed_token_id=bootstrap_token_id,
            draft_token_ids=list(round_state.draft_token_ids),
            draft_q_values=list(round_state.draft_q_values),
        )
    )

    assert round_state.draft_token_ids
    assert response.accepted_len in (0, 1)
    if response.accepted_len == 1:
        assert response.bonus_token_id is not None
    else:
        assert response.rejected_target_logits is not None

    transport.close_session(req_id)


def test_real_external_draft_all_accept_matches_direct_target(
    real_dssd_runtime,
) -> None:
    from vllm.dssd.service.edge_service import DSSDEdgeService

    edge_engine, verifier_engine, transport = _build_real_components(
        real_dssd_runtime,
        verifier_gamma=1,
    )
    edge_service = DSSDEdgeService(
        decode_engine=edge_engine,
        verifier=transport,
        eos_token_id=-1,
        gamma=1,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        ignore_eos=True,
    )
    prompt_token_ids = [1, 2, 3]

    opened = transport.open_session(
        req_id="accept-req",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )
    edge_session = _make_edge_shadow_session(
        req_id="accept-req",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )
    bootstrap_token_id = edge_engine.commit_external_token(
        edge_session,
        opened.bootstrap_token_id,
    )
    round_state = edge_engine.draft(
        edge_session,
        first_token_id=bootstrap_token_id,
        gamma=1,
    )
    response = transport.verify_round(
        VerifyRoundRequest(
            req_id="accept-req",
            committed_token_id=bootstrap_token_id,
            draft_token_ids=list(round_state.draft_token_ids),
            draft_q_values=list(round_state.draft_q_values),
        )
    )

    committed_token_id, committed_count = edge_service._commit_verify_result(
        edge_session,
        response,
    )
    direct_token_ids = _decode_via_direct_verifier_target_only(
        verifier_engine=verifier_engine,
        req_id="accept-direct",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        output_tokens=3,
    )

    assert response.accepted_len == 1
    assert response.bonus_token_id is not None
    assert committed_token_id == direct_token_ids[-1]
    assert committed_count == 2
    assert edge_session.committed_output_ids() == direct_token_ids

    transport.close_session("accept-req")


def test_real_external_draft_reject_resamples_to_target_token(
    monkeypatch: pytest.MonkeyPatch,
    real_dssd_runtime,
) -> None:
    from vllm.dssd.service.edge_service import DSSDEdgeService
    import torch

    edge_engine, verifier_engine, transport = _build_real_components(
        real_dssd_runtime,
        verifier_gamma=1,
    )
    edge_service = DSSDEdgeService(
        decode_engine=edge_engine,
        verifier=transport,
        eos_token_id=-1,
        gamma=1,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        ignore_eos=True,
    )
    prompt_token_ids = [1, 2, 3]

    monkeypatch.setattr(
        torch,
        "multinomial",
        lambda probs, num_samples: torch.argmax(
            probs,
            dim=-1,
            keepdim=True,
        ).to(dtype=torch.int64),
    )

    opened = transport.open_session(
        req_id="reject-req",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )
    edge_session = _make_edge_shadow_session(
        req_id="reject-req",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )
    bootstrap_token_id = edge_engine.commit_external_token(
        edge_session,
        opened.bootstrap_token_id,
    )
    round_state = edge_engine.draft(
        edge_session,
        first_token_id=bootstrap_token_id,
        gamma=1,
    )
    response = transport.verify_round(
        VerifyRoundRequest(
            req_id="reject-req",
            committed_token_id=bootstrap_token_id,
            draft_token_ids=[
                (int(round_state.draft_token_ids[0]) + 1) % edge_engine.vocab_size
            ],
            draft_q_values=[1.0],
        )
    )

    committed_token_id, committed_count = edge_service._commit_verify_result(
        edge_session,
        response,
    )
    direct_token_ids = _decode_via_direct_verifier_target_only(
        verifier_engine=verifier_engine,
        req_id="reject-direct",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        output_tokens=2,
    )

    assert response.accepted_len == 0
    assert response.rejected_target_logits is not None
    assert committed_token_id == direct_token_ids[-1]
    assert committed_count == 1
    assert edge_session.committed_output_ids() == direct_token_ids

    transport.close_session("reject-req")


def test_real_edge_service_generate_matches_direct_target(
    monkeypatch: pytest.MonkeyPatch,
    real_dssd_runtime,
) -> None:
    from vllm.dssd.service.edge_service import DSSDEdgeService
    import torch

    edge_engine, verifier_engine, transport = _build_real_components(
        real_dssd_runtime,
        verifier_gamma=1,
        remote_req_id_factory=lambda req_id: f"{req_id}::verifier",
    )
    edge_service = DSSDEdgeService(
        decode_engine=edge_engine,
        verifier=transport,
        eos_token_id=-1,
        gamma=1,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=3,
        ignore_eos=True,
    )
    prompt_token_ids = [1, 2, 3]

    monkeypatch.setattr(
        torch,
        "multinomial",
        lambda probs, num_samples: torch.argmax(
            probs,
            dim=-1,
            keepdim=True,
        ).to(dtype=torch.int64),
    )

    output_ids = edge_service.generate(
        req_id="generate-req",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )
    direct_token_ids = _decode_via_direct_verifier_target_only(
        verifier_engine=verifier_engine,
        req_id="generate-direct",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        output_tokens=3,
    )

    assert output_ids == direct_token_ids


def test_real_edge_service_generate_matches_direct_target_v1(
    monkeypatch: pytest.MonkeyPatch,
    real_dssd_runtime_v1_edge_v2_verifier,
) -> None:
    from vllm.dssd.service.edge_service import DSSDEdgeService
    import torch

    edge_engine, verifier_engine, transport = _build_real_components_from_runtimes(
        edge_runtime=real_dssd_runtime_v1_edge_v2_verifier.edge_runtime,
        verifier_runtime=real_dssd_runtime_v1_edge_v2_verifier.verifier_runtime,
        edge_model_runner_version="v1",
        verifier_gamma=1,
        remote_req_id_factory=lambda req_id: f"{req_id}::verifier",
    )
    edge_service = DSSDEdgeService(
        decode_engine=edge_engine,
        verifier=transport,
        eos_token_id=-1,
        gamma=1,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=3,
        ignore_eos=True,
    )
    prompt_token_ids = [1, 2, 3]

    monkeypatch.setattr(
        torch,
        "multinomial",
        lambda probs, num_samples: torch.argmax(
            probs,
            dim=-1,
            keepdim=True,
        ).to(dtype=torch.int64),
    )

    output_ids = edge_service.generate(
        req_id="generate-v1-req",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )
    direct_token_ids = _decode_via_direct_verifier_target_only(
        verifier_engine=verifier_engine,
        req_id="generate-v1-direct",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        output_tokens=3,
    )

    assert output_ids == direct_token_ids


def test_real_edge_service_generate_matches_direct_target_split_runtime(
    monkeypatch: pytest.MonkeyPatch,
    real_split_dssd_runtime,
) -> None:
    from vllm.dssd.service.edge_service import DSSDEdgeService
    import torch

    edge_engine, verifier_engine, transport = _build_real_components_from_runtimes(
        edge_runtime=real_split_dssd_runtime.edge_runtime,
        verifier_runtime=real_split_dssd_runtime.verifier_runtime,
        verifier_gamma=1,
    )
    edge_service = DSSDEdgeService(
        decode_engine=edge_engine,
        verifier=transport,
        eos_token_id=-1,
        gamma=1,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=3,
        ignore_eos=True,
    )
    prompt_token_ids = [1, 2, 3]

    monkeypatch.setattr(
        torch,
        "multinomial",
        lambda probs, num_samples: torch.argmax(
            probs,
            dim=-1,
            keepdim=True,
        ).to(dtype=torch.int64),
    )

    output_ids = edge_service.generate(
        req_id="split-generate-req",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )
    direct_token_ids = _decode_via_direct_verifier_target_only(
        verifier_engine=verifier_engine,
        req_id="split-generate-direct",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        output_tokens=3,
    )

    assert output_ids == direct_token_ids
