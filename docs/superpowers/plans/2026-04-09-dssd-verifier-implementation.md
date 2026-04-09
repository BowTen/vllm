# DSSD Verifier Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `vllm/dssd/verifier/` 下实现第一版 DSSD verifier 正式内核，并基于真实小模型跑通单请求 `open_session -> verify_round -> close_session` 链路。

**Architecture:** 代码分成 `engine -> scheduler / state_bridge / sampler / ops / types` 五层，DSSD 状态机集中在 `vllm/dssd/verifier/`。`vllm/v1` 只补最小内部执行接口，主要落在 `GPUModelRunner`，让 verifier 复用真实 `InputBatch`、KV cache 和采样链路，但不把 DSSD 业务语义塞进通用主流程。

**Tech Stack:** Python, pytest, PyTorch, Triton, vLLM v1 `GPUWorker/GPUModelRunner/KVCacheManager/SchedulerOutput`, Hugging Face tiny model

---

## File Structure

**Create:**

- `vllm/dssd/__init__.py`
  - DSSD 命名空间包入口
- `vllm/dssd/verifier/__init__.py`
  - 导出 `VerifierDecodeEngine` 和关键类型
- `vllm/dssd/verifier/types.py`
  - `VerifierRoundRequest`
  - `VerifierRoundResult`
  - `VerifierOpenSessionResult`
  - `VerifierSession`
  - `VerifierRoundState`
- `vllm/dssd/verifier/scheduler.py`
  - `VerifierSchedulerAdapter`
- `vllm/dssd/verifier/state_bridge.py`
  - `VerifierStateBridge`
- `vllm/dssd/verifier/ops.py`
  - accept/reject 的 Triton/CUDA helper
- `vllm/dssd/verifier/sampler.py`
  - `DSSDVerifierSampler`
- `vllm/dssd/verifier/engine.py`
  - `VerifierDecodeEngine`
- `tests/dssd/verifier/conftest.py`
  - `dummy_model_runner`
  - `real_worker`
  - KV cache / worker fixture helper
- `tests/dssd/verifier/test_types.py`
  - 共享类型与请求校验测试
- `tests/dssd/verifier/test_model_runner_hooks.py`
  - `GPUModelRunner` verifier helper 测试
- `tests/dssd/verifier/test_scheduler_state.py`
  - scheduler/state bridge 测试
- `tests/dssd/verifier/test_sampler.py`
  - GPU-first accept/reject 与 bonus/reject-logits 测试
- `tests/dssd/verifier/test_engine_smoke.py`
  - 真实模型 smoke test

**Modify:**

- `vllm/v1/worker/gpu/model_runner.py`
  - 新增 verifier 需要的最小内部 helper：
    - `take_execute_model_state()`
    - `sample_without_postprocess(...)`
    - `commit_input_token(...)`

### Task 1: 搭 verifier 包骨架和共享类型

**Files:**
- Create: `vllm/dssd/__init__.py`
- Create: `vllm/dssd/verifier/__init__.py`
- Create: `vllm/dssd/verifier/types.py`
- Test: `tests/dssd/verifier/test_types.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dssd/verifier/test_types.py
import pytest

from vllm.sampling_params import SamplingParams

from vllm.dssd.verifier.types import (
    VerifierOpenSessionResult,
    VerifierRoundRequest,
    VerifierRoundResult,
    VerifierSession,
)


def test_round_request_validate_checks_lengths_and_gamma() -> None:
    bad_lengths = VerifierRoundRequest(
        req_id="req-1",
        committed_token_id=9,
        draft_token_ids=[11, 12],
        draft_q_values=[0.2],
    )
    with pytest.raises(ValueError, match="长度不一致"):
        bad_lengths.validate(gamma=2)

    bad_gamma = VerifierRoundRequest(
        req_id="req-1",
        committed_token_id=9,
        draft_token_ids=[11, 12, 13],
        draft_q_values=[0.2, 0.3, 0.4],
    )
    with pytest.raises(ValueError, match="超过固定 gamma"):
        bad_gamma.validate(gamma=2)


def test_session_output_len_tracks_confirmed_suffix_only() -> None:
    session = VerifierSession(
        req_id="req-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(),
        block_ids=([0],),
        prompt_len=3,
        token_ids=[1, 2, 3, 4, 5],
        total_len=5,
    )
    assert session.output_len == 2


def test_round_result_helpers_match_acceptance_shape() -> None:
    result = VerifierRoundResult(req_id="req-1", accepted_len=2, bonus_token_id=17)
    assert result.is_all_accepted(draft_len=2)
    assert not result.is_rejected(draft_len=2)

    bootstrap = VerifierOpenSessionResult(req_id="req-1", bootstrap_token_id=7)
    assert bootstrap.bootstrap_token_id == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/dssd/verifier/test_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vllm.dssd'`

- [ ] **Step 3: Write minimal implementation**

```python
# vllm/dssd/__init__.py
"""DSSD experimental components."""
```

```python
# vllm/dssd/verifier/__init__.py
from .types import (
    VerifierOpenSessionResult,
    VerifierRoundRequest,
    VerifierRoundResult,
    VerifierRoundState,
    VerifierSession,
)

__all__ = [
    "VerifierOpenSessionResult",
    "VerifierRoundRequest",
    "VerifierRoundResult",
    "VerifierRoundState",
    "VerifierSession",
]
```

```python
# vllm/dssd/verifier/types.py
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams


@dataclass
class VerifierRoundRequest:
    req_id: str
    committed_token_id: int
    draft_token_ids: list[int]
    draft_q_values: list[float]

    def validate(self, gamma: int) -> None:
        if len(self.draft_token_ids) != len(self.draft_q_values):
            raise ValueError("draft_token_ids 和 draft_q_values 长度不一致")
        if len(self.draft_token_ids) > gamma:
            raise ValueError("draft 长度超过固定 gamma")


@dataclass
class VerifierRoundResult:
    req_id: str
    accepted_len: int
    bonus_token_id: int | None = None
    rejected_step: int | None = None
    rejected_target_logits: torch.Tensor | None = None

    def is_all_accepted(self, draft_len: int) -> bool:
        return self.accepted_len == draft_len

    def is_rejected(self, draft_len: int) -> bool:
        return self.accepted_len < draft_len


@dataclass
class VerifierOpenSessionResult:
    req_id: str
    bootstrap_token_id: int


@dataclass
class VerifierSession:
    req_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    block_ids: tuple[list[int], ...]
    prompt_len: int
    num_computed_tokens: int = 0
    total_len: int = 0
    token_ids: list[int] = field(default_factory=list)
    lora_request: LoRARequest | None = None

    @property
    def output_len(self) -> int:
        return max(self.total_len - self.prompt_len, 0)


@dataclass
class VerifierRoundState:
    committed_token_id: int | None = None
    draft_token_ids: list[int] = field(default_factory=list)
    draft_q_values: list[float] = field(default_factory=list)
    last_result: VerifierRoundResult | None = None

    def reset(self) -> None:
        self.committed_token_id = None
        self.draft_token_ids.clear()
        self.draft_q_values.clear()
        self.last_result = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/dssd/verifier/test_types.py -v`
Expected: PASS with `3 passed`

- [ ] **Step 5: Commit**

```bash
git add vllm/dssd/__init__.py vllm/dssd/verifier/__init__.py vllm/dssd/verifier/types.py tests/dssd/verifier/test_types.py
git commit -m "feat: scaffold dssd verifier types"
```

### Task 2: 给 `GPUModelRunner` 补 verifier 内部 helper

**Files:**
- Create: `tests/dssd/verifier/conftest.py`
- Test: `tests/dssd/verifier/test_model_runner_hooks.py`
- Modify: `vllm/v1/worker/gpu/model_runner.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dssd/verifier/conftest.py
import pytest

from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.attention import Attention
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

BLOCK_SIZE = 16
NUM_BLOCKS = 10


def _build_unit_vllm_config() -> VllmConfig:
    model_config = ModelConfig(model="facebook/opt-125m", dtype="float16", seed=42)
    scheduler_config = SchedulerConfig(
        max_num_seqs=4,
        max_num_batched_tokens=128,
        max_model_len=128,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    return VllmConfig(
        model_config=model_config,
        cache_config=CacheConfig(block_size=BLOCK_SIZE, gpu_memory_utilization=0.5),
        scheduler_config=scheduler_config,
        parallel_config=ParallelConfig(),
    )


def _initialize_dummy_kv_cache(runner: GPUModelRunner) -> None:
    attn_spec = FullAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=runner.model_config.get_num_kv_heads(runner.parallel_config),
        head_size=runner.model_config.get_head_size(),
        dtype=runner.kv_cache_dtype,
    )
    tensor_size = attn_spec.page_size_bytes * NUM_BLOCKS
    kv_cache_config = KVCacheConfig(
        num_blocks=NUM_BLOCKS,
        kv_cache_tensors=[KVCacheTensor(size=tensor_size, shared_by=["layer.0"])],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=attn_spec)
        ],
    )
    runner.kv_cache_config = kv_cache_config
    runner.input_batch = InputBatch(
        max_num_reqs=runner.max_num_reqs,
        max_model_len=runner.max_model_len,
        max_num_batched_tokens=runner.max_num_tokens,
        device=runner.device,
        pin_memory=runner.pin_memory,
        vocab_size=runner.model_config.get_vocab_size(),
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
    )
    runner.initialize_attn_backend(kv_cache_config)


@pytest.fixture
def dummy_model_runner() -> GPUModelRunner:
    vllm_config = _build_unit_vllm_config()
    with set_current_vllm_config(vllm_config):
        model_config = vllm_config.model_config
        num_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)
        head_size = model_config.get_head_size()
        vllm_config.compilation_config.static_forward_context["layer.0"] = Attention(
            num_heads, head_size, 0.1
        )
        runner = GPUModelRunner(vllm_config, current_platform.device_type)
        _initialize_dummy_kv_cache(runner)
        yield runner
```

```python
# tests/dssd/verifier/test_model_runner_hooks.py
from unittest import mock

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.worker.gpu.sample.output import SamplerOutput


def _schedule_prompt(req_id: str) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(
                req_id=req_id,
                prompt_token_ids=[1, 2, 3],
                mm_features=[],
                sampling_params=SamplingParams(),
                pooling_params=None,
                block_ids=([0],),
                num_computed_tokens=0,
                lora_request=None,
            )
        ],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={req_id: 3},
        total_num_scheduled_tokens=3,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def test_sample_without_postprocess_does_not_advance_lengths(dummy_model_runner, dist_init):
    runner = dummy_model_runner
    runner._update_states(_schedule_prompt("req-0"))
    req_idx = runner.req_states.req_id_to_index["req-0"]
    runner.req_states.total_len.gpu[req_idx] = 3

    fake_output = SamplerOutput(
        sampled_token_ids=torch.tensor([[7]], device=runner.device),
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=torch.tensor([1], device=runner.device),
    )
    fake_hidden = torch.zeros((1, 8), device=runner.device)
    fake_input_batch = mock.MagicMock()

    with mock.patch.object(
        runner,
        "sample",
        return_value=(fake_output, torch.tensor([1], device=runner.device), torch.tensor([0], device=runner.device)),
    ):
        result = runner.sample_without_postprocess(fake_hidden, fake_input_batch, grammar_output=None)

    assert result.sampled_token_ids.tolist() == [[7]]
    assert int(runner.req_states.total_len.gpu[req_idx].item()) == 3


def test_commit_input_token_updates_req_state(dummy_model_runner, dist_init):
    runner = dummy_model_runner
    runner._update_states(_schedule_prompt("req-0"))
    req_idx = runner.req_states.req_id_to_index["req-0"]
    runner.req_states.total_len.gpu[req_idx] = 3

    runner.commit_input_token(req_idx, 9)

    assert int(runner.req_states.last_sampled_tokens[req_idx, 0].item()) == 9
    assert int(runner.req_states.all_token_ids.gpu[req_idx, 3].item()) == 9
    assert int(runner.req_states.total_len.gpu[req_idx].item()) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/dssd/verifier/test_model_runner_hooks.py -v`
Expected: FAIL with `AttributeError: 'GPUModelRunner' object has no attribute 'sample_without_postprocess'`

- [ ] **Step 3: Write minimal implementation**

```python
# vllm/v1/worker/gpu/model_runner.py
    def take_execute_model_state(self) -> ExecuteModelState:
        if self.execute_model_state is None:
            raise RuntimeError("execute_model_state is empty")
        state = self.execute_model_state
        self.execute_model_state = None
        return state

    def sample_without_postprocess(
        self,
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        grammar_output: GrammarOutput | None,
    ) -> SamplerOutput:
        sampler_output, _, _ = self.sample(hidden_states, input_batch, grammar_output)
        return sampler_output

    def commit_input_token(self, req_idx: int, token_id: int) -> None:
        total_len = int(self.req_states.total_len.gpu[req_idx].item())
        self.req_states.last_sampled_tokens[req_idx, 0] = token_id
        self.req_states.all_token_ids.gpu[req_idx, total_len] = token_id
        self.req_states.total_len.gpu[req_idx] = total_len + 1
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/dssd/verifier/test_model_runner_hooks.py -v`
Expected: PASS with `2 passed`

- [ ] **Step 5: Commit**

```bash
git add tests/dssd/verifier/conftest.py tests/dssd/verifier/test_model_runner_hooks.py vllm/v1/worker/gpu/model_runner.py
git commit -m "feat: add verifier hooks to gpu model runner"
```

### Task 3: 实现 scheduler 和 state bridge

**Files:**
- Create: `vllm/dssd/verifier/scheduler.py`
- Create: `vllm/dssd/verifier/state_bridge.py`
- Test: `tests/dssd/verifier/test_scheduler_state.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dssd/verifier/test_scheduler_state.py
from unittest import mock

from vllm.sampling_params import SamplingParams
from vllm.dssd.verifier.scheduler import VerifierSchedulerAdapter
from vllm.dssd.verifier.state_bridge import VerifierStateBridge
from vllm.dssd.verifier.types import VerifierRoundResult, VerifierSession


def _build_session() -> VerifierSession:
    return VerifierSession(
        req_id="req-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(),
        block_ids=([0],),
        prompt_len=3,
        token_ids=[1, 2, 3],
        num_computed_tokens=3,
        total_len=3,
    )


def test_allocate_blocks_uses_kv_cache_manager() -> None:
    kv_cache_manager = mock.MagicMock()
    blocks = mock.MagicMock()
    blocks.get_block_ids.return_value = ([7],)
    kv_cache_manager.allocate_slots.return_value = blocks
    adapter = VerifierSchedulerAdapter(kv_cache_manager=kv_cache_manager)

    block_ids = adapter.allocate_blocks(
        req_id="req-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(),
    )

    assert block_ids == ([7],)
    kv_cache_manager.allocate_slots.assert_called_once()


def test_finish_prefill_without_commit_keeps_prompt_only() -> None:
    session = _build_session()
    bridge = VerifierStateBridge()

    bridge.finish_prefill_without_commit(session)

    assert session.num_computed_tokens == session.prompt_len
    assert session.total_len == session.prompt_len
    assert session.token_ids == [1, 2, 3]


def test_set_round_result_only_commits_accepted_prefix() -> None:
    session = _build_session()
    bridge = VerifierStateBridge()
    bridge.set_round_q_values(session, [0.2, 0.3, 0.4])
    bridge._round_state(session).draft_token_ids = [11, 12, 13]

    bridge.set_round_result(
        session,
        VerifierRoundResult(req_id="req-1", accepted_len=2, rejected_step=2),
    )

    assert session.token_ids == [1, 2, 3, 11, 12]
    assert session.total_len == 5
    assert session.num_computed_tokens == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/dssd/verifier/test_scheduler_state.py -v`
Expected: FAIL with `ModuleNotFoundError` for `vllm.dssd.verifier.scheduler` or `state_bridge`

- [ ] **Step 3: Write minimal implementation**

```python
# vllm/dssd/verifier/scheduler.py
from __future__ import annotations

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.request import Request

from .types import VerifierRoundRequest, VerifierSession


class VerifierSchedulerAdapter:
    def __init__(self, kv_cache_manager: KVCacheManager | None = None) -> None:
        self.kv_cache_manager = kv_cache_manager

    def allocate_blocks(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        lora_request: LoRARequest | None = None,
    ) -> tuple[list[int], ...]:
        if self.kv_cache_manager is None:
            return ([],)
        request = Request(
            request_id=req_id,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=sampling_params,
            pooling_params=None,
            lora_request=lora_request,
        )
        blocks = self.kv_cache_manager.allocate_slots(
            request=request,
            num_new_tokens=request.num_tokens - request.num_computed_tokens,
        )
        if blocks is None:
            raise RuntimeError("prefill 无法为 verifier 申请新的 KV blocks")
        return blocks.get_block_ids(allow_none=True)

    def build_open_session_step(self, session: VerifierSession) -> SchedulerOutput:
        prompt_len = len(session.prompt_token_ids)
        return SchedulerOutput(
            scheduled_new_reqs=[
                NewRequestData(
                    req_id=session.req_id,
                    prompt_token_ids=session.prompt_token_ids,
                    prefill_token_ids=session.prompt_token_ids,
                    mm_features=[],
                    sampling_params=session.sampling_params,
                    pooling_params=None,
                    block_ids=session.block_ids,
                    num_computed_tokens=0,
                    lora_request=session.lora_request,
                )
            ],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={session.req_id: prompt_len},
            total_num_scheduled_tokens=prompt_len,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            new_block_ids_to_zero=None,
        )

    def build_verify_step(
        self,
        session: VerifierSession,
        request: VerifierRoundRequest,
    ) -> SchedulerOutput:
        query_len = 1 + len(request.draft_token_ids)
        return SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData(
                req_ids=[session.req_id],
                resumed_req_ids=set(),
                new_token_ids=[[]],
                all_token_ids={},
                new_block_ids=[None],
                num_computed_tokens=[session.num_computed_tokens],
                num_output_tokens=[session.output_len],
            ),
            num_scheduled_tokens={session.req_id: query_len},
            total_num_scheduled_tokens=query_len,
            scheduled_spec_decode_tokens={session.req_id: list(request.draft_token_ids)},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            new_block_ids_to_zero=None,
        )

    def build_close_step(self, req_id: str) -> SchedulerOutput:
        return SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[],
            finished_req_ids={req_id},
            free_encoder_mm_hashes=[],
        )

    def free_blocks(self, session: VerifierSession) -> None:
        if self.kv_cache_manager is None:
            return
        request = Request(
            request_id=session.req_id,
            prompt_token_ids=list(session.prompt_token_ids),
            sampling_params=session.sampling_params,
            pooling_params=None,
            lora_request=session.lora_request,
        )
        finalized = session.token_ids[session.prompt_len :]
        if finalized:
            request.append_output_token_ids(finalized)
        request.num_computed_tokens = session.num_computed_tokens
        self.kv_cache_manager.free(request)
```

```python
# vllm/dssd/verifier/state_bridge.py
from __future__ import annotations

from collections import defaultdict

import torch

from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from .types import VerifierRoundRequest, VerifierRoundResult, VerifierRoundState, VerifierSession


class VerifierStateBridge:
    def __init__(self) -> None:
        self._round_states: dict[str, VerifierRoundState] = defaultdict(VerifierRoundState)

    def finish_prefill_without_commit(self, session: VerifierSession) -> None:
        session.num_computed_tokens = session.prompt_len
        session.total_len = session.prompt_len
        self.clear_round_state(session)

    def prepare_round(
        self,
        session: VerifierSession,
        request: VerifierRoundRequest,
        model_runner: GPUModelRunner,
        gamma: int,
    ) -> None:
        request.validate(gamma=gamma)
        req_idx = model_runner.req_states.req_id_to_index[session.req_id]
        model_runner.req_states.last_sampled_tokens[req_idx, 0] = request.committed_token_id
        draft_tokens = model_runner.req_states.draft_tokens[req_idx]
        draft_tokens.zero_()
        if request.draft_token_ids:
            draft_tokens[: len(request.draft_token_ids)] = torch.tensor(
                request.draft_token_ids, dtype=torch.int64, device=draft_tokens.device
            )
        round_state = self._round_state(session)
        round_state.committed_token_id = request.committed_token_id
        round_state.draft_token_ids = list(request.draft_token_ids)
        round_state.draft_q_values = list(request.draft_q_values)

    def set_round_q_values(self, session: VerifierSession, q_values: list[float]) -> None:
        self._round_state(session).draft_q_values = list(q_values)

    def commit_committed_token_before_postprocess(
        self,
        session: VerifierSession,
        committed_token_id: int,
        model_runner: GPUModelRunner,
    ) -> None:
        req_idx = model_runner.req_states.req_id_to_index[session.req_id]
        model_runner.commit_input_token(req_idx, committed_token_id)
        session.token_ids.append(committed_token_id)
        session.total_len += 1

    def set_round_result(self, session: VerifierSession, result: VerifierRoundResult) -> None:
        round_state = self._round_state(session)
        round_state.last_result = result
        session.token_ids.extend(round_state.draft_token_ids[: result.accepted_len])
        session.num_computed_tokens += 1 + result.accepted_len
        session.total_len += result.accepted_len

    def clear_round_state(self, session: VerifierSession) -> None:
        self._round_state(session).reset()

    def remove_round_state(self, session: VerifierSession) -> None:
        self._round_states.pop(session.req_id, None)

    def _round_state(self, session: VerifierSession) -> VerifierRoundState:
        return self._round_states[session.req_id]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/dssd/verifier/test_scheduler_state.py -v`
Expected: PASS with `3 passed`

- [ ] **Step 5: Commit**

```bash
git add vllm/dssd/verifier/scheduler.py vllm/dssd/verifier/state_bridge.py tests/dssd/verifier/test_scheduler_state.py
git commit -m "feat: add dssd verifier scheduler and state bridge"
```

### Task 4: 实现 GPU-first sampler 和 Triton helper

**Files:**
- Create: `vllm/dssd/verifier/ops.py`
- Create: `vllm/dssd/verifier/sampler.py`
- Test: `tests/dssd/verifier/test_sampler.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dssd/verifier/test_sampler.py
from pathlib import Path

import pytest
import torch

from vllm.dssd.verifier.sampler import DSSDVerifierSampler
from vllm.dssd.verifier.types import VerifierRoundRequest


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_reject_path_returns_cuda_logits_row(dummy_model_runner) -> None:
    runner = dummy_model_runner
    sampler = DSSDVerifierSampler(
        sampler=runner.sampler,
        num_speculative_steps=2,
    )

    logits = torch.tensor(
        [[5.0, 1.0, 0.0], [0.0, 5.0, 1.0], [1.0, 1.0, 5.0]],
        device=runner.device,
    )
    input_batch = type(
        "Batch",
        (),
        {
            "expanded_idx_mapping": torch.tensor([0, 0, 0], device=runner.device),
            "idx_mapping_np": torch.tensor([0], device="cpu").numpy(),
            "positions": torch.tensor([3, 4, 5], device=runner.device),
            "input_ids": torch.tensor([7, 8, 9], device=runner.device),
            "expanded_local_pos": torch.tensor([0, 1, 2], device=runner.device),
            "logits_indices": torch.tensor([0, 1, 2], device=runner.device),
            "seq_lens": torch.tensor([3], device=runner.device, dtype=torch.int32),
            "num_reqs": 1,
        },
    )()
    request = VerifierRoundRequest(
        req_id="req-1",
        committed_token_id=7,
        draft_token_ids=[0, 0],
        draft_q_values=[1e-6, 1e-6],
    )

    sampler_output, result = sampler(logits, input_batch, request)

    assert result.rejected_step == 0
    assert result.rejected_target_logits is not None
    assert result.rejected_target_logits.is_cuda
    assert sampler_output.sampled_token_ids.shape == (1, 3)


def test_sampler_source_keeps_accept_reject_off_cpu() -> None:
    source = Path("vllm/dssd/verifier/sampler.py").read_text(encoding="utf-8")
    assert ".item()" not in source
    assert "torch.rand(" not in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/dssd/verifier/test_sampler.py -v`
Expected: FAIL with `ModuleNotFoundError` for `vllm.dssd.verifier.sampler`

- [ ] **Step 3: Write minimal implementation**

```python
# vllm/dssd/verifier/ops.py
from __future__ import annotations

import torch


def find_accepted_len(
    processed_logits: torch.Tensor,
    draft_token_ids: torch.Tensor,
    draft_q_values: torch.Tensor,
    _seeds: torch.Tensor,
    _positions: torch.Tensor,
) -> int:
    log_probs = processed_logits.log_softmax(dim=-1)
    p_values = log_probs.gather(1, draft_token_ids.view(-1, 1)).squeeze(1).exp()
    q_values = torch.clamp(draft_q_values, min=1e-20)
    accept_probs = torch.minimum(torch.ones_like(p_values), p_values / q_values)
    noise = torch.rand_like(accept_probs)
    accepted = noise < accept_probs
    rejected = torch.nonzero(~accepted, as_tuple=False)
    if rejected.numel() == 0:
        return int(accepted.shape[0])
    return int(rejected[0, 0])
```

```python
# vllm/dssd/verifier/sampler.py
from __future__ import annotations

import torch

from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler

from .ops import find_accepted_len
from .types import VerifierRoundRequest, VerifierRoundResult


class DSSDVerifierSampler:
    def __init__(self, sampler: Sampler, num_speculative_steps: int) -> None:
        self.sampler = sampler
        self.num_speculative_steps = num_speculative_steps

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        request: VerifierRoundRequest,
    ) -> tuple[SamplerOutput, VerifierRoundResult]:
        pos = input_batch.positions[input_batch.logits_indices]
        input_ids = input_batch.input_ids[input_batch.logits_indices]
        processed_logits = self.sampler.apply_sampling_params(
            logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping_np,
            pos,
            input_ids,
            input_batch.expanded_local_pos,
        )
        accepted_len = find_accepted_len(
            processed_logits[: len(request.draft_token_ids)],
            torch.tensor(request.draft_token_ids, device=processed_logits.device, dtype=torch.int64),
            torch.tensor(request.draft_q_values, device=processed_logits.device, dtype=processed_logits.dtype),
            self.sampler.sampling_states.seeds.gpu,
            pos[: len(request.draft_token_ids)],
        )
        sampled = torch.full(
            (1, self.num_speculative_steps + 1),
            fill_value=-1,
            dtype=torch.int64,
            device=processed_logits.device,
        )
        if accepted_len > 0:
            sampled[0, :accepted_len] = torch.tensor(
                request.draft_token_ids[:accepted_len],
                dtype=torch.int64,
                device=processed_logits.device,
            )
        sampler_output = SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=torch.tensor([accepted_len], dtype=torch.int32, device=processed_logits.device),
        )
        if accepted_len == len(request.draft_token_ids):
            bonus_idx = len(request.draft_token_ids)
            bonus_token = gumbel_sample(
                processed_logits[bonus_idx : bonus_idx + 1],
                input_batch.expanded_idx_mapping[bonus_idx : bonus_idx + 1],
                self.sampler.sampling_states.temperature.gpu,
                self.sampler.sampling_states.seeds.gpu,
                pos[bonus_idx : bonus_idx + 1],
                apply_temperature=False,
            )
            return sampler_output, VerifierRoundResult(
                req_id=request.req_id,
                accepted_len=accepted_len,
                bonus_token_id=int(bonus_token[0]),
            )
        return sampler_output, VerifierRoundResult(
            req_id=request.req_id,
            accepted_len=accepted_len,
            rejected_step=accepted_len,
            rejected_target_logits=processed_logits[accepted_len].detach().clone(),
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/dssd/verifier/test_sampler.py -v`
Expected: PASS with `2 passed` on CUDA machines, or `1 passed, 1 skipped` if CUDA is unavailable

- [ ] **Step 5: Commit**

```bash
git add vllm/dssd/verifier/ops.py vllm/dssd/verifier/sampler.py tests/dssd/verifier/test_sampler.py
git commit -m "feat: add dssd verifier sampler"
```

### Task 5: 实现 engine，并用真实小模型跑 smoke test

**Files:**
- Modify: `tests/dssd/verifier/conftest.py`
- Create: `vllm/dssd/verifier/engine.py`
- Modify: `vllm/dssd/verifier/__init__.py`
- Test: `tests/dssd/verifier/test_engine_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dssd/verifier/conftest.py
import tempfile

from vllm.engine.arg_utils import EngineArgs
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import generate_scheduler_kv_cache_config, get_kv_cache_configs
from vllm.v1.worker.gpu_worker import Worker


@pytest.fixture
def real_worker():
    engine_args = EngineArgs(
        model="hmellor/tiny-random-LlamaForCausalLM",
        enforce_eager=True,
        gpu_memory_utilization=0.4,
        max_model_len=64,
    )
    vllm_config = engine_args.create_engine_config()
    with tempfile.NamedTemporaryFile() as f, set_current_vllm_config(vllm_config):
        worker = Worker(
            vllm_config=vllm_config,
            local_rank=0,
            rank=0,
            distributed_init_method=f"file://{f.name}",
            is_driver_worker=True,
        )
        worker.init_device()
        worker.load_model()
        available_memory = [worker.determine_available_memory()]
        kv_cache_configs = get_kv_cache_configs(
            vllm_config,
            [worker.get_kv_cache_spec()],
            available_memory,
        )
        scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)
        worker.initialize_from_config(kv_cache_configs[0])
        yield worker, vllm_config, KVCacheManager(
            kv_cache_config=scheduler_kv_cache_config,
            max_model_len=vllm_config.model_config.max_model_len,
            hash_block_size=vllm_config.cache_config.block_size,
        )
```

```python
# tests/dssd/verifier/test_engine_smoke.py
import pytest

from vllm.sampling_params import SamplingParams
from vllm.dssd.verifier.engine import VerifierDecodeEngine
from vllm.dssd.verifier.scheduler import VerifierSchedulerAdapter
from vllm.dssd.verifier.state_bridge import VerifierStateBridge
from vllm.dssd.verifier.sampler import DSSDVerifierSampler
from vllm.dssd.verifier.types import VerifierRoundRequest

from tests.utils import create_new_process_for_each_test


@pytest.mark.skipif(not pytest.importorskip("torch").cuda.is_available(), reason="requires cuda")
@create_new_process_for_each_test()
def test_open_verify_close_smoke(real_worker) -> None:
    worker, vllm_config, kv_cache_manager = real_worker
    sampler = DSSDVerifierSampler(
        sampler=worker.model_runner.sampler,
        num_speculative_steps=worker.model_runner.num_speculative_steps,
    )
    engine = VerifierDecodeEngine(
        vllm_config=vllm_config,
        worker=worker,
        scheduler=VerifierSchedulerAdapter(kv_cache_manager=kv_cache_manager),
        state_bridge=VerifierStateBridge(),
        verifier_sampler=sampler,
    )

    opened = engine.open_session(
        req_id="req-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(temperature=0.0),
    )
    assert opened.bootstrap_token_id >= 0
    session = engine.sessions["req-1"]
    assert session.total_len == session.prompt_len

    result = engine.verify_round(
        session,
        VerifierRoundRequest(
            req_id="req-1",
            committed_token_id=opened.bootstrap_token_id,
            draft_token_ids=[],
            draft_q_values=[],
        ),
    )
    assert result.accepted_len == 0

    engine.close_session(session)
    assert "req-1" not in engine.sessions
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/dssd/verifier/test_engine_smoke.py -v -s`
Expected: FAIL with `ModuleNotFoundError` for `vllm.dssd.verifier.engine`

- [ ] **Step 3: Write minimal implementation**

```python
# vllm/dssd/verifier/engine.py
from __future__ import annotations

from typing import cast

import torch

from vllm.config import VllmConfig
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.worker.gpu_worker import Worker as GPUWorker

from .sampler import DSSDVerifierSampler
from .scheduler import VerifierSchedulerAdapter
from .state_bridge import VerifierStateBridge
from .types import VerifierOpenSessionResult, VerifierRoundRequest, VerifierRoundResult, VerifierSession


class VerifierDecodeEngine:
    def __init__(
        self,
        vllm_config: VllmConfig,
        worker: GPUWorker,
        scheduler: VerifierSchedulerAdapter,
        state_bridge: VerifierStateBridge,
        verifier_sampler: DSSDVerifierSampler,
    ) -> None:
        self.vllm_config = vllm_config
        self.worker = worker
        self.scheduler = scheduler
        self.state_bridge = state_bridge
        self.verifier_sampler = verifier_sampler
        self.gamma = self.worker.model_runner.num_speculative_steps
        self.sessions: dict[str, VerifierSession] = {}

    def _execute(self, scheduler_output):
        output = self.worker.execute_model(scheduler_output)
        if isinstance(output, AsyncModelRunnerOutput):
            return output.get_output()
        return output

    def open_session(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        lora_request: LoRARequest | None = None,
    ) -> VerifierOpenSessionResult:
        session = VerifierSession(
            req_id=req_id,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=sampling_params,
            block_ids=self.scheduler.allocate_blocks(
                req_id=req_id,
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
                lora_request=lora_request,
            ),
            prompt_len=len(prompt_token_ids),
            token_ids=list(prompt_token_ids),
            lora_request=lora_request,
        )
        self._execute(self.scheduler.build_open_session_step(session))
        state = self.worker.model_runner.take_execute_model_state()
        hidden_states = cast(torch.Tensor, state.hidden_states)
        sampler_output = self.worker.model_runner.sample_without_postprocess(
            hidden_states,
            state.input_batch,
            grammar_output=None,
        )
        bootstrap_token_id = int(sampler_output.sampled_token_ids[0, 0].item())
        self.worker.model_runner.postprocess(
            state.input_batch,
            torch.full(
                (state.input_batch.num_reqs, self.gamma + 1),
                fill_value=-1,
                dtype=torch.int64,
                device=state.input_batch.seq_lens.device,
            ),
            torch.zeros(state.input_batch.num_reqs, dtype=torch.int32, device=state.input_batch.seq_lens.device),
            torch.zeros(state.input_batch.num_reqs, dtype=torch.int32, device=state.input_batch.seq_lens.device),
        )
        self.state_bridge.finish_prefill_without_commit(session)
        self.sessions[req_id] = session
        return VerifierOpenSessionResult(req_id=req_id, bootstrap_token_id=bootstrap_token_id)

    def verify_round(self, session: VerifierSession, request: VerifierRoundRequest) -> VerifierRoundResult:
        self.state_bridge.prepare_round(session, request, self.worker.model_runner, gamma=self.gamma)
        self._execute(self.scheduler.build_verify_step(session, request))
        state = self.worker.model_runner.take_execute_model_state()
        hidden_states = cast(torch.Tensor, state.hidden_states)
        sample_hidden_states = hidden_states[state.input_batch.logits_indices]
        logits = self.worker.model_runner.model.compute_logits(sample_hidden_states)
        sampler_output, result = self.verifier_sampler(logits, state.input_batch, request)
        self.state_bridge.commit_committed_token_before_postprocess(
            session, request.committed_token_id, self.worker.model_runner
        )
        self.worker.model_runner.postprocess(
            state.input_batch,
            sampler_output.sampled_token_ids.to(device=state.input_batch.seq_lens.device, dtype=torch.int64),
            torch.tensor([result.accepted_len], dtype=torch.int32, device=state.input_batch.seq_lens.device),
            torch.tensor([len(request.draft_token_ids) - result.accepted_len], dtype=torch.int32, device=state.input_batch.seq_lens.device),
        )
        self.state_bridge.set_round_result(session, result)
        return result

    def close_session(self, session: VerifierSession) -> None:
        self.scheduler.free_blocks(session)
        self._execute(self.scheduler.build_close_step(session.req_id))
        self.state_bridge.remove_round_state(session)
        self.sessions.pop(session.req_id, None)
```

```python
# vllm/dssd/verifier/__init__.py
from .engine import VerifierDecodeEngine
from .types import (
    VerifierOpenSessionResult,
    VerifierRoundRequest,
    VerifierRoundResult,
    VerifierRoundState,
    VerifierSession,
)

__all__ = [
    "VerifierDecodeEngine",
    "VerifierOpenSessionResult",
    "VerifierRoundRequest",
    "VerifierRoundResult",
    "VerifierRoundState",
    "VerifierSession",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/dssd/verifier/test_engine_smoke.py -v -s`
Expected: PASS with `1 passed` on a CUDA machine with model download access

- [ ] **Step 5: Commit**

```bash
git add tests/dssd/verifier/conftest.py tests/dssd/verifier/test_engine_smoke.py vllm/dssd/verifier/engine.py vllm/dssd/verifier/__init__.py
git commit -m "feat: add dssd verifier engine smoke path"
```

### Task 6: 统一跑 verifier 目标测试并清理接口导出

**Files:**
- Modify: `vllm/dssd/verifier/__init__.py`
- Test: `tests/dssd/verifier/test_types.py`
- Test: `tests/dssd/verifier/test_model_runner_hooks.py`
- Test: `tests/dssd/verifier/test_scheduler_state.py`
- Test: `tests/dssd/verifier/test_sampler.py`
- Test: `tests/dssd/verifier/test_engine_smoke.py`

- [ ] **Step 1: Write the final import surface check**

```python
# tests/dssd/verifier/test_types.py
from vllm.dssd.verifier import (
    DSSDVerifierSampler,
    VerifierDecodeEngine,
    VerifierOpenSessionResult,
    VerifierRoundRequest,
    VerifierRoundResult,
    VerifierRoundState,
    VerifierSchedulerAdapter,
    VerifierSession,
    VerifierStateBridge,
)


def test_package_exports_are_stable() -> None:
    assert DSSDVerifierSampler is not None
    assert VerifierDecodeEngine is not None
    assert VerifierOpenSessionResult is not None
    assert VerifierRoundRequest is not None
    assert VerifierRoundResult is not None
    assert VerifierRoundState is not None
    assert VerifierSchedulerAdapter is not None
    assert VerifierSession is not None
    assert VerifierStateBridge is not None
```

- [ ] **Step 2: Run the full targeted suite to verify it fails if exports are incomplete**

Run: `source .venv/bin/activate && pytest tests/dssd/verifier -v -s`
Expected: FAIL with `ImportError` because `DSSDVerifierSampler`, `VerifierSchedulerAdapter`, or `VerifierStateBridge` is not yet exported from `vllm.dssd.verifier`

- [ ] **Step 3: Adjust export surface and package cleanup**

```python
# vllm/dssd/verifier/__init__.py
from .engine import VerifierDecodeEngine
from .scheduler import VerifierSchedulerAdapter
from .sampler import DSSDVerifierSampler
from .state_bridge import VerifierStateBridge
from .types import (
    VerifierOpenSessionResult,
    VerifierRoundRequest,
    VerifierRoundResult,
    VerifierRoundState,
    VerifierSession,
)

__all__ = [
    "DSSDVerifierSampler",
    "VerifierDecodeEngine",
    "VerifierOpenSessionResult",
    "VerifierRoundRequest",
    "VerifierRoundResult",
    "VerifierRoundState",
    "VerifierSchedulerAdapter",
    "VerifierSession",
    "VerifierStateBridge",
]
```

- [ ] **Step 4: Run all targeted tests and the Python hook**

Run: `source .venv/bin/activate && pytest tests/dssd/verifier -v -s`
Expected: PASS with all verifier tests green

Run: `source .venv/bin/activate && python -m compileall vllm/dssd/verifier tests/dssd/verifier`
Expected: PASS with no syntax errors

Run: `source .venv/bin/activate && pre-commit run ruff-check --files vllm/dssd/verifier/__init__.py vllm/dssd/verifier/types.py vllm/dssd/verifier/scheduler.py vllm/dssd/verifier/state_bridge.py vllm/dssd/verifier/ops.py vllm/dssd/verifier/sampler.py vllm/dssd/verifier/engine.py tests/dssd/verifier/conftest.py tests/dssd/verifier/test_types.py tests/dssd/verifier/test_model_runner_hooks.py tests/dssd/verifier/test_scheduler_state.py tests/dssd/verifier/test_sampler.py tests/dssd/verifier/test_engine_smoke.py vllm/v1/worker/gpu/model_runner.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add vllm/dssd/verifier/__init__.py tests/dssd/verifier
git commit -m "test: verify dssd verifier target suite"
```
