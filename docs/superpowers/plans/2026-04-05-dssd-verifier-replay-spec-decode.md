# DSSD Verifier Replay Spec-Decode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `DSSDVerifierExecutionRequest` 接入真实 verifier logits 前向路径，先用单请求 replay 方案复用 `GPUModelRunner` 的 spec-decode 输入准备逻辑，并返回真实 `VerifierForwardResult`。

**Architecture:** 把“把 DSSD verifier 请求变成单请求 spec-decode batch”的逻辑放进 `vllm/v1/dssd/worker/verifier_runner.py`，先做成可单测的纯 helper，再让 `GPUModelRunner.dssd_verify_round()` 只做薄委托。第一版不做 worker-side KV 复用，而是每轮重放 `committed_token_ids` 并把 `draft_token_ids` 作为 `scheduled_spec_decode_tokens`，直接调用现有 `execute_model()` 产出 logits，再从 `execute_model_state.spec_decode_metadata` 提取 `P_1...P_{gamma+1}`。

**Tech Stack:** Python 3.12, PyTorch, msgspec, pytest, vLLM V1 worker path, `SchedulerOutput`, `SamplingParams`

---

## Scope and Constraints

- 只覆盖 verifier 路径；`dssd_draft_round()` 继续保持 placeholder。
- 只做 replay，不做 worker-side verifier session / KV 复用。
- 只支持 text-only verifier worker；对 pooling、multimodal、async scheduling fail-closed。
- 第一版要求 `committed_token_ids` 非空；如果 prefix 为空，显式抛错，后续单独处理 BOS / empty-prefix 语义。
- synthetic replay request 每轮执行后立即从 `input_batch` 和 `requests` 清理，不留下持久 worker batch 状态。

## Planned Files

### Verifier Helper Surface

- Modify: `vllm/v1/dssd/worker/verifier_runner.py`
  增加 replay request view、synthetic `SchedulerOutput` builder、以及可单测的 verifier forward orchestration helper。

### GPU Model Runner Integration

- Modify: `vllm/v1/worker/gpu_model_runner.py`
  把 `dssd_verify_round()` 从 placeholder 改成薄委托，直接调用 verifier helper。

### Tests

- Modify: `tests/v1/worker/test_dssd_verifier_runner.py`
  增加 replay request view、scheduler output builder、orchestration cleanup 的 focused unit tests。
- Modify: `tests/v1/worker/test_dssd_worker_base.py`
  只在需要时更新 import/load 顺序；接口形状不变时不改断言。

---

### Task 1: Build Replay Request View and Synthetic SchedulerOutput

**Files:**
- Modify: `vllm/v1/dssd/worker/verifier_runner.py`
- Modify: `tests/v1/worker/test_dssd_verifier_runner.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/v1/worker/test_dssd_verifier_runner.py
import math

from vllm.v1.dssd.worker.verifier_runner import (
    build_verifier_replay_request_view,
    build_verifier_replay_scheduler_output,
)


def test_build_verifier_replay_request_view_preserves_prefix_and_spec_layout():
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-1",
        seq_no=3,
        committed_token_ids=[11, 12, 13],
        draft_token_ids=[21, 22],
        q_values=[0.6, 0.4],
    )

    view = build_verifier_replay_request_view(
        request,
        block_sizes=(4, 8),
    )

    assert view.request_id == "dssd-verify:vs-1:3"
    assert view.prompt_token_ids == [11, 12, 13]
    assert view.spec_token_ids == [21, 22]
    assert view.num_scheduled_tokens == 5
    assert view.block_ids == ([0, 1], [0])


def test_build_verifier_replay_scheduler_output_uses_committed_prompt_and_spec_tokens():
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-2",
        seq_no=4,
        committed_token_ids=[101, 102, 103],
        draft_token_ids=[201, 202],
        q_values=[0.3, 0.7],
    )

    view, scheduler_output = build_verifier_replay_scheduler_output(
        request,
        block_sizes=(4,),
    )

    assert scheduler_output.num_scheduled_tokens == {view.request_id: 5}
    assert scheduler_output.total_num_scheduled_tokens == 5
    assert scheduler_output.scheduled_spec_decode_tokens == {
        view.request_id: [201, 202],
    }

    new_req = scheduler_output.scheduled_new_reqs[0]
    assert new_req.req_id == "dssd-verify:vs-2:4"
    assert new_req.prompt_token_ids == [101, 102, 103]
    assert new_req.block_ids == ([0, 1],)
    assert new_req.num_computed_tokens == 0
    assert new_req.sampling_params.temperature == 0.0
    assert new_req.sampling_params.max_tokens == 1


def test_build_verifier_replay_request_view_rejects_empty_committed_prefix():
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-3",
        seq_no=0,
        committed_token_ids=[],
        draft_token_ids=[7],
        q_values=[0.9],
    )

    with pytest.raises(ValueError, match="committed_token_ids"):
        build_verifier_replay_request_view(request, block_sizes=(16,))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
pytest --noconftest tests/v1/worker/test_dssd_verifier_runner.py -q -k "replay_request_view or replay_scheduler_output"
```

Expected:

```text
3 failed, 0 passed because the replay helper functions do not exist yet
```

- [ ] **Step 3: Write the minimal implementation**

```python
# vllm/v1/dssd/worker/verifier_runner.py
from dataclasses import dataclass
from math import ceil

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)


@dataclass(frozen=True)
class DSSDVerifierReplayRequestView:
    request_id: str
    prompt_token_ids: list[int]
    spec_token_ids: list[int]
    num_scheduled_tokens: int
    block_ids: tuple[list[int], ...]


def build_verifier_replay_request_view(
    request: DSSDVerifierExecutionRequest,
    *,
    block_sizes: tuple[int, ...],
) -> DSSDVerifierReplayRequestView:
    if not request.committed_token_ids:
        raise ValueError("DSSD verifier replay requires committed_token_ids")
    if not request.draft_token_ids:
        raise ValueError("DSSD verifier replay requires draft_token_ids")
    if len(request.q_values) != len(request.draft_token_ids):
        raise ValueError("q_values must align with draft_token_ids")
    if not block_sizes:
        raise ValueError("block_sizes must not be empty")

    total_num_tokens = len(request.committed_token_ids) + len(request.draft_token_ids)
    block_ids = tuple(
        list(range(ceil(total_num_tokens / block_size)))
        for block_size in block_sizes
    )
    return DSSDVerifierReplayRequestView(
        request_id=f"dssd-verify:{request.verifier_session_id}:{request.seq_no}",
        prompt_token_ids=list(request.committed_token_ids),
        spec_token_ids=list(request.draft_token_ids),
        num_scheduled_tokens=total_num_tokens,
        block_ids=block_ids,
    )


def build_verifier_replay_scheduler_output(
    request: DSSDVerifierExecutionRequest,
    *,
    block_sizes: tuple[int, ...],
) -> tuple[DSSDVerifierReplayRequestView, SchedulerOutput]:
    view = build_verifier_replay_request_view(request, block_sizes=block_sizes)
    new_req = NewRequestData(
        req_id=view.request_id,
        prompt_token_ids=view.prompt_token_ids,
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
        pooling_params=None,
        block_ids=view.block_ids,
        num_computed_tokens=0,
        lora_request=None,
    )
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[new_req],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={view.request_id: view.num_scheduled_tokens},
        total_num_scheduled_tokens=view.num_scheduled_tokens,
        scheduled_spec_decode_tokens={view.request_id: view.spec_token_ids},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )
    return view, scheduler_output
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
source .venv/bin/activate
pytest --noconftest tests/v1/worker/test_dssd_verifier_runner.py -q -k "replay_request_view or replay_scheduler_output"
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Commit**

```bash
git add vllm/v1/dssd/worker/verifier_runner.py \
  tests/v1/worker/test_dssd_verifier_runner.py
git commit -m "feat: add verifier replay request builders"
```

### Task 2: Delegate GPU Verifier Forward to Replay Helper

**Files:**
- Modify: `vllm/v1/dssd/worker/verifier_runner.py`
- Modify: `vllm/v1/worker/gpu_model_runner.py`
- Modify: `tests/v1/worker/test_dssd_verifier_runner.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/v1/worker/test_dssd_verifier_runner.py
from unittest.mock import Mock

from vllm.v1.dssd.worker.verifier_runner import run_verifier_replay_forward


def test_run_verifier_replay_forward_executes_model_and_cleans_up_batch_state():
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-4",
        seq_no=6,
        committed_token_ids=[1, 2, 3],
        draft_token_ids=[4, 5],
        q_values=[0.2, 0.8],
    )
    metadata = types.SimpleNamespace(
        target_logits_indices=torch.tensor([0, 1], dtype=torch.int32),
        bonus_logits_indices=torch.tensor([2], dtype=torch.int32),
    )
    fake_runner = types.SimpleNamespace(
        use_async_scheduling=False,
        is_pooling_model=False,
        supports_mm_inputs=False,
        kv_cache_config=types.SimpleNamespace(
            kv_cache_groups=[
                types.SimpleNamespace(
                    kv_cache_spec=types.SimpleNamespace(block_size=16)
                )
            ]
        ),
        execute_model=Mock(return_value=None),
        execute_model_state=types.SimpleNamespace(
            logits=torch.tensor(
                [
                    [4.0, 0.0],
                    [0.0, 4.0],
                    [2.0, 0.0],
                ],
                dtype=torch.float32,
            ),
            spec_decode_metadata=metadata,
        ),
        input_batch=types.SimpleNamespace(
            prev_sampled_token_ids="stale",
            remove_request=Mock(return_value=0),
            condense=Mock(),
        ),
        requests={"dssd-verify:vs-4:6": object()},
        num_prompt_logprobs={"dssd-verify:vs-4:6": 1},
        late_interaction_runner=types.SimpleNamespace(
            on_requests_finished=Mock(),
        ),
        _draft_token_ids=[99],
        _draft_token_req_ids=["old-req"],
    )

    result = run_verifier_replay_forward(fake_runner, request)

    fake_runner.execute_model.assert_called_once()
    fake_runner.input_batch.remove_request.assert_called_once_with(
        "dssd-verify:vs-4:6"
    )
    fake_runner.input_batch.condense.assert_called_once_with()
    fake_runner.late_interaction_runner.on_requests_finished.assert_called_once_with(
        {"dssd-verify:vs-4:6"}
    )
    assert fake_runner.execute_model_state is None
    assert fake_runner.input_batch.prev_sampled_token_ids is None
    assert fake_runner._draft_token_ids is None
    assert fake_runner._draft_token_req_ids is None
    assert result.verifier_session_id == "vs-4"
    assert result.seq_no == 6
    assert len(result.seq_probs) == 2


def test_run_verifier_replay_forward_rejects_async_scheduling():
    request = DSSDVerifierExecutionRequest(
        binding_id="bind-1",
        verifier_session_id="vs-5",
        seq_no=0,
        committed_token_ids=[7],
        draft_token_ids=[8],
        q_values=[0.5],
    )
    fake_runner = types.SimpleNamespace(use_async_scheduling=True)

    with pytest.raises(NotImplementedError, match="async scheduling"):
        run_verifier_replay_forward(fake_runner, request)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
pytest --noconftest tests/v1/worker/test_dssd_verifier_runner.py -q -k "run_verifier_replay_forward"
```

Expected:

```text
2 failed because the replay forward helper does not exist yet
```

- [ ] **Step 3: Write the minimal implementation**

```python
# vllm/v1/dssd/worker/verifier_runner.py
def _get_verifier_replay_block_sizes(model_runner) -> tuple[int, ...]:
    return tuple(
        group.kv_cache_spec.block_size
        for group in model_runner.kv_cache_config.kv_cache_groups
    )


def _cleanup_verifier_replay_request(model_runner, request_id: str) -> None:
    model_runner.requests.pop(request_id, None)
    model_runner.num_prompt_logprobs.pop(request_id, None)
    model_runner.late_interaction_runner.on_requests_finished({request_id})
    removed = model_runner.input_batch.remove_request(request_id)
    if removed is not None:
        model_runner.input_batch.condense()


def run_verifier_replay_forward(
    model_runner,
    request: DSSDVerifierExecutionRequest,
) -> VerifierForwardResult:
    if not isinstance(request, DSSDVerifierExecutionRequest):
        raise TypeError("dssd_verify_round expects DSSDVerifierExecutionRequest")
    if model_runner.use_async_scheduling:
        raise NotImplementedError(
            "DSSD verifier replay does not support async scheduling yet"
        )
    if model_runner.is_pooling_model:
        raise NotImplementedError(
            "DSSD verifier replay does not support pooling models"
        )
    if model_runner.supports_mm_inputs:
        raise NotImplementedError(
            "DSSD verifier replay does not support multimodal models"
        )

    view, scheduler_output = build_verifier_replay_scheduler_output(
        request,
        block_sizes=_get_verifier_replay_block_sizes(model_runner),
    )
    result = model_runner.execute_model(scheduler_output)
    if result is not None:
        raise RuntimeError(
            "DSSD verifier replay expected execute_model to cache logits state"
        )

    state = model_runner.execute_model_state
    try:
        if state is None or state.spec_decode_metadata is None:
            raise RuntimeError(
                "DSSD verifier replay expected spec decode metadata"
            )
        return build_verifier_result_from_logits(
            request=request,
            logits=state.logits,
            metadata=state.spec_decode_metadata,
            finish_reason="gpu-replay-forward",
        )
    finally:
        model_runner.execute_model_state = None
        model_runner._draft_token_ids = None
        model_runner._draft_token_req_ids = None
        model_runner.input_batch.prev_sampled_token_ids = None
        _cleanup_verifier_replay_request(model_runner, view.request_id)
```

```python
# vllm/v1/worker/gpu_model_runner.py
def dssd_verify_round(self, request: Any) -> Any:
    from vllm.v1.dssd.worker.verifier_runner import run_verifier_replay_forward

    return run_verifier_replay_forward(self, request)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
source .venv/bin/activate
pytest --noconftest \
  tests/v1/worker/test_dssd_verifier_runner.py \
  tests/v1/worker/test_dssd_worker_base.py \
  tests/v1/dssd/test_verifier_service.py -q
```

Expected:

```text
16 passed
```

- [ ] **Step 5: Commit**

```bash
git add vllm/v1/dssd/worker/verifier_runner.py \
  vllm/v1/worker/gpu_model_runner.py \
  tests/v1/worker/test_dssd_verifier_runner.py
git commit -m "feat: run DSSD verifier replay through gpu model runner"
```

## Notes for Execution

- `run_verifier_replay_forward()` 必须是可单测的普通函数，避免把主要逻辑埋进 `GPUModelRunner` 大文件里。
- `build_verifier_replay_scheduler_output()` 里 `prompt_token_ids` 只放 committed prefix；`draft_token_ids` 只通过 `scheduled_spec_decode_tokens` 进入 batch，这样 `_calc_spec_decode_metadata()` 才会把 logits 索引对齐到 `m-1 ... m+gamma-1`。
- cleanup 必须在 `finally` 里做，否则失败路径会把 synthetic replay request 残留在 persistent batch 里，下一轮会污染输入状态。
- 本计划故意不做 verifier worker session / KV 复用。只要 replay logits 正确，再单独规划下一轮“把 committed prefix replay 换成真实 request/KV state”。
