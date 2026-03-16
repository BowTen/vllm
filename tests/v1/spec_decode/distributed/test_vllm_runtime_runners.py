# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.v1.engine import FinishReason
from vllm.v1.spec_decode.distributed import engine_runtime as runtime_mod
from vllm.v1.spec_decode.distributed.logprobs import pack_sample_logprobs
from vllm.v1.spec_decode.distributed.protocol import (
    DraftProposal,
    OpenSessionRequest,
    PackedLogprobs,
    SamplingMetadata,
)


def _sampling(*, logprobs: int | None = 1) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        max_tokens=8,
        min_tokens=0,
        stop_token_ids=[],
        eos_token_id=None,
        ignore_eos=False,
        logprobs=logprobs,
        prompt_logprobs=None,
    )


def _packed(probs: list[float], token_id: int) -> PackedLogprobs:
    packed = pack_sample_logprobs(
        torch.tensor(probs, dtype=torch.float32),
        sampled_token_id=token_id,
        num_logprobs=-1,
    )
    assert packed is not None
    return packed


class FakeRuntime:
    def __init__(self, responses: list[runtime_mod.EngineChunkResult]) -> None:
        self.vocab_size = 4
        self._responses = deque(responses)
        self.open_calls: list[tuple[str, list[int], SamplingMetadata, int | None]] = []
        self.run_calls: list[tuple[str, list[int], int, int | None]] = []
        self.batch_calls: list[list[tuple[str, list[int], int, int | None]]] = []
        self.closed: list[str] = []
        self.shutdown_called = False

    async def open_session(
        self,
        session_id: str,
        prompt_token_ids: list[int],
        sampling: SamplingMetadata,
        *,
        prompt_logprobs: int | None = None,
    ) -> list[PackedLogprobs]:
        self.open_calls.append(
            (session_id, list(prompt_token_ids), sampling, prompt_logprobs)
        )
        return []

    async def run_chunk(
        self,
        session_id: str,
        prefix_token_ids: list[int],
        sampling: SamplingMetadata,
        *,
        max_tokens: int,
        logprobs: int | None = None,
        prompt_logprobs: int | None = None,
    ) -> runtime_mod.EngineChunkResult:
        del sampling, prompt_logprobs
        self.run_calls.append((session_id, list(prefix_token_ids), max_tokens, logprobs))
        return self._responses.popleft()

    async def run_queries(
        self,
        queries: list[runtime_mod.EngineQuery],
    ) -> list[runtime_mod.EngineChunkResult]:
        batch: list[tuple[str, list[int], int, int | None]] = []
        outputs: list[runtime_mod.EngineChunkResult] = []
        for query in queries:
            record = (
                query.session_id,
                list(query.prefix_token_ids),
                query.max_tokens,
                query.logprobs,
            )
            self.run_calls.append(record)
            batch.append(record)
            outputs.append(self._responses.popleft())
        self.batch_calls.append(batch)
        return outputs

    async def close_session(self, session_id: str) -> None:
        self.closed.append(session_id)

    def shutdown(self) -> None:
        self.shutdown_called = True


def _replace_namespace(obj: Any, /, **changes: Any) -> Any:
    updated = vars(obj).copy()
    updated.update(changes)
    return SimpleNamespace(**updated)


def _fake_structured_output_factory(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    return SimpleNamespace(
        create_session=lambda *a, **k: None,
        close=lambda: None,
    )


@pytest.mark.asyncio
async def test_vllm_target_runner_returns_reject_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_runtime = FakeRuntime(
        [
            runtime_mod.EngineChunkResult(
                token_ids=[1],
                logprobs=[_packed([0.0, 1.0, 0.0, 0.0], 1)],
                prompt_logprobs=[],
                finish_reason=FinishReason.LENGTH,
                stop_reason=None,
            ),
            runtime_mod.EngineChunkResult(
                token_ids=[3],
                logprobs=[_packed([0.0, 0.0, 0.2, 0.8], 3)],
                prompt_logprobs=[],
                finish_reason=FinishReason.LENGTH,
                stop_reason=None,
            ),
        ]
    )
    monkeypatch.setattr(
        runtime_mod,
        "build_target_runtime_config",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(runtime_mod, "VllmEngineSessionRuntime", lambda _cfg: fake_runtime)
    monkeypatch.setattr(
        runtime_mod,
        "StructuredOutputFactory",
        _fake_structured_output_factory,
    )

    runner = runtime_mod.VllmTargetVerificationRunner(
        model_name="fake-model",
        device=None,
        dtype="auto",
        trust_remote_code=False,
    )
    await runner.open_session(
        OpenSessionRequest(
            session_id="session-1",
            prompt_token_ids=[9, 8],
            sampling_metadata=_sampling(logprobs=1),
            initial_version=4,
        )
    )

    result = await runner.verify_proposal(
        DraftProposal(
            session_id="session-1",
            proposal_id=3,
            base_version=4,
            accepted_prefix_len=2,
            draft_token_ids=[1, 2],
            draft_token_probs=[1.0, 1.0],
        )
    )

    assert fake_runtime.run_calls == [
        ("session-1", [9, 8], 1, -1),
        ("session-1", [9, 8, 1], 1, -1),
    ]
    assert result.accepted_len == 1
    assert result.accepted_token_ids == [1]
    assert result.reject_pos == 1
    assert result.target_probs_at_reject_pos is not None
    assert result.verifier_version == 5
    assert len(result.accepted_logprobs) == 1


@pytest.mark.asyncio
async def test_vllm_target_runner_returns_bonus_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_runtime = FakeRuntime(
        [
            runtime_mod.EngineChunkResult(
                token_ids=[1],
                logprobs=[_packed([0.0, 1.0, 0.0, 0.0], 1)],
                prompt_logprobs=[],
                finish_reason=FinishReason.LENGTH,
                stop_reason=None,
            ),
            runtime_mod.EngineChunkResult(
                token_ids=[2],
                logprobs=[_packed([0.0, 0.0, 1.0, 0.0], 2)],
                prompt_logprobs=[],
                finish_reason=FinishReason.LENGTH,
                stop_reason=None,
            ),
            runtime_mod.EngineChunkResult(
                token_ids=[3],
                logprobs=[_packed([0.0, 0.0, 0.0, 1.0], 3)],
                prompt_logprobs=[],
                finish_reason=FinishReason.LENGTH,
                stop_reason=None,
            ),
        ]
    )
    monkeypatch.setattr(
        runtime_mod,
        "build_target_runtime_config",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(runtime_mod, "VllmEngineSessionRuntime", lambda _cfg: fake_runtime)
    monkeypatch.setattr(
        runtime_mod,
        "StructuredOutputFactory",
        _fake_structured_output_factory,
    )

    runner = runtime_mod.VllmTargetVerificationRunner(
        model_name="fake-model",
        device=None,
        dtype="auto",
        trust_remote_code=False,
    )
    await runner.open_session(
        OpenSessionRequest(
            session_id="session-2",
            prompt_token_ids=[4],
            sampling_metadata=_sampling(logprobs=1),
            initial_version=0,
        )
    )

    result = await runner.verify_proposal(
        DraftProposal(
            session_id="session-2",
            proposal_id=0,
            base_version=0,
            accepted_prefix_len=1,
            draft_token_ids=[1, 2],
            draft_token_probs=[1.0, 1.0],
            draft_stopped=False,
        )
    )

    assert fake_runtime.run_calls == [
        ("session-2", [4], 1, -1),
        ("session-2", [4, 1], 1, -1),
        ("session-2", [4, 1, 2], 1, -1),
    ]
    assert result.accepted_len == 2
    assert result.accepted_token_ids == [1, 2]
    assert result.bonus_token_id == 3
    assert result.reject_pos is None
    assert result.verifier_version == 3
    assert len(result.accepted_logprobs) == 2
    assert result.bonus_logprobs is not None


@pytest.mark.asyncio
async def test_vllm_target_runner_batches_verification_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_runtime = FakeRuntime(
        [
            runtime_mod.EngineChunkResult(
                token_ids=[1],
                logprobs=[_packed([0.0, 1.0, 0.0, 0.0], 1)],
                prompt_logprobs=[],
                finish_reason=FinishReason.LENGTH,
                stop_reason=None,
            ),
            runtime_mod.EngineChunkResult(
                token_ids=[3],
                logprobs=[_packed([0.0, 0.0, 0.0, 1.0], 3)],
                prompt_logprobs=[],
                finish_reason=FinishReason.LENGTH,
                stop_reason=None,
            ),
            runtime_mod.EngineChunkResult(
                token_ids=[2],
                logprobs=[_packed([0.0, 0.0, 1.0, 0.0], 2)],
                prompt_logprobs=[],
                finish_reason=FinishReason.LENGTH,
                stop_reason=None,
            ),
        ]
    )
    monkeypatch.setattr(
        runtime_mod,
        "build_target_runtime_config",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(runtime_mod, "VllmEngineSessionRuntime", lambda _cfg: fake_runtime)
    monkeypatch.setattr(
        runtime_mod,
        "StructuredOutputFactory",
        _fake_structured_output_factory,
    )

    runner = runtime_mod.VllmTargetVerificationRunner(
        model_name="fake-model",
        device=None,
        dtype="auto",
        trust_remote_code=False,
    )
    await runner.open_session(
        OpenSessionRequest(
            session_id="session-a",
            prompt_token_ids=[9],
            sampling_metadata=_sampling(logprobs=1),
            initial_version=0,
        )
    )
    await runner.open_session(
        OpenSessionRequest(
            session_id="session-b",
            prompt_token_ids=[8],
            sampling_metadata=_sampling(logprobs=1),
            initial_version=0,
        )
    )

    results = await runner.verify_proposals_batch(
        [
            DraftProposal(
                session_id="session-a",
                proposal_id=0,
                base_version=0,
                accepted_prefix_len=1,
                draft_token_ids=[1],
                draft_token_probs=[1.0],
                draft_stopped=False,
            ),
            DraftProposal(
                session_id="session-b",
                proposal_id=1,
                base_version=0,
                accepted_prefix_len=1,
                draft_token_ids=[2],
                draft_token_probs=[1.0],
                draft_stopped=True,
            ),
        ]
    )

    assert [result.accepted_token_ids for result in results] == [[1], []]
    assert results[0].bonus_token_id == 2
    assert results[1].reject_pos == 0
    assert fake_runtime.batch_calls == [
        [
            ("session-a", [9], 1, -1),
            ("session-b", [8], 1, -1),
        ],
        [
            ("session-a", [9, 1], 1, -1),
        ],
    ]
    assert runner._sessions["session-a"].version == 2
    assert runner._sessions["session-b"].version == 0


def test_single_worker_runtime_config_disables_async_scheduling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_mod, "replace", _replace_namespace)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            data_parallel_size=2,
            decode_context_parallel_size=2,
            prefill_context_parallel_size=2,
            distributed_executor_backend="mp",
            rank=7,
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=16,
            max_num_batched_tokens=32,
            max_num_scheduled_tokens=32,
            async_scheduling=True,
        ),
        speculative_config=object(),
    )

    updated = runtime_mod._single_worker_runtime_config(config)

    assert updated.parallel_config.tensor_parallel_size == 1
    assert updated.parallel_config.pipeline_parallel_size == 1
    assert updated.parallel_config.data_parallel_size == 1
    assert updated.parallel_config.distributed_executor_backend == "uni"
    assert updated.scheduler_config.max_num_seqs == 1
    assert updated.scheduler_config.async_scheduling is False
    assert updated.speculative_config is None


def test_build_target_runtime_config_disables_async_scheduling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=True),
        speculative_config=object(),
        device_config=SimpleNamespace(device=torch.device("cpu"), device_type="cpu"),
    )

    class FakeEngineArgs:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def create_engine_config(self) -> Any:
            return fake_config

    monkeypatch.setattr(runtime_mod, "EngineArgs", FakeEngineArgs)
    monkeypatch.setattr(runtime_mod, "replace", _replace_namespace)

    updated = runtime_mod.build_target_runtime_config(
        "fake-model",
        device=None,
        dtype="auto",
        trust_remote_code=False,
    )

    assert updated.scheduler_config.async_scheduling is False
    assert updated.speculative_config is None
