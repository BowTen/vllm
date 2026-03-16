# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.spec_decode.distributed.runtime import BaseCausalLMRuntime


class FakeCache:
    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = tuple(token_ids)

    def to_legacy_cache(self) -> tuple[int, ...]:
        return self.token_ids

    @classmethod
    def from_legacy_cache(cls, cache: tuple[int, ...]) -> "FakeCache":
        return cls(list(cache))


class FakeRuntime(BaseCausalLMRuntime):
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.processed_chunks: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def prefill_runtime_state(
        self,
        token_ids: list[int],
        num_prompt_logprobs: int | None = None,
    ):
        del num_prompt_logprobs
        cache, next_logits = self._run_tokens(token_ids, cache=None)
        return self._build_state(token_ids, cache, next_logits), []

    def _run_tokens(
        self,
        token_ids: list[int],
        cache: FakeCache | None,
    ) -> tuple[FakeCache, torch.Tensor]:
        prefix = list(cache.token_ids) if cache is not None else []
        self.processed_chunks.append((tuple(token_ids), tuple(prefix)))
        full_sequence = prefix + list(token_ids)
        next_logits = torch.tensor(
            [float(full_sequence[-1]), float(len(full_sequence))],
            dtype=torch.float32,
        )
        return FakeCache(full_sequence), next_logits

    def _build_state(self, token_ids, cache, next_logits):
        from vllm.v1.spec_decode.distributed.runtime import IncrementalRuntimeState

        return IncrementalRuntimeState(
            token_ids=list(token_ids),
            cache=cache,
            next_logits=next_logits,
        )


def test_sync_runtime_state_reuses_matching_prefix():
    runtime = FakeRuntime()

    state = runtime.build_runtime_state([1, 2, 3])
    runtime.processed_chunks.clear()

    synced = runtime.sync_runtime_state(state, [1, 2, 3, 4, 5])

    assert synced is state
    assert state.token_ids == [1, 2, 3, 4, 5]
    assert state.cache.token_ids == (1, 2, 3, 4, 5)
    assert torch.equal(state.next_logits, torch.tensor([5.0, 5.0]))
    assert runtime.processed_chunks == [((4, 5), (1, 2, 3))]


def test_sync_runtime_state_is_noop_for_identical_prefix():
    runtime = FakeRuntime()

    state = runtime.build_runtime_state([7, 8])
    runtime.processed_chunks.clear()

    synced = runtime.sync_runtime_state(state, [7, 8])

    assert synced is state
    assert runtime.processed_chunks == []


def test_sync_runtime_state_rebuilds_after_divergence():
    runtime = FakeRuntime()

    state = runtime.build_runtime_state([1, 2, 3])
    runtime.processed_chunks.clear()

    rebuilt = runtime.sync_runtime_state(state, [1, 9])

    assert rebuilt is not state
    assert rebuilt.token_ids == [1, 9]
    assert rebuilt.cache.token_ids == (1, 9)
    assert runtime.processed_chunks == [((1, 9), ())]


def test_clone_runtime_state_copies_cache_and_logits():
    runtime = FakeRuntime()

    state = runtime.build_runtime_state([3, 4])
    clone = runtime.clone_runtime_state(state)
    clone.token_ids.append(5)
    clone.next_logits[0] = -1

    assert clone.cache is not state.cache
    assert clone.cache.token_ids == state.cache.token_ids
    assert state.token_ids == [3, 4]
    assert torch.equal(state.next_logits, torch.tensor([4.0, 2.0]))
