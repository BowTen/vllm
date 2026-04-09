from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verifier_demo.scheduler import VerifierSchedulerAdapter
from verifier_demo.types import VerifierRoundRequest, VerifierSession
from vllm.sampling_params import SamplingParams


class _FakeKVCacheBlocks:
    def __init__(self, block_ids: tuple[list[int], ...]) -> None:
        self._block_ids = block_ids

    def get_block_ids(
        self,
        allow_none: bool = False,
    ) -> tuple[list[int], ...] | None:
        if allow_none and all(len(group) == 0 for group in self._block_ids):
            return None
        return self._block_ids


class _FakeKVCacheManager:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def allocate_slots(self, request, num_new_tokens: int, **kwargs):
        self.calls.append(
            {
                "request": request,
                "num_new_tokens": num_new_tokens,
                "kwargs": kwargs,
            }
        )
        return _FakeKVCacheBlocks(([41, 42],))

    def take_new_block_ids(self) -> list[int]:
        return [41, 42]


class VerifierSchedulerAdapterTest(unittest.TestCase):
    def test_build_verify_step_allocates_new_blocks_for_cached_request(self) -> None:
        manager = _FakeKVCacheManager()
        adapter = VerifierSchedulerAdapter(kv_cache_manager=manager)
        session = VerifierSession(
            req_id="req-1",
            prompt_token_ids=[10, 11, 12],
            sampling_params=SamplingParams(max_tokens=16),
            block_ids=([1, 2],),
            prompt_len=3,
            num_computed_tokens=5,
            total_len=5,
            token_ids=[10, 11, 12, 20, 21],
        )
        request = VerifierRoundRequest(
            req_id="req-1",
            committed_token_id=22,
            draft_token_ids=[30, 31],
            draft_q_values=[0.4, 0.6],
        )

        step = adapter.build_verify_step(session, request)

        self.assertEqual(step.scheduled_cached_reqs.new_block_ids, [([41, 42],)])
        self.assertEqual(step.new_block_ids_to_zero, [41, 42])
        self.assertEqual(step.num_scheduled_tokens["req-1"], 3)

        call = manager.calls[0]
        built_request = call["request"]
        self.assertEqual(call["num_new_tokens"], 3)
        self.assertEqual(built_request.num_computed_tokens, 5)
        self.assertEqual(list(built_request.spec_token_ids), [30, 31])
        self.assertEqual(list(built_request.all_token_ids), [10, 11, 12, 20, 21, 22])


if __name__ == "__main__":
    unittest.main()
