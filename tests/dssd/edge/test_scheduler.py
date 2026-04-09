from vllm.sampling_params import SamplingParams

from vllm.dssd.edge.scheduler import EdgeSchedulerAdapter
from vllm.dssd.edge.types import EdgeSession


class FakeBlocks:
    def __init__(self, block_ids):
        self._block_ids = block_ids

    def get_block_ids(self, allow_none=True):
        return self._block_ids


class FakeKVCacheManager:
    def __init__(self):
        self.allocate_calls = []
        self.freed_requests = []
        self._new_block_ids = []
        self._new_block_id_batches = [[41, 42], [51, 52]]

    def allocate_slots(self, request, num_new_tokens):
        self.allocate_calls.append((request, num_new_tokens))
        batch_index = len(self.allocate_calls) - 1
        if batch_index < len(self._new_block_id_batches):
            self._new_block_ids = self._new_block_id_batches[batch_index]
        else:
            self._new_block_ids = []
        return FakeBlocks(([7, 8],))

    def take_new_block_ids(self):
        value = self._new_block_ids
        self._new_block_ids = []
        return value

    def free(self, request):
        self.freed_requests.append(request)


def make_session() -> EdgeSession:
    return EdgeSession(
        req_id="req-1",
        prompt_token_ids=[10, 11],
        sampling_params=SamplingParams(max_tokens=8),
        block_ids=([1, 2],),
        prompt_len=2,
        num_computed_tokens=2,
        total_len=3,
        token_ids=[10, 11, 20],
    )


def test_prefill_and_decode_steps_use_kv_cache_manager() -> None:
    adapter = EdgeSchedulerAdapter(FakeKVCacheManager())
    session = make_session()

    block_ids = adapter.allocate_blocks(
        req_id=session.req_id,
        prompt_token_ids=session.prompt_token_ids,
        sampling_params=session.sampling_params,
    )
    prefill = adapter.build_prefill_step(session)
    decode = adapter.build_decode_step(session)

    assert block_ids == ([7, 8],)
    assert prefill.num_scheduled_tokens == {"req-1": 2}
    assert prefill.new_block_ids_to_zero == [41, 42]
    assert decode.num_scheduled_tokens == {"req-1": 1}
    assert decode.scheduled_cached_reqs.req_ids == ["req-1"]
    assert decode.new_block_ids_to_zero == [51, 52]


def test_close_step_and_free_blocks() -> None:
    kv = FakeKVCacheManager()
    adapter = EdgeSchedulerAdapter(kv)
    session = make_session()

    adapter.free_blocks(session)
    close_step = adapter.build_close_step(session.req_id)

    assert len(kv.freed_requests) == 1
    assert close_step.finished_req_ids == {"req-1"}
