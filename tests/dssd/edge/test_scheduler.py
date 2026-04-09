from vllm.sampling_params import SamplingParams

from vllm.dssd.edge.scheduler import EdgeSchedulerAdapter
from vllm.dssd.edge.types import EdgeSession


class FakeBlocks:
    def __init__(self, block_ids):
        self._block_ids = block_ids

    def get_block_ids(self, allow_none=True):
        return self._block_ids


class FakeKVCacheManager:
    def __init__(
        self,
        allocated_block_ids=(([7, 8],), ([7, 8],)),
        new_block_id_batches=([41, 42], [51, 52]),
    ):
        self.allocate_calls = []
        self.freed_requests = []
        self._new_block_ids = []
        self._allocated_block_ids = list(allocated_block_ids)
        self._new_block_id_batches = list(new_block_id_batches)

    def allocate_slots(self, request, num_new_tokens):
        self.allocate_calls.append((request, num_new_tokens))
        batch_index = len(self.allocate_calls) - 1
        if batch_index < len(self._new_block_id_batches):
            self._new_block_ids = self._new_block_id_batches[batch_index]
        else:
            self._new_block_ids = []
        if batch_index < len(self._allocated_block_ids):
            return FakeBlocks(self._allocated_block_ids[batch_index])
        return FakeBlocks(None)

    def take_new_block_ids(self):
        value = self._new_block_ids
        self._new_block_ids = []
        return value

    def free(self, request):
        self.freed_requests.append(request)

    def set_new_block_ids(self, block_ids):
        self._new_block_ids = block_ids


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
    kv = FakeKVCacheManager()
    adapter = EdgeSchedulerAdapter(kv)
    session = make_session()

    block_ids = adapter.allocate_blocks(
        req_id=session.req_id,
        prompt_token_ids=session.prompt_token_ids,
        sampling_params=session.sampling_params,
    )
    kv.set_new_block_ids([99])
    prefill = adapter.build_prefill_step(session)
    preserved_queue_ids = kv.take_new_block_ids()
    decode = adapter.build_decode_step(session)

    prefill_request, prefill_num_new_tokens = kv.allocate_calls[0]
    decode_request, decode_num_new_tokens = kv.allocate_calls[1]

    assert block_ids == ([7, 8],)
    assert len(kv.allocate_calls) == 2
    assert prefill_num_new_tokens == len(session.prompt_token_ids)
    assert prefill.num_scheduled_tokens == {"req-1": 2}
    assert prefill.new_block_ids_to_zero == [41, 42]
    assert preserved_queue_ids == [99]
    assert decode_num_new_tokens == 1
    assert decode_num_new_tokens == (
        decode_request.num_tokens - decode_request.num_computed_tokens
    )
    assert list(decode_request.output_token_ids) == session.token_ids[
        session.prompt_len:
    ]
    assert decode.num_scheduled_tokens == {"req-1": 1}
    assert decode.scheduled_cached_reqs.req_ids == ["req-1"]
    assert decode.new_block_ids_to_zero == [51, 52]


def test_request_block_hasher_is_threaded_into_built_requests() -> None:
    kv = FakeKVCacheManager()
    request_block_hasher = lambda request: [f"hash:{tuple(request.all_token_ids)}"]
    adapter = EdgeSchedulerAdapter(
        kv, request_block_hasher=request_block_hasher
    )
    session = make_session()

    adapter.allocate_blocks(
        req_id=session.req_id,
        prompt_token_ids=session.prompt_token_ids,
        sampling_params=session.sampling_params,
    )
    adapter.build_decode_step(session)
    adapter.free_blocks(session)

    prefill_request, _ = kv.allocate_calls[0]
    decode_request, _ = kv.allocate_calls[1]
    close_request = kv.freed_requests[0]

    assert prefill_request.block_hashes == ["hash:(10, 11)"]
    assert decode_request.block_hashes == [
        "hash:(10, 11)",
        "hash:(10, 11, 20)",
    ]
    assert close_request.block_hashes == [
        "hash:(10, 11)",
        "hash:(10, 11, 20)",
    ]


def test_scheduler_adapter_scope_is_documented() -> None:
    assert EdgeSchedulerAdapter.__doc__ is not None
    assert "single-request" in EdgeSchedulerAdapter.__doc__
    assert "decoder-only text path" in EdgeSchedulerAdapter.__doc__


def test_decode_step_handles_missing_new_blocks() -> None:
    kv = FakeKVCacheManager(
        allocated_block_ids=(([7, 8],), None),
        new_block_id_batches=([41, 42], []),
    )
    adapter = EdgeSchedulerAdapter(kv)
    session = make_session()

    adapter.allocate_blocks(
        req_id=session.req_id,
        prompt_token_ids=session.prompt_token_ids,
        sampling_params=session.sampling_params,
    )
    prefill = adapter.build_prefill_step(session)
    decode = adapter.build_decode_step(session)

    assert prefill.new_block_ids_to_zero == [41, 42]
    assert decode.num_scheduled_tokens == {"req-1": 1}
    assert decode.total_num_scheduled_tokens == 1
    assert decode.scheduled_cached_reqs.req_ids == ["req-1"]
    assert decode.scheduled_cached_reqs.new_block_ids == [None]
    assert decode.new_block_ids_to_zero is None


def test_close_step_and_free_blocks() -> None:
    kv = FakeKVCacheManager()
    adapter = EdgeSchedulerAdapter(kv)
    session = make_session()

    adapter.free_blocks(session)
    close_step = adapter.build_close_step(session.req_id)

    assert len(kv.freed_requests) == 1
    assert close_step.finished_req_ids == {"req-1"}


def test_free_blocks_clears_pending_prefill_zeroing_metadata() -> None:
    kv = FakeKVCacheManager()
    adapter = EdgeSchedulerAdapter(kv)
    session = make_session()

    adapter.allocate_blocks(
        req_id=session.req_id,
        prompt_token_ids=session.prompt_token_ids,
        sampling_params=session.sampling_params,
    )
    kv.set_new_block_ids([99])
    adapter.free_blocks(session)
    prefill = adapter.build_prefill_step(session)

    assert prefill.new_block_ids_to_zero is None
    assert kv.take_new_block_ids() == [99]
