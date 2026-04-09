from vllm.sampling_params import SamplingParams

from vllm.dssd.edge.types import EdgeRoundState, EdgeSession


def test_edge_session_append_and_rollback() -> None:
    session = EdgeSession(
        req_id="req-1",
        prompt_token_ids=[10, 11],
        sampling_params=SamplingParams(max_tokens=8),
        block_ids=([1],),
        prompt_len=2,
        num_computed_tokens=2,
        total_len=2,
        token_ids=[10, 11],
    )

    session.append_token(20, computed_delta=1)
    session.append_token(21, computed_delta=0)

    assert session.output_len == 2
    assert session.token_ids == [10, 11, 20, 21]
    assert session.num_computed_tokens == 3

    session.rollback(1)

    assert session.token_ids == [10, 11, 20]
    assert session.total_len == 3
    assert session.num_computed_tokens == 3
    assert session.committed_output_ids() == [20]


def test_edge_round_state_buffer_and_reset() -> None:
    round_state = EdgeRoundState()
    round_state.prepare_logits_buffer(
        gamma=2,
        vocab_size=4,
        device="cpu",
        dtype=None,
    )
    round_state.append_step(7, 0.25)

    assert round_state.logits_row_view(0).shape == (1, 4)
    assert round_state.draft_token_ids == [7]
    assert round_state.draft_q_values == [0.25]

    round_state.reset()

    assert round_state.draft_token_ids == []
    assert round_state.draft_q_values == []
    assert round_state.committed_token_id is None
