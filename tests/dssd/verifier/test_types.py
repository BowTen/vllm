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
    result = VerifierRoundResult(req_id="req-1", accepted_len=2,
                                 bonus_token_id=17)
    assert result.is_all_accepted(draft_len=2)
    assert not result.is_rejected(draft_len=2)

    bootstrap = VerifierOpenSessionResult(req_id="req-1",
                                          bootstrap_token_id=7)
    assert bootstrap.bootstrap_token_id == 7
