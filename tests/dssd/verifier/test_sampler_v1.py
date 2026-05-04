from types import SimpleNamespace

import pytest
import torch

from vllm.dssd.verifier.sampler_v1 import DSSDVerifierSamplerV1
from vllm.dssd.verifier.types import VerifierRoundRequest
from vllm.v1.sample.metadata import SamplingMetadata


def _sampling_metadata() -> SamplingMetadata:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(1234)
    return SamplingMetadata(
        temperature=torch.tensor([1.0]),
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None,
        generators={0: generator},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(1),
        presence_penalties=torch.zeros(1),
        repetition_penalties=torch.ones(1),
        output_token_ids=[[]],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=SimpleNamespace(
            non_argmax_invariant=[],
            argmax_invariant=[],
            all=[],
        ),
        spec_token_ids=[[]],
    )


def _greedy_sampling_metadata() -> SamplingMetadata:
    metadata = _sampling_metadata()
    metadata.temperature = torch.tensor([0.0])
    metadata.all_greedy = True
    metadata.all_random = False
    return metadata


class _OldSampler:
    def __init__(self, sampled_token_id: int = 2) -> None:
        self.sampled_token_id = sampled_token_id
        self.calls: list[SimpleNamespace] = []

    @staticmethod
    def apply_logits_processors(logits, sampling_metadata, predict_bonus_token):
        del sampling_metadata, predict_bonus_token
        row_offsets = torch.arange(
            1,
            logits.shape[0] + 1,
            device=logits.device,
            dtype=logits.dtype,
        ).unsqueeze(1)
        return logits + (row_offsets * 0.25)

    @staticmethod
    def apply_temperature(logits, temperature, all_random):
        del all_random
        return logits.div_(temperature.unsqueeze(dim=1))

    def __call__(
        self,
        logits,
        sampling_metadata,
        predict_bonus_token=False,
        logprobs_mode_override=None,
    ):
        self.calls.append(
            SimpleNamespace(
                logits=logits.clone(),
                predict_bonus_token=predict_bonus_token,
                max_num_logprobs=sampling_metadata.max_num_logprobs,
                logprobs_mode_override=logprobs_mode_override,
            ))
        return SimpleNamespace(
            sampled_token_ids=torch.tensor([[self.sampled_token_id]],
                                           dtype=torch.int32))


def _metadata(device: torch.device) -> SimpleNamespace:
    return SimpleNamespace(
        target_logits_indices=torch.tensor([0, 1],
                                           device=device,
                                           dtype=torch.int64),
        bonus_logits_indices=torch.tensor([2], device=device, dtype=torch.int64),
    )


def test_zero_draft_round_returns_bonus_only() -> None:
    old_sampler = _OldSampler()
    sampler = DSSDVerifierSamplerV1(old_sampler)
    logits = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 3.0]],
        dtype=torch.float32,
    )
    sampling_metadata = _sampling_metadata()
    result = sampler.verify_round(
        logits=logits,
        spec_decode_metadata=_metadata(torch.device("cpu")),
        sampling_metadata=sampling_metadata,
        request=VerifierRoundRequest(
            req_id="req-1",
            committed_token_id=7,
            draft_token_ids=[],
            draft_q_values=[],
        ),
    )

    assert result.accepted_len == 0
    assert result.bonus_token_id == 2
    assert result.rejected_target_logits is None
    assert len(old_sampler.calls) == 1
    assert old_sampler.calls[0].predict_bonus_token
    assert old_sampler.calls[0].max_num_logprobs is None
    assert old_sampler.calls[0].logprobs_mode_override is None
    assert torch.equal(
        old_sampler.calls[0].logits,
        logits[_metadata(torch.device("cpu")).bonus_logits_indices],
    )


def test_zero_draft_round_uses_logits_directly_without_metadata() -> None:
    old_sampler = _OldSampler(sampled_token_id=1)
    sampler = DSSDVerifierSamplerV1(old_sampler)
    logits = torch.tensor([[0.0, 5.0, 1.0]], dtype=torch.float32)

    result = sampler.verify_round(
        logits=logits,
        spec_decode_metadata=None,
        sampling_metadata=_sampling_metadata(),
        request=VerifierRoundRequest(
            req_id="req-1",
            committed_token_id=7,
            draft_token_ids=[],
            draft_q_values=[],
        ),
    )

    assert result.accepted_len == 0
    assert result.bonus_token_id == 1
    assert result.rejected_target_logits is None
    assert len(old_sampler.calls) == 1
    assert old_sampler.calls[0].predict_bonus_token
    assert old_sampler.calls[0].max_num_logprobs is None
    assert torch.equal(old_sampler.calls[0].logits, logits)


def test_sample_bootstrap_uses_no_logprobs_and_returns_sampled_token() -> None:
    old_sampler = _OldSampler(sampled_token_id=11)
    sampler = DSSDVerifierSamplerV1(old_sampler)
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    sampling_metadata = _sampling_metadata()
    sampling_metadata.max_num_logprobs = 9

    result = sampler.sample_bootstrap(
        logits=logits,
        sampling_metadata=sampling_metadata,
    )

    assert result == 11
    assert len(old_sampler.calls) == 1
    assert not old_sampler.calls[0].predict_bonus_token
    assert old_sampler.calls[0].max_num_logprobs is None
    assert old_sampler.calls[0].logprobs_mode_override is None
    assert torch.equal(old_sampler.calls[0].logits, logits)


def test_zero_draft_round_uses_bonus_indices_strictly() -> None:
    sampler = DSSDVerifierSamplerV1(_OldSampler())
    logits = torch.tensor([[1.0, 0.0, 3.0]], dtype=torch.float32)

    with pytest.raises(IndexError):
        sampler.verify_round(
            logits=logits,
            spec_decode_metadata=_metadata(torch.device("cpu")),
            sampling_metadata=_sampling_metadata(),
            request=VerifierRoundRequest(
                req_id="req-1",
                committed_token_id=7,
                draft_token_ids=[],
                draft_q_values=[],
            ),
        )


def test_verify_round_rejects_mismatched_draft_inputs() -> None:
    sampler = DSSDVerifierSamplerV1(_OldSampler())

    with pytest.raises(ValueError, match="draft_token_ids .* draft_q_values"):
        sampler.verify_round(
            logits=torch.tensor([[1.0, 0.0, 3.0]], dtype=torch.float32),
            spec_decode_metadata=SimpleNamespace(
                target_logits_indices=torch.tensor([0], dtype=torch.int64),
                bonus_logits_indices=torch.tensor([0], dtype=torch.int64),
            ),
            sampling_metadata=_sampling_metadata(),
            request=VerifierRoundRequest(
                req_id="req-1",
                committed_token_id=7,
                draft_token_ids=[0],
                draft_q_values=[],
            ),
        )


def test_reject_round_returns_processed_target_logits() -> None:
    sampler = DSSDVerifierSamplerV1(_OldSampler())
    logits = torch.tensor(
        [[0.0, 3.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=torch.float32,
    )
    result = sampler.verify_round(
        logits=logits,
        spec_decode_metadata=_metadata(torch.device("cpu")),
        sampling_metadata=_sampling_metadata(),
        request=VerifierRoundRequest(
            req_id="req-1",
            committed_token_id=7,
            draft_token_ids=[0, 1],
            draft_q_values=[0.99, 0.99],
        ),
    )

    assert result.accepted_len in {0, 1}
    if result.accepted_len < 2:
        processed_target_logits = _OldSampler.apply_logits_processors(
            logits[_metadata(torch.device("cpu")).target_logits_indices].to(
                torch.float32).clone(),
            _sampling_metadata(),
            False,
        )
        assert torch.equal(
            result.rejected_target_logits,
            processed_target_logits[result.accepted_len],
        )
        assert result.bonus_token_id is None


def test_reject_round_returns_sampling_target_logits() -> None:
    sampler = DSSDVerifierSamplerV1(_OldSampler())
    sampling_metadata = _sampling_metadata()
    sampling_metadata.temperature = torch.tensor([2.0])
    sampling_metadata.top_k = torch.tensor([1], dtype=torch.int64)
    logits = torch.tensor(
        [[0.0, 4.0, 1.0], [5.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=torch.float32,
    )

    result = sampler.verify_round(
        logits=logits,
        spec_decode_metadata=_metadata(torch.device("cpu")),
        sampling_metadata=sampling_metadata,
        request=VerifierRoundRequest(
            req_id="req-1",
            committed_token_id=7,
            draft_token_ids=[0, 1],
            draft_q_values=[1.0, 1.0],
        ),
    )

    assert result.accepted_len == 0
    assert result.rejected_target_logits is not None
    assert torch.isneginf(result.rejected_target_logits[0])
    assert torch.isfinite(result.rejected_target_logits[1])
    assert torch.isneginf(result.rejected_target_logits[2])


def test_random_sampling_processors_expand_top_k_per_target_row() -> None:
    sampler = DSSDVerifierSamplerV1(_OldSampler())
    sampling_metadata = _sampling_metadata()
    sampling_metadata.top_k = torch.tensor([1], dtype=torch.int64)
    logits = torch.tensor(
        [[0.0, 100.0, 0.0], [0.0, 0.0, 10.0]],
        dtype=torch.float32,
    )

    processed = sampler._apply_random_sampling_processors(  # noqa: SLF001
        logits,
        sampling_metadata,
    )

    assert torch.isneginf(processed[0, 0])
    assert torch.isfinite(processed[0, 1])
    assert torch.isneginf(processed[0, 2])
    assert torch.isneginf(processed[1, 0])
    assert torch.isneginf(processed[1, 1])
    assert torch.isfinite(processed[1, 2])


def test_greedy_round_accepts_matching_draft_tokens_by_argmax() -> None:
    old_sampler = _OldSampler(sampled_token_id=2)
    sampler = DSSDVerifierSamplerV1(old_sampler)
    logits = torch.tensor(
        [[4.0, 1.0, 0.0], [0.0, 4.0, 1.0], [0.0, 0.0, 4.0]],
        dtype=torch.float32,
    )

    result = sampler.verify_round(
        logits=logits,
        spec_decode_metadata=_metadata(torch.device("cpu")),
        sampling_metadata=_greedy_sampling_metadata(),
        request=VerifierRoundRequest(
            req_id="req-1",
            committed_token_id=7,
            draft_token_ids=[0, 1],
            draft_q_values=[100.0, 100.0],
        ),
    )

    assert result.accepted_len == 2
    assert result.bonus_token_id == 2
    assert result.rejected_target_logits is None


def test_greedy_round_reject_returns_target_argmax_token_without_logits() -> None:
    sampler = DSSDVerifierSamplerV1(_OldSampler())
    logits = torch.tensor(
        [[4.0, 1.0, 0.0], [0.0, 1.0, 4.0], [0.0, 0.0, 4.0]],
        dtype=torch.float32,
    )

    result = sampler.verify_round(
        logits=logits,
        spec_decode_metadata=_metadata(torch.device("cpu")),
        sampling_metadata=_greedy_sampling_metadata(),
        request=VerifierRoundRequest(
            req_id="req-1",
            committed_token_id=7,
            draft_token_ids=[0, 1],
            draft_q_values=[100.0, 100.0],
        ),
    )

    assert result.accepted_len == 1
    assert result.rejected_token_id == 2
    assert result.rejected_target_logits is None
    assert result.bonus_token_id is None
