import pytest
import torch

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

from experiments.dssd_correctness.sampling import (
    DSSDReferenceSamplingConfig,
    apply_reference_logits_processors,
    sample_from_processed_logits,
    sample_vllm_v1_random,
)


def test_sample_vllm_v1_random_matches_exponential_race_with_generator():
    probs = torch.tensor(
        [[0.1, 0.7, 0.2], [0.4, 0.3, 0.3]], dtype=torch.float32
    )
    sample_generator = torch.Generator(device="cpu").manual_seed(17)
    expected_generator = torch.Generator(device="cpu").manual_seed(17)

    q = torch.empty_like(probs)
    for row in range(probs.shape[0]):
        q[row].exponential_(generator=expected_generator)
    expected = (probs / q).argmax(dim=-1).view(-1)

    actual = sample_vllm_v1_random(probs, sample_generator)

    assert torch.equal(actual, expected)


def test_apply_reference_logits_processors_applies_temperature_and_top_k():
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float16)
    config = DSSDReferenceSamplingConfig(temperature=2.0, top_k=2)

    processed = apply_reference_logits_processors(logits, config)

    expected = torch.tensor(
        [[-float("inf"), -float("inf"), 1.5, 2.0]], dtype=torch.float32
    )
    assert processed.dtype == torch.float32
    torch.testing.assert_close(processed, expected)


def test_apply_reference_logits_processors_keeps_top_k_cutoff_ties():
    logits = torch.tensor([[1.0, 2.0, 2.0]], dtype=torch.float32)
    config = DSSDReferenceSamplingConfig(top_k=1)

    processed = apply_reference_logits_processors(logits, config)

    expected = torch.tensor([[-float("inf"), 2.0, 2.0]], dtype=torch.float32)
    torch.testing.assert_close(processed, expected)


def test_apply_reference_logits_processors_masks_top_p_like_vllm():
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32)
    config = DSSDReferenceSamplingConfig(top_p=0.6)

    processed = apply_reference_logits_processors(logits, config)

    sorted_logits, sorted_indices = logits.sort(dim=-1, descending=False)
    sorted_probs = sorted_logits.softmax(dim=-1)
    top_p_mask = torch.cumsum(sorted_probs, dim=-1) <= 1 - config.top_p
    top_p_mask[:, -1] = False
    expected_sorted = sorted_logits.masked_fill(top_p_mask, -float("inf"))
    expected = torch.empty_like(logits).scatter(
        dim=-1, index=sorted_indices, src=expected_sorted
    )

    torch.testing.assert_close(processed, expected)
    assert torch.isfinite(processed[0, 3])


def test_apply_reference_logits_processors_matches_vllm_top_k_top_p():
    logits = torch.tensor(
        [[0.0, 1.0, 1.0, 3.0], [2.0, -1.0, 0.5, 0.5]], dtype=torch.float32
    )
    config = DSSDReferenceSamplingConfig(top_k=2, top_p=0.75)

    processed = apply_reference_logits_processors(logits, config)
    expected = apply_top_k_top_p_pytorch(
        logits.clone(),
        k=torch.tensor([config.top_k] * logits.shape[0], dtype=torch.int64),
        p=torch.tensor([config.top_p] * logits.shape[0], dtype=torch.float32),
    )

    torch.testing.assert_close(processed, expected)


def test_apply_reference_logits_processors_supports_arbitrary_leading_dims():
    logits = torch.tensor(
        [[[0.0, 1.0, 2.0], [3.0, 1.0, 3.0]]], dtype=torch.float32
    )
    config = DSSDReferenceSamplingConfig(top_k=1, top_p=0.8)

    processed = apply_reference_logits_processors(logits, config)
    flat_expected = apply_top_k_top_p_pytorch(
        logits.reshape(-1, logits.shape[-1]).clone(),
        k=torch.tensor([config.top_k] * 2, dtype=torch.int64),
        p=torch.tensor([config.top_p] * 2, dtype=torch.float32),
    )

    assert processed.shape == logits.shape
    torch.testing.assert_close(processed, flat_expected.reshape_as(logits))


def test_apply_reference_logits_processors_does_not_mutate_input():
    logits = torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.float32)
    original = logits.clone()
    config = DSSDReferenceSamplingConfig(temperature=0.5, top_k=1, top_p=0.9)

    _ = apply_reference_logits_processors(logits, config)

    torch.testing.assert_close(logits, original)


def test_sample_from_processed_logits_returns_greedy_for_zero_temperature():
    logits = torch.tensor([0.0, 2.0, 1.0], dtype=torch.float32)

    token_id, q_value = sample_from_processed_logits(
        logits, DSSDReferenceSamplingConfig(temperature=0.0), None
    )

    assert token_id == 1
    assert q_value == 1.0


def test_sample_from_processed_logits_returns_sampled_token_probability():
    logits = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(29)
    expected_generator = torch.Generator(device="cpu").manual_seed(29)
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    q = torch.empty_like(probs.unsqueeze(0))
    q[0].exponential_(generator=expected_generator)
    expected_token = int((probs.unsqueeze(0) / q).argmax(dim=-1).item())

    token_id, q_value = sample_from_processed_logits(
        logits, DSSDReferenceSamplingConfig(temperature=1.0), generator
    )

    assert token_id == expected_token
    assert q_value == probs[expected_token].item()


def test_sample_from_processed_logits_samples_single_row_logits():
    logits = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(31)
    expected_generator = torch.Generator(device="cpu").manual_seed(31)
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    q = torch.empty_like(probs)
    q[0].exponential_(generator=expected_generator)
    expected_token = int((probs / q).argmax(dim=-1).item())

    token_id, q_value = sample_from_processed_logits(
        logits, DSSDReferenceSamplingConfig(temperature=1.0), generator
    )

    assert token_id == expected_token
    assert q_value == probs[0, expected_token].item()


def test_sample_from_processed_logits_greedy_accepts_single_row_logits():
    logits = torch.tensor([[0.0, 3.0, 1.0]], dtype=torch.float32)

    token_id, q_value = sample_from_processed_logits(
        logits, DSSDReferenceSamplingConfig(temperature=0.0), None
    )

    assert token_id == 1
    assert q_value == 1.0


def test_sample_from_processed_logits_rejects_batched_logits():
    logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match=r"shape \[vocab\] or \[1, vocab\]"):
        sample_from_processed_logits(
            logits, DSSDReferenceSamplingConfig(temperature=1.0), None
        )
