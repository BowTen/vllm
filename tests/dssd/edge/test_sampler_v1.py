from types import SimpleNamespace

import pytest
import torch

from vllm.dssd.edge.sampler_v1 import DSSDEdgeDraftSamplerV1
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p


class FakeArgmaxInvariantProcessor:
    def __init__(self, delta: torch.Tensor):
        self.delta = delta
        self.calls = []

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        self.calls.append((logits.data_ptr(), logits.dtype))
        return logits + self.delta


class FakeOldSampler:
    def __init__(self, *, replacement_tensor: bool):
        self.replacement_tensor = replacement_tensor
        self.apply_calls = []

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata,
        predict_bonus_token: bool,
    ) -> torch.Tensor | None:
        self.apply_calls.append(
            (
                logits.data_ptr(),
                logits.dtype,
                sampling_metadata,
                predict_bonus_token,
            )
        )
        logits.add_(10.0)
        if not self.replacement_tensor:
            return None
        return logits + 5.0

    @staticmethod
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1).view(-1)

    @staticmethod
    def apply_temperature(
        logits: torch.Tensor,
        temp: torch.Tensor,
        all_random: bool,
    ) -> torch.Tensor:
        del all_random
        return logits.div_(temp.unsqueeze(dim=1))

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata,
        logprobs_mode_override=None,
    ) -> torch.Tensor:
        del logprobs_mode_override
        final_logits = logits.clone()
        if not sampling_metadata.all_greedy:
            final_logits = self.apply_temperature(
                final_logits,
                sampling_metadata.temperature,
                sampling_metadata.all_random,
            )
            for processor in sampling_metadata.logitsprocs.argmax_invariant:
                final_logits = processor.apply(final_logits)
            final_logits = apply_top_k_top_p(
                final_logits,
                sampling_metadata.top_k,
                sampling_metadata.top_p,
            )
        return final_logits.argmax(dim=-1).to(torch.int32)


def make_sampling_metadata(
    *,
    temperature: float,
    all_greedy: bool,
    all_random: bool,
    top_k: int | None = None,
    top_p: float | None = None,
    argmax_invariant: list | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        temperature=torch.tensor([temperature], dtype=torch.float32),
        all_greedy=all_greedy,
        all_random=all_random,
        top_k=None if top_k is None else torch.tensor([top_k], dtype=torch.int32),
        top_p=None if top_p is None else torch.tensor([top_p], dtype=torch.float32),
        generators={},
        logitsprocs=SimpleNamespace(
            argmax_invariant=[] if argmax_invariant is None else argmax_invariant
        ),
        tag="sampling-metadata",
    )


def test_sample_step_reuses_destination_buffer_for_processed_logits() -> None:
    old_sampler = FakeOldSampler(replacement_tensor=False)
    sampler = DSSDEdgeDraftSamplerV1(old_sampler)
    sampling_metadata = make_sampling_metadata(
        temperature=1.0,
        all_greedy=True,
        all_random=False,
    )
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float16)
    processed_logits_dst = torch.full((1, 3), -999.0, dtype=torch.float32)

    token_id, q_value = sampler.sample_step(
        logits,
        sampling_metadata,
        processed_logits_dst,
    )

    expected_processed_logits = torch.tensor(
        [[11.0, 12.0, 13.0]],
        dtype=torch.float32,
    )

    assert token_id == 2
    assert q_value == pytest.approx(
        torch.softmax(expected_processed_logits[0], dim=-1)[2].item()
    )
    assert torch.equal(processed_logits_dst, expected_processed_logits)
    assert old_sampler.apply_calls == [
        (
            processed_logits_dst.data_ptr(),
            torch.float32,
            sampling_metadata,
            False,
        )
    ]


def test_sample_step_copies_replacement_logits_back_into_reused_buffer() -> None:
    old_sampler = FakeOldSampler(replacement_tensor=True)
    sampler = DSSDEdgeDraftSamplerV1(old_sampler)
    sampling_metadata = make_sampling_metadata(
        temperature=1.0,
        all_greedy=True,
        all_random=False,
    )
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.bfloat16)
    processed_logits_dst = torch.full((1, 3), -999.0, dtype=torch.float32)

    token_id, q_value = sampler.sample_step(
        logits,
        sampling_metadata,
        processed_logits_dst,
    )

    expected_processed_logits = torch.tensor(
        [[16.0, 17.0, 18.0]],
        dtype=torch.float32,
    )

    assert token_id == 2
    assert q_value == pytest.approx(
        torch.softmax(expected_processed_logits[0], dim=-1)[2].item()
    )
    assert torch.equal(processed_logits_dst, expected_processed_logits)
    assert old_sampler.apply_calls == [
        (
            processed_logits_dst.data_ptr(),
            torch.float32,
            sampling_metadata,
            False,
        )
    ]


def test_sample_step_tracks_final_sample_logits_for_q_value() -> None:
    old_sampler = FakeOldSampler(replacement_tensor=False)
    sampler = DSSDEdgeDraftSamplerV1(old_sampler)
    processor = FakeArgmaxInvariantProcessor(
        torch.tensor([[0.0, 2.0, 0.0]], dtype=torch.float32)
    )
    sampling_metadata = make_sampling_metadata(
        temperature=2.0,
        all_greedy=False,
        all_random=True,
        top_k=1,
        argmax_invariant=[processor],
    )
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float16)
    processed_logits_dst = torch.full((1, 3), -999.0, dtype=torch.float32)

    token_id, q_value = sampler.sample_step(
        logits,
        sampling_metadata,
        processed_logits_dst,
    )

    expected_processed_logits = torch.tensor(
        [[-float("inf"), 8.0, -float("inf")]],
        dtype=torch.float32,
    )

    assert token_id == 1
    assert torch.equal(processed_logits_dst, expected_processed_logits)
    assert q_value == pytest.approx(
        torch.softmax(expected_processed_logits[0], dim=-1)[1].item()
    )
