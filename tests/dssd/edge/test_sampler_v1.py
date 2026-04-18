from types import SimpleNamespace

import pytest
import torch

from vllm.dssd.edge.sampler_v1 import DSSDEdgeDraftSamplerV1


class FakeSampler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object, object]] = []

    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata,
    ) -> torch.Tensor:
        self.calls.append(("apply_logits_processors", logits, sampling_metadata))
        logits.add_(torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32))
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata,
    ) -> tuple[torch.Tensor, None]:
        self.calls.append(("sample", logits, sampling_metadata))
        return torch.tensor([2], dtype=torch.int64), None


def test_sample_step_applies_processors_samples_and_returns_q_value() -> None:
    sampler = DSSDEdgeDraftSamplerV1(FakeSampler())
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float16)
    processed_logits_dst = torch.full((1, 3), -999.0, dtype=torch.float32)
    sampling_metadata = SimpleNamespace(tag="metadata")

    sampled_token_id, q_value = sampler.sample_step(
        logits,
        sampling_metadata,
        processed_logits_dst,
    )

    expected_processed_logits = torch.tensor([[1.0, 3.0, 5.0]], dtype=torch.float32)

    assert sampled_token_id == 2
    assert q_value == pytest.approx(
        torch.softmax(expected_processed_logits[0], dim=-1)[2].item()
    )
    assert torch.equal(processed_logits_dst, expected_processed_logits)
    assert sampler.sampler.calls[0][0] == "apply_logits_processors"
    assert sampler.sampler.calls[1][0] == "sample"
    assert sampler.sampler.calls[0][1].data_ptr() == processed_logits_dst.data_ptr()
    assert sampler.sampler.calls[1][1].data_ptr() == processed_logits_dst.data_ptr()
    assert sampler.sampler.calls[0][2] is sampling_metadata
    assert sampler.sampler.calls[1][2] is sampling_metadata
    assert sampler.sampler.calls[0][1].dtype == torch.float32
