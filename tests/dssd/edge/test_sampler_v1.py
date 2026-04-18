from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.dssd.edge.sampler_v1 import DSSDEdgeDraftSamplerV1


class FakeStage:
    def __init__(self, fn) -> None:
        self.fn = fn
        self.calls = []
        self.returns = []

    def __call__(self, *args):
        self.calls.append(args)
        result = self.fn(*args)
        self.returns.append(result)
        return result


class FakeSamplingStates:
    def __init__(self, *, top_k_returns_new_tensor: bool) -> None:
        self.temperature = SimpleNamespace(
            gpu=torch.tensor([0.0, 1.5, 0.75], dtype=torch.float32)
        )
        self.seeds = SimpleNamespace(
            gpu=torch.tensor([11, 22, 33], dtype=torch.int64)
        )
        self.apply_temperature = FakeStage(
            lambda logits, _expanded_idx_mapping, _idx_mapping_np: logits.mul_(2.0)
        )
        self.apply_min_p = FakeStage(
            lambda logits, _expanded_idx_mapping, _idx_mapping_np: logits.add_(-3.0)
        )
        self.apply_top_k_top_p = FakeStage(
            (
                lambda logits, _expanded_idx_mapping, _idx_mapping_np:
                logits + 5.0
            )
            if top_k_returns_new_tensor
            else (
                lambda logits, _expanded_idx_mapping, _idx_mapping_np:
                logits.add_(5.0)
            )
        )


class FakeSampler:
    def __init__(self, *, top_k_returns_new_tensor: bool) -> None:
        self.num_speculative_tokens = 1
        self.logit_bias_state = SimpleNamespace(
            apply_logit_bias=FakeStage(
                lambda logits, _expanded_idx_mapping, _idx_mapping_np, _pos:
                logits.add_(1.0)
            )
        )
        self.penalties_state = SimpleNamespace(
            apply_penalties=FakeStage(
                lambda logits, _expanded_idx_mapping, _idx_mapping_np, _input_ids,
                _expanded_local_pos, _num_speculative_tokens: logits.add_(10.0)
            )
        )
        self.bad_words_state = SimpleNamespace(
            apply_bad_words=FakeStage(
                lambda logits, _expanded_idx_mapping, _idx_mapping_np, _input_ids,
                _expanded_local_pos: logits
            )
        )
        self.sampling_states = FakeSamplingStates(
            top_k_returns_new_tensor=top_k_returns_new_tensor
        )


def make_input_batch() -> SimpleNamespace:
    return SimpleNamespace(
        positions=torch.tensor([7], dtype=torch.int64),
        logits_indices=torch.tensor([0], dtype=torch.int64),
        input_ids=torch.tensor([101], dtype=torch.int32),
        expanded_idx_mapping=torch.tensor([1], dtype=torch.int32),
        idx_mapping_np=np.array([1], dtype=np.int32),
        expanded_local_pos=torch.tensor([0], dtype=torch.int32),
    )


def _make_sampler(*, top_k_returns_new_tensor: bool) -> DSSDEdgeDraftSamplerV1:
    return DSSDEdgeDraftSamplerV1(
        FakeSampler(top_k_returns_new_tensor=top_k_returns_new_tensor)
    )


def test_sample_step_applies_gpu_sampler_states_and_returns_q_value(
    monkeypatch,
) -> None:
    sampler = _make_sampler(top_k_returns_new_tensor=False)
    input_batch = make_input_batch()
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float16)
    processed_logits_dst = torch.full((1, 3), -999.0, dtype=torch.float32)
    observed = {}

    def fake_gumbel_sample(
        logits_arg,
        req_idx_arg,
        temperature_arg,
        seed_arg,
        pos_arg,
        apply_temperature,
    ):
        observed["logits"] = logits_arg.clone()
        observed["req_idx"] = req_idx_arg.clone()
        observed["temperature"] = temperature_arg
        observed["seed"] = seed_arg
        observed["pos"] = pos_arg.clone()
        observed["apply_temperature"] = apply_temperature
        return torch.tensor([2], dtype=torch.int64)

    monkeypatch.setattr("vllm.dssd.edge.sampler_v1.gumbel_sample", fake_gumbel_sample)

    sampled_token_id, q_value = sampler.sample_step(
        logits,
        input_batch,
        processed_logits_dst,
    )

    expected_processed_logits = torch.tensor([[26.0, 28.0, 30.0]], dtype=torch.float32)

    assert sampled_token_id == 2
    assert q_value == pytest.approx(
        torch.softmax(expected_processed_logits[0], dim=-1)[2].item()
    )
    assert torch.equal(processed_logits_dst, expected_processed_logits)
    assert len(sampler.sampler.logit_bias_state.apply_logit_bias.calls) == 1
    assert len(sampler.sampler.penalties_state.apply_penalties.calls) == 1
    assert len(sampler.sampler.bad_words_state.apply_bad_words.calls) == 1
    assert len(sampler.sampler.sampling_states.apply_temperature.calls) == 1
    assert len(sampler.sampler.sampling_states.apply_min_p.calls) == 1
    assert len(sampler.sampler.sampling_states.apply_top_k_top_p.calls) == 1
    assert torch.equal(observed["logits"], expected_processed_logits[:1])
    assert observed["req_idx"].tolist() == [1]
    assert observed["temperature"] is sampler.sampler.sampling_states.temperature.gpu
    assert observed["seed"] is sampler.sampler.sampling_states.seeds.gpu
    assert observed["pos"].tolist() == [7]
    assert observed["apply_temperature"] is False


def test_sample_step_copies_replaced_processed_logits_back_into_destination_buffer(
    monkeypatch,
) -> None:
    sampler = _make_sampler(top_k_returns_new_tensor=True)
    input_batch = make_input_batch()
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float16)
    processed_logits_dst = torch.full((1, 3), -999.0, dtype=torch.float32)

    monkeypatch.setattr(
        "vllm.dssd.edge.sampler_v1.gumbel_sample",
        lambda *args, **kwargs: torch.tensor([1], dtype=torch.int64),
    )

    sampled_token_id, q_value = sampler.sample_step(
        logits,
        input_batch,
        processed_logits_dst,
    )

    expected_processed_logits = torch.tensor([[26.0, 28.0, 30.0]], dtype=torch.float32)
    returned_logits = sampler.sampler.sampling_states.apply_top_k_top_p.returns[0]

    assert sampled_token_id == 1
    assert q_value == pytest.approx(
        torch.softmax(expected_processed_logits[0], dim=-1)[1].item()
    )
    assert torch.equal(processed_logits_dst, expected_processed_logits)
    assert returned_logits.data_ptr() != processed_logits_dst.data_ptr()
    assert torch.equal(returned_logits, expected_processed_logits)
