from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.dssd.edge import DSSDEdgeDraftSampler


class FakeStage:
    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.fn(*args)


class FakeSamplingStates:
    def __init__(self):
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
            lambda logits, _expanded_idx_mapping, _idx_mapping_np: logits + 5.0
        )


class FakeSampler:
    def __init__(self):
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
        self.sampling_states = FakeSamplingStates()


def make_input_batch() -> SimpleNamespace:
    return SimpleNamespace(
        positions=torch.tensor([7], dtype=torch.int64),
        logits_indices=torch.tensor([0], dtype=torch.int64),
        input_ids=torch.tensor([101], dtype=torch.int32),
        expanded_idx_mapping=torch.tensor([1], dtype=torch.int32),
        idx_mapping_np=np.array([1], dtype=np.int32),
        expanded_local_pos=torch.tensor([0], dtype=torch.int32),
    )


def test_extract_q_value_uses_processed_logits_probability() -> None:
    edge_sampler = DSSDEdgeDraftSampler(FakeSampler())
    processed_logits = torch.tensor([0.0, torch.log(torch.tensor(2.0))])

    q_value = edge_sampler.extract_q_value(processed_logits, sampled_token_id=1)

    assert q_value == pytest.approx(torch.tensor(2.0 / 3.0).item())


def test_helper_tensor_builders_return_edge_shapes_on_cpu() -> None:
    edge_sampler = DSSDEdgeDraftSampler(FakeSampler())
    device = torch.device("cpu")

    sampled_tokens = edge_sampler.build_sampled_tokens(17, device)
    num_sampled = edge_sampler.build_num_sampled(device)
    num_rejected = edge_sampler.build_num_rejected(device)

    assert sampled_tokens.device.type == "cpu"
    assert sampled_tokens.dtype == torch.int64
    assert sampled_tokens.tolist() == [[17]]
    assert num_sampled.dtype == torch.int32
    assert num_sampled.tolist() == [1]
    assert num_rejected.dtype == torch.int32
    assert num_rejected.tolist() == [0]


def test_apply_sampling_params_into_reuses_destination_buffer() -> None:
    base_sampler = FakeSampler()
    edge_sampler = DSSDEdgeDraftSampler(base_sampler)
    input_batch = make_input_batch()
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float16)
    processed_logits_dst = torch.full((1, 3), -999.0, dtype=torch.float32)

    processed_logits = edge_sampler.apply_sampling_params_into(
        logits,
        input_batch,
        processed_logits_dst,
    )

    assert processed_logits.data_ptr() == processed_logits_dst.data_ptr()
    assert processed_logits.dtype == torch.float32
    assert torch.equal(
        processed_logits,
        torch.tensor([[26.0, 28.0, 30.0]], dtype=torch.float32),
    )
    assert len(base_sampler.logit_bias_state.apply_logit_bias.calls) == 1
    assert len(base_sampler.penalties_state.apply_penalties.calls) == 1
    assert len(base_sampler.bad_words_state.apply_bad_words.calls) == 1
    assert len(base_sampler.sampling_states.apply_temperature.calls) == 1
    assert len(base_sampler.sampling_states.apply_min_p.calls) == 1
    assert len(base_sampler.sampling_states.apply_top_k_top_p.calls) == 1


def test_sample_step_samples_one_token_and_returns_q_value(monkeypatch) -> None:
    base_sampler = FakeSampler()
    edge_sampler = DSSDEdgeDraftSampler(base_sampler)
    input_batch = make_input_batch()
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    processed_logits_dst = torch.empty_like(logits)
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

    monkeypatch.setattr("vllm.dssd.edge.sampler.gumbel_sample", fake_gumbel_sample)

    sampled_token_id, q_value = edge_sampler.sample_step(
        logits,
        input_batch,
        processed_logits_dst,
    )

    expected_processed_logits = torch.tensor([[26.0, 28.0, 30.0]])
    expected_q_value = torch.softmax(expected_processed_logits[0], dim=-1)[2].item()

    assert sampled_token_id == 2
    assert q_value == pytest.approx(expected_q_value)
    assert torch.equal(observed["logits"], expected_processed_logits)
    assert observed["req_idx"].tolist() == [1]
    assert observed["temperature"] is base_sampler.sampling_states.temperature.gpu
    assert observed["seed"] is base_sampler.sampling_states.seeds.gpu
    assert observed["pos"].tolist() == [7]
    assert observed["apply_temperature"] is False
