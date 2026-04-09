# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.dssd.verifier.sampler import DSSDVerifierSampler
from vllm.dssd.verifier.types import VerifierRoundRequest


def _build_fake_sampler(device: torch.device) -> SimpleNamespace:
    return SimpleNamespace(
        apply_sampling_params=lambda logits, *_args: logits.to(dtype=torch.float32),
        sampling_states=SimpleNamespace(
            temperature=SimpleNamespace(
                gpu=torch.ones(1, device=device, dtype=torch.float32)
            ),
            seeds=SimpleNamespace(gpu=torch.tensor([1234], device=device)),
        ),
    )


def _build_input_batch(device: torch.device) -> SimpleNamespace:
    return SimpleNamespace(
        expanded_idx_mapping=torch.tensor([0, 0, 0], device=device),
        idx_mapping_np=torch.tensor([0], device="cpu").numpy(),
        positions=torch.tensor([3, 4, 5], device=device, dtype=torch.int64),
        input_ids=torch.tensor([7, 8, 9], device=device, dtype=torch.int64),
        expanded_local_pos=torch.tensor([0, 1, 2], device=device, dtype=torch.int64),
        logits_indices=torch.tensor([0, 1, 2], device=device, dtype=torch.int64),
        seq_lens=torch.tensor([3], device=device, dtype=torch.int32),
        num_reqs=1,
    )


def test_all_accept_keeps_bonus_token_as_bypass_result(monkeypatch) -> None:
    import vllm.dssd.verifier.sampler as sampler_module

    device = torch.device("cpu")
    sampler = DSSDVerifierSampler(
        sampler=_build_fake_sampler(device),
        num_speculative_steps=2,
    )
    input_batch = _build_input_batch(device)
    logits = torch.tensor(
        [[5.0, 1.0, 0.0], [0.0, 5.0, 1.0], [1.0, 1.0, 5.0]],
        device=device,
    )
    request = VerifierRoundRequest(
        req_id="req-1",
        committed_token_id=7,
        draft_token_ids=[0, 1],
        draft_q_values=[0.9, 0.9],
    )

    monkeypatch.setattr(
        sampler_module,
        "find_accepted_len",
        lambda **_kwargs: torch.tensor(2, device=device, dtype=torch.int32),
    )
    monkeypatch.setattr(
        sampler_module,
        "gumbel_sample",
        lambda *_args, **_kwargs: torch.tensor([17], device=device, dtype=torch.int64),
    )

    sampler_output, raw_result = sampler(logits, input_batch, request)
    result = raw_result.to_round_result()

    assert torch.equal(
        raw_result.accepted_len,
        torch.tensor(2, device=device, dtype=torch.int32),
    )
    assert torch.equal(raw_result.all_accepted, torch.tensor(True, device=device))
    assert result.accepted_len == 2
    assert isinstance(result.accepted_len, int)
    assert result.is_all_accepted()
    assert not result.is_rejected()
    assert result.rejected_target_logits is None
    assert result.bonus_token_id == 17
    assert isinstance(result.bonus_token_id, int)
    assert sampler_output.sampled_token_ids.tolist() == [[0, 1, -1]]
    assert torch.equal(
        sampler_output.num_sampled,
        torch.tensor([2], device=device, dtype=torch.int32),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_reject_path_returns_cuda_logits_row() -> None:
    device = torch.device("cuda")
    sampler = DSSDVerifierSampler(
        sampler=_build_fake_sampler(device),
        num_speculative_steps=2,
    )
    input_batch = _build_input_batch(device)
    logits = torch.tensor(
        [[-100.0, 100.0, -100.0], [0.0, 5.0, 1.0], [1.0, 1.0, 5.0]],
        device=device,
    )
    request = VerifierRoundRequest(
        req_id="req-1",
        committed_token_id=7,
        draft_token_ids=[0, 1],
        draft_q_values=[1.0, 1.0],
    )

    sampler_output, raw_result = sampler(logits, input_batch, request)
    result = raw_result.to_round_result()

    assert raw_result.accepted_len.is_cuda
    assert torch.equal(
        raw_result.accepted_len,
        torch.tensor(0, device=device, dtype=torch.int32),
    )
    assert raw_result.all_accepted.is_cuda
    assert torch.equal(raw_result.all_accepted, torch.tensor(False, device=device))
    assert result.accepted_len == 0
    assert isinstance(result.accepted_len, int)
    assert result.bonus_token_id is None
    assert result.is_rejected()
    assert not result.is_all_accepted()
    assert result.rejected_target_logits is not None
    assert result.rejected_target_logits.is_cuda
    assert torch.equal(result.rejected_target_logits, logits[0].to(torch.float32))
    assert sampler_output.sampled_token_ids.tolist() == [[-1, -1, -1]]
    assert torch.equal(
        sampler_output.num_sampled,
        torch.tensor([0], device=device, dtype=torch.int32),
    )


def test_sampler_source_keeps_accept_reject_off_cpu() -> None:
    sampler_source = Path(
        "vllm/dssd/verifier/sampler.py"
    ).read_text(encoding="utf-8")
    ops_source = Path("vllm/dssd/verifier/ops.py").read_text(encoding="utf-8")

    assert ".item()" not in sampler_source
    assert ".item()" not in ops_source
    assert "torch.equal(" not in sampler_source
    assert "torch.equal(" not in ops_source
    assert '.to(device="cpu")' not in sampler_source
    assert '.to(device="cpu")' not in ops_source
    assert ".cpu()" not in sampler_source
    assert ".cpu()" not in ops_source
    assert "torch.rand(" not in sampler_source
    assert "torch.rand(" not in ops_source
    assert "torch.rand_like(" not in sampler_source
    assert "torch.rand_like(" not in ops_source
