from types import SimpleNamespace

import torch

from experiments.dssd_correctness.vllm_logprobs_backend import VllmLogprobsModel
from vllm.logprobs import Logprob


class _FakeLLM:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []
        self.llm_engine = SimpleNamespace(
            model_config=SimpleNamespace(get_vocab_size=lambda: 4)
        )

    def generate(self, prompts, sampling_params, use_tqdm=False):
        self.calls.append((prompts, sampling_params, use_tqdm))
        return [self.outputs.pop(0)]


def test_from_model_requests_raw_logits_logprobs_mode(monkeypatch):
    captured_kwargs = {}

    class FakeConstructedLLM:
        def __init__(self, *args, **kwargs):
            captured_kwargs.update(kwargs)
            self.llm_engine = SimpleNamespace(
                model_config=SimpleNamespace(get_vocab_size=lambda: 4)
            )

    monkeypatch.setattr("vllm.LLM", FakeConstructedLLM)

    backend = VllmLogprobsModel.from_model(
        "fake-model",
        device="cpu",
        max_model_len=8,
    )

    assert backend.vocab_size == 4
    assert captured_kwargs["logprobs_mode"] == "raw_logits"


def _row(values):
    return {
        token_id: Logprob(logprob=value)
        for token_id, value in enumerate(values)
    }


def test_next_logits_reads_full_sample_logprobs_as_logits_like_row():
    output = SimpleNamespace(
        outputs=[SimpleNamespace(logprobs=[_row([-4.0, -3.0, -2.0, -1.0])])],
        prompt_logprobs=None,
    )
    backend = VllmLogprobsModel(_FakeLLM([output]), device="cpu")

    logits = backend.next_logits([10, 11])

    assert logits.shape == (1, 4)
    torch.testing.assert_close(
        logits,
        torch.tensor([[-4.0, -3.0, -2.0, -1.0]], dtype=torch.float32),
    )
    prompt = backend.llm.calls[0][0][0]
    assert prompt["prompt_token_ids"] == [10, 11]
    assert backend.llm.calls[0][1].logprobs == -1


def test_verify_logits_reads_draft_rows_and_bonus_row():
    output = SimpleNamespace(
        prompt_logprobs=[
            None,
            _row([-1.0, -2.0, -3.0, -4.0]),
            _row([-5.0, -6.0, -7.0, -8.0]),
            _row([-9.0, -10.0, -11.0, -12.0]),
        ],
        outputs=[SimpleNamespace(logprobs=[_row([-13.0, -14.0, -15.0, -16.0])])],
    )
    backend = VllmLogprobsModel(_FakeLLM([output]), device="cpu")

    logits = backend.verify_logits([101, 201, 202, 203], draft_len=2)

    assert logits.shape == (1, 3, 4)
    torch.testing.assert_close(
        logits[0],
        torch.tensor(
            [
                [-5.0, -6.0, -7.0, -8.0],
                [-9.0, -10.0, -11.0, -12.0],
                [-13.0, -14.0, -15.0, -16.0],
            ],
            dtype=torch.float32,
        ),
    )
    assert backend.llm.calls[0][1].prompt_logprobs == -1
