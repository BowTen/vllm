from types import SimpleNamespace

import torch

from experiments.dssd_correctness.reference import generate_dssd_reference
from experiments.dssd_correctness.sampling import DSSDReferenceSamplingConfig


class _FakeTokenizer:
    eos_token_id = None

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(str(token_id) for token_id in token_ids)


class _FakeModel:
    device = torch.device("cpu")

    def __init__(self, token_batches, eos_token_id=None):
        self.token_batches = list(token_batches)
        self.generation_config = SimpleNamespace(eos_token_id=eos_token_id)
        self._parameter = torch.nn.Parameter(torch.empty(0))

    def parameters(self):
        return iter([self._parameter])

    def __call__(self, input_ids):
        token_batch = self.token_batches.pop(0)
        logits = torch.full(
            (1, input_ids.shape[1], 16),
            -100.0,
            dtype=torch.float32,
            device=input_ids.device,
        )
        start = input_ids.shape[1] - len(token_batch)
        for index, token_spec in enumerate(token_batch, start=start):
            if isinstance(token_spec, dict):
                for token_id, score in token_spec.items():
                    logits[0, index, token_id] = float(score)
            else:
                logits[0, index, token_spec] = 100.0
        return SimpleNamespace(logits=logits)


def test_generate_dssd_reference_truncates_round_at_generation_config_eos():
    target_model = _FakeModel(
        token_batches=[
            [2],
            [3, 4, 5, 6],
        ],
        eos_token_id=4,
    )
    draft_model = _FakeModel(token_batches=[[3], [4], [5]])

    output = generate_dssd_reference(
        case_id="case",
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=_FakeTokenizer(),
        prompt_token_ids=[1],
        config=DSSDReferenceSamplingConfig(
            max_tokens=8,
            gamma=3,
            temperature=0.0,
            ignore_eos=False,
        ),
        seed=0,
    )

    assert output.output_token_ids == [2, 3, 4]
    assert output.rounds[-1].committed_token_ids == [3, 4]
    assert output.rounds[-1].accepted_len == 2


def test_generate_dssd_reference_greedy_uses_raw_logits_before_top_k():
    target_model = _FakeModel(
        token_batches=[
            [{2: 100.0, 8: 100.0}],
            [{3: 100.0, 9: 100.0}, {3: 100.0, 10: 100.0}, 4],
        ],
    )
    draft_model = _FakeModel(
        token_batches=[
            [{3: 100.0, 9: 100.0}],
            [{3: 100.0, 10: 100.0}],
        ]
    )

    output = generate_dssd_reference(
        case_id="case",
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=_FakeTokenizer(),
        prompt_token_ids=[1],
        config=DSSDReferenceSamplingConfig(
            max_tokens=3,
            gamma=2,
            temperature=0.0,
            top_p=0.01,
            ignore_eos=True,
        ),
        seed=0,
    )

    assert output.output_token_ids == [2, 3, 3]
    assert output.rounds[-1].accepted_len == 2
