from __future__ import annotations

from dataclasses import dataclass, field

import torch

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams


@dataclass
class VerifierRoundRequest:
    req_id: str
    committed_token_id: int
    draft_token_ids: list[int]
    draft_q_values: list[float]

    def validate(self, gamma: int) -> None:
        if len(self.draft_token_ids) != len(self.draft_q_values):
            raise ValueError("draft_token_ids 和 draft_q_values 长度不一致")
        if len(self.draft_token_ids) > gamma:
            raise ValueError("draft 长度超过固定 gamma")


@dataclass
class VerifierRoundResult:
    req_id: str
    accepted_len: int
    bonus_token_id: int | None = None
    rejected_step: int | None = None
    rejected_target_logits: torch.Tensor | None = None

    def is_all_accepted(self) -> bool:
        return self.rejected_step is None

    def is_rejected(self) -> bool:
        return self.rejected_step is not None


@dataclass
class VerifierOpenSessionResult:
    req_id: str
    bootstrap_token_id: int


@dataclass
class VerifierSession:
    req_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    block_ids: tuple[list[int], ...]
    prompt_len: int
    num_computed_tokens: int = 0
    total_len: int = 0
    token_ids: list[int] = field(default_factory=list)
    lora_request: LoRARequest | None = None

    @property
    def output_len(self) -> int:
        return max(self.total_len - self.prompt_len, 0)


@dataclass
class VerifierRoundState:
    committed_token_id: int | None = None
    draft_token_ids: list[int] = field(default_factory=list)
    draft_q_values: list[float] = field(default_factory=list)
    last_result: VerifierRoundResult | None = None

    def reset(self) -> None:
        self.committed_token_id = None
        self.draft_token_ids.clear()
        self.draft_q_values.clear()
        self.last_result = None
