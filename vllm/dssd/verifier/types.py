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
    rejected_target_logits: torch.Tensor | None = None

    def __post_init__(self) -> None:
        has_bonus = self.bonus_token_id is not None
        has_reject_logits = self.rejected_target_logits is not None
        if has_bonus == has_reject_logits:
            raise ValueError(
                "VerifierRoundResult requires exactly one bypass payload"
            )

    def is_all_accepted(self) -> bool:
        return self.bonus_token_id is not None

    def is_rejected(self) -> bool:
        return self.rejected_target_logits is not None


@dataclass
class VerifierSamplerRoundResult:
    req_id: str
    accepted_len: torch.Tensor
    all_accepted: torch.Tensor
    bonus_token_id: torch.Tensor
    rejected_target_logits: torch.Tensor

    def to_round_result(self) -> VerifierRoundResult:
        accepted_len = int(self.accepted_len.item())
        if bool(self.all_accepted.item()):
            return VerifierRoundResult(
                req_id=self.req_id,
                accepted_len=accepted_len,
                bonus_token_id=int(self.bonus_token_id.item()),
            )
        return VerifierRoundResult(
            req_id=self.req_id,
            accepted_len=accepted_len,
            rejected_target_logits=self.rejected_target_logits,
        )


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
    committed_token_committed: bool = False
    draft_token_ids: list[int] = field(default_factory=list)
    draft_q_values: list[float] = field(default_factory=list)

    def reset(self) -> None:
        self.committed_token_id = None
        self.committed_token_committed = False
        self.draft_token_ids.clear()
        self.draft_q_values.clear()
