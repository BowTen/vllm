from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import SamplingParams


@dataclass
class OpenSessionRequest:
    req_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    lora_request: LoRARequest | None = None


@dataclass
class OpenSessionResponse:
    req_id: str
    bootstrap_token_id: int


@dataclass
class VerifyRoundRequest:
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
class VerifyRoundResponse:
    req_id: str
    accepted_len: int
    bonus_token_id: int | None = None
    rejected_token_id: int | None = None
    rejected_target_logits: torch.Tensor | None = None

    def __post_init__(self) -> None:
        payload_count = sum(
            payload is not None
            for payload in (
                self.bonus_token_id,
                self.rejected_token_id,
                self.rejected_target_logits,
            )
        )
        if payload_count != 1:
            raise ValueError(
                "VerifyRoundResponse requires exactly one bypass payload"
            )


@dataclass
class CloseSessionRequest:
    req_id: str


@dataclass
class CloseSessionAck:
    req_id: str
