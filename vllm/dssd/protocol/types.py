from __future__ import annotations

from dataclasses import dataclass

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


@dataclass
class VerifyRoundResponse:
    req_id: str
    accepted_len: int
    bonus_token_id: int | None = None
    rejected_target_logits: torch.Tensor | None = None

    def __post_init__(self) -> None:
        has_bonus = self.bonus_token_id is not None
        has_logits = self.rejected_target_logits is not None
        if has_bonus == has_logits:
            raise ValueError(
                "VerifyRoundResponse requires exactly one bypass payload"
            )


@dataclass
class CloseSessionRequest:
    req_id: str


@dataclass
class CloseSessionAck:
    req_id: str
