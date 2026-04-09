from __future__ import annotations

from dataclasses import dataclass, field

import torch

from vllm.sampling_params import SamplingParams


@dataclass
class EdgeRoundState:
    draft_token_ids: list[int] = field(default_factory=list)
    draft_q_values: list[float] = field(default_factory=list)
    draft_logits_buffer: torch.Tensor | None = None
    committed_token_id: int | None = None

    def reset(self) -> None:
        self.draft_token_ids.clear()
        self.draft_q_values.clear()
        self.committed_token_id = None

    def prepare_logits_buffer(
        self,
        gamma: int,
        vocab_size: int,
        device: torch.device | str,
        dtype: torch.dtype | None,
    ) -> None:
        actual_dtype = torch.float32 if dtype is None else dtype
        self.draft_logits_buffer = torch.empty(
            (gamma, vocab_size),
            dtype=actual_dtype,
            device=device,
        )

    def logits_row_view(self, index: int) -> torch.Tensor:
        if self.draft_logits_buffer is None:
            raise RuntimeError("round logits buffer is not initialized")
        return self.draft_logits_buffer[index : index + 1]

    def append_step(self, token_id: int, q_value: float) -> None:
        self.draft_token_ids.append(int(token_id))
        self.draft_q_values.append(float(q_value))


@dataclass
class EdgeSession:
    req_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    block_ids: tuple[list[int], ...]
    prompt_len: int
    num_computed_tokens: int = 0
    total_len: int = 0
    token_ids: list[int] = field(default_factory=list)

    @property
    def output_len(self) -> int:
        return max(self.total_len - self.prompt_len, 0)

    def append_token(self, token_id: int, *, computed_delta: int) -> None:
        self.token_ids.append(int(token_id))
        self.total_len += 1
        self.num_computed_tokens += computed_delta

    def rollback(self, count: int) -> None:
        if count <= 0:
            return
        self.token_ids = self.token_ids[: max(0, len(self.token_ids) - count)]
        self.total_len = len(self.token_ids)
        self.num_computed_tokens = max(
            self.prompt_len,
            min(self.num_computed_tokens, self.total_len),
        )

    def committed_output_ids(self) -> list[int]:
        return self.token_ids[self.prompt_len :]


@dataclass
class EdgeOpenSessionResult:
    req_id: str
    bootstrap_token_id: int
