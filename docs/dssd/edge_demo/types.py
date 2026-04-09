from __future__ import annotations

from dataclasses import dataclass, field

import torch

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams


@dataclass
class EdgeVerifyRequest:
    """edge 发给 verifier 的单轮校验请求。"""

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
class EdgeVerifyResponse:
    """verifier 返回给 edge 的单轮结果。"""

    req_id: str
    accepted_len: int
    next_token_id: int | None = None
    rejected_step: int | None = None
    rejected_target_logits: torch.Tensor | None = None

    def is_all_accepted(self, draft_len: int) -> bool:
        return self.accepted_len == draft_len

    def is_rejected(self, draft_len: int) -> bool:
        return self.accepted_len < draft_len


@dataclass
class EdgeRoundState:
    """只保存当前轮的暂态数据。"""

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
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        need_new_buffer = (
            self.draft_logits_buffer is None
            or self.draft_logits_buffer.shape != (gamma, vocab_size)
            or self.draft_logits_buffer.device != device
            or self.draft_logits_buffer.dtype != dtype
        )
        if need_new_buffer:
            self.draft_logits_buffer = torch.empty(
                (gamma, vocab_size),
                dtype=dtype,
                device=device,
            )

    def logits_row_view(self, index: int) -> torch.Tensor:
        if self.draft_logits_buffer is None:
            raise RuntimeError("round logits buffer 尚未初始化")
        return self.draft_logits_buffer[index : index + 1]

    def append_step(
        self,
        token_id: int,
        q_value: float,
    ) -> None:
        self.draft_token_ids.append(int(token_id))
        self.draft_q_values.append(float(q_value))

    def q_dist_at(self, index: int) -> torch.Tensor:
        if self.draft_logits_buffer is None:
            raise RuntimeError("round logits buffer 尚未初始化")
        return self.draft_logits_buffer[index]


@dataclass
class EdgeSession:
    """协议层视角下的 edge 会话状态。"""

    req_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    block_ids: tuple[list[int], ...]
    prompt_len: int
    num_computed_tokens: int = 0
    total_len: int = 0
    token_ids: list[int] = field(default_factory=list)
    lora_request: LoRARequest | None = None
    round_state: EdgeRoundState = field(default_factory=EdgeRoundState)

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
        kept_len = max(self.prompt_len, len(self.token_ids) - count)
        self.token_ids = self.token_ids[:kept_len]
        self.total_len = kept_len
        self.num_computed_tokens = max(
            self.prompt_len,
            min(self.num_computed_tokens, kept_len),
        )

    def committed_output_ids(self) -> list[int]:
        return self.token_ids[self.prompt_len :]


@dataclass
class EdgeOpenSessionResult:
    """edge 侧 open_session 的返回值。"""

    req_id: str
    bootstrap_token_id: int
    session: EdgeSession | None = None
