from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams

from .engine import VerifierDecodeEngine
from .types import (
    VerifierOpenSessionResult,
    VerifierRoundRequest,
    VerifierRoundResult,
)


@dataclass
class VerifierOpenSessionRequest:
    """transport 层收到的 open_session 消息。"""

    req_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    lora_request: LoRARequest | None = None


@dataclass
class VerifierCloseSessionRequest:
    """transport 层收到的 close_session 消息。"""

    req_id: str


@dataclass
class VerifierStopRequest:
    """仅用于 demo，方便显式退出 serve_forever。"""


class VerifierServiceTransport(Protocol):
    """最小 transport 抽象。

    这里故意不去碰真实网络或 mock network，只保留服务循环需要的接口。
    """

    def recv(self) -> (
        VerifierOpenSessionRequest
        | VerifierRoundRequest
        | VerifierCloseSessionRequest
        | VerifierStopRequest
    ): ...

    def send_open_session_result(
        self,
        result: VerifierOpenSessionResult,
    ) -> None: ...

    def send_verify_result(self, result: VerifierRoundResult) -> None: ...

    def send_close_session_ack(self, req_id: str) -> None: ...


class DSSDVerifierService:
    """最外层 verifier 服务入口。"""

    def __init__(
        self,
        decode_engine: VerifierDecodeEngine,
        eos_token_id: int,
        gamma: int,
    ) -> None:
        self.decode_engine = decode_engine
        self.eos_token_id = eos_token_id
        if gamma != self.decode_engine.gamma:
            raise ValueError(
                "DSSDVerifierService 的 gamma 必须在初始化时和 "
                "decode_engine.gamma 保持一致"
            )
        self.gamma = self.decode_engine.gamma

    def open_session(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        lora_request: LoRARequest | None = None,
    ) -> VerifierOpenSessionResult:
        return self.decode_engine.open_session(
            req_id=req_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )

    def verify(self, request: VerifierRoundRequest) -> VerifierRoundResult:
        request.validate(self.gamma)
        session = self.decode_engine.sessions[request.req_id]
        return self.decode_engine.verify_round(session, request)

    def close_session(self, req_id: str) -> None:
        session = self.decode_engine.sessions[req_id]
        self.decode_engine.close_session(session)

    def serve_forever(self, transport: VerifierServiceTransport) -> None:
        """把伪代码里的 verify_loop 显式写成服务循环。

        语义上等价于：
        1. 等待 edge 发来 open_session
        2. 返回 bootstrap token
        3. 反复处理 verify round
        4. 收到 close_session 后清理会话
        """

        while True:
            message = transport.recv()

            if isinstance(message, VerifierStopRequest):
                break

            if isinstance(message, VerifierOpenSessionRequest):
                result = self.open_session(
                    req_id=message.req_id,
                    prompt_token_ids=message.prompt_token_ids,
                    sampling_params=message.sampling_params,
                    lora_request=message.lora_request,
                )
                transport.send_open_session_result(result)
                continue

            if isinstance(message, VerifierRoundRequest):
                result = self.verify(message)
                transport.send_verify_result(result)
                continue

            if isinstance(message, VerifierCloseSessionRequest):
                self.close_session(message.req_id)
                transport.send_close_session_ack(message.req_id)
                continue

            raise TypeError(f"不支持的 verifier service message: {type(message)}")

    def _all_accepted(
        self,
        result: VerifierRoundResult,
        draft_len: int,
    ) -> bool:
        return result.accepted_len == draft_len

    def _is_eos(self, token_id: int) -> bool:
        return token_id == self.eos_token_id
