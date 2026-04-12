from __future__ import annotations

from vllm.dssd.protocol import (
    CloseSessionAck,
    OpenSessionRequest,
    OpenSessionResponse,
    VerifyRoundRequest,
    VerifyRoundResponse,
)


class DSSDVerifierService:
    def __init__(self, decode_engine) -> None:
        self.decode_engine = decode_engine

    def open_session(self, request: OpenSessionRequest) -> OpenSessionResponse:
        result = self.decode_engine.open_session(
            req_id=request.req_id,
            prompt_token_ids=request.prompt_token_ids,
            sampling_params=request.sampling_params,
            lora_request=request.lora_request,
        )
        return OpenSessionResponse(
            req_id=result.req_id,
            bootstrap_token_id=result.bootstrap_token_id,
        )

    def verify_round(self, request: VerifyRoundRequest) -> VerifyRoundResponse:
        session = self.decode_engine.sessions[request.req_id]
        result = self.decode_engine.verify_round(session, request)
        return VerifyRoundResponse(
            req_id=result.req_id,
            accepted_len=result.accepted_len,
            bonus_token_id=result.bonus_token_id,
            rejected_target_logits=result.rejected_target_logits,
        )

    def close_session(self, req_id: str) -> CloseSessionAck:
        session = self.decode_engine.sessions[req_id]
        self.decode_engine.close_session(session)
        return CloseSessionAck(req_id=req_id)
