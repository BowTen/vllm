# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import msgspec


class BindVerifierRequest(msgspec.Struct, omit_defaults=True):
    protocol_version: str
    edge_instance_id: str
    tokenizer_hash: str
    vocab_hash: str
    supported_gamma_max: int
    auth_payload: dict[str, str] | None = None


class BindVerifierResponse(msgspec.Struct, omit_defaults=True):
    binding_id: str
    protocol_version: str
    verifier_model_id: str
    tokenizer_hash: str
    vocab_hash: str
    supported_gamma_max: int
    capabilities: dict[str, str] | None = None


class CreateSessionRequest(msgspec.Struct, omit_defaults=True):
    binding_id: str
    request_id: str
    prompt_token_ids: list[int]
    sampling_params_digest: str
    max_new_tokens: int | None = None
    stop_token_ids: list[int] | None = None


class CreateSessionResponse(msgspec.Struct, omit_defaults=True):
    verifier_session_id: str
    accepted_prompt_len: int
    expires_at: float | None = None


class VerifyRoundRequest(msgspec.Struct, omit_defaults=True):
    binding_id: str
    verifier_session_id: str
    seq_no: int
    prefix_delta_token_ids: list[int]
    draft_token_ids: list[int]
    q_values: list[float]


class VerifyRoundResponse(msgspec.Struct, omit_defaults=True):
    verifier_session_id: str
    seq_no: int
    accepted_count: int
    all_accepted: bool
    bonus_token_id: int | None = None
    reject_index: int | None = None
    reject_target_probs: list[float] | None = None
    finished: bool = False
    finish_reason: str | None = None


class CloseSessionRequest(msgspec.Struct, omit_defaults=True):
    verifier_session_id: str
    reason: str | None = None


class CloseSessionResponse(msgspec.Struct, omit_defaults=True):
    closed: bool
