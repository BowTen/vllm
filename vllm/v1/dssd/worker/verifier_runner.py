# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch

from vllm.v1.dssd.protocol import (
    DSSDVerifierExecutionRequest,
    VerifierForwardResult,
    VerifyRoundRequest,
)


def extract_forward_probs(
    logits: torch.Tensor,
    metadata,
) -> tuple[list[list[float]], list[float]]:
    target_probs = torch.softmax(
        logits[metadata.target_logits_indices].to(torch.float32),
        dim=-1,
    )
    bonus_probs = torch.softmax(
        logits[metadata.bonus_logits_indices].to(torch.float32),
        dim=-1,
    )
    seq_probs = target_probs.tolist()
    bonus_list = bonus_probs[0].tolist()
    return seq_probs, bonus_list


def build_verifier_result(
    request: VerifyRoundRequest | DSSDVerifierExecutionRequest,
    target_probs: list[list[float]],
    *,
    finished: bool = False,
    finish_reason: str | None = None,
) -> VerifierForwardResult:
    expected_num_probs = len(request.draft_token_ids) + 1
    if len(target_probs) != expected_num_probs:
        raise ValueError("target_probs must include one bonus distribution")
    return VerifierForwardResult(
        verifier_session_id=request.verifier_session_id,
        seq_no=request.seq_no,
        seq_probs=target_probs[:-1],
        bonus_probs=target_probs[-1],
        finished=finished,
        finish_reason=finish_reason,
    )


def build_verifier_result_from_logits(
    *,
    request: VerifyRoundRequest | DSSDVerifierExecutionRequest,
    logits: torch.Tensor,
    metadata,
    finished: bool = False,
    finish_reason: str | None = None,
) -> VerifierForwardResult:
    seq_probs, bonus_probs = extract_forward_probs(logits, metadata)
    return build_verifier_result(
        request=request,
        target_probs=[*seq_probs, bonus_probs],
        finished=finished,
        finish_reason=finish_reason,
    )
