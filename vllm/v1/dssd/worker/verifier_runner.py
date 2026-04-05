# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from vllm.v1.dssd.protocol import VerifierForwardResult, VerifyRoundRequest


def build_verifier_result(
    request: VerifyRoundRequest,
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
