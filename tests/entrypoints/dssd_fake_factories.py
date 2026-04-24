# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass
class _FakeOpenResult:
    req_id: str
    bootstrap_token_id: int


@dataclass
class _FakeVerifyResult:
    req_id: str
    accepted_len: int
    bonus_token_id: int | None = None
    rejected_target_logits: object | None = None


class _FakeVerifierEngine:
    def __init__(self) -> None:
        self.sessions: dict[str, object] = {}

    def open_session(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params,
        lora_request=None,
    ) -> _FakeOpenResult:
        self.sessions[req_id] = object()
        return _FakeOpenResult(req_id=req_id, bootstrap_token_id=17)

    def generate_local(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params,
        lora_request=None,
    ) -> list[int]:
        del prompt_token_ids, sampling_params, lora_request
        self.sessions.pop(req_id, None)
        return [31, 32, 33]

    def verify_round(self, session, request) -> _FakeVerifyResult:
        return _FakeVerifyResult(
            req_id=request.req_id,
            accepted_len=len(request.draft_token_ids),
            bonus_token_id=21,
        )

    def close_session(self, session) -> None:
        return None


class _FailingVerifyVerifierEngine(_FakeVerifierEngine):
    def verify_round(self, session, request) -> _FakeVerifyResult:
        raise RuntimeError("verify failed")


class _RejectingVerifyVerifierEngine(_FakeVerifierEngine):
    def verify_round(self, session, request) -> _FakeVerifyResult:
        import torch

        return _FakeVerifyResult(
            req_id=request.req_id,
            accepted_len=0,
            rejected_target_logits=torch.tensor(
                [0.5, -0.25, 1.25],
                dtype=torch.float32,
            ),
        )


@dataclass
class _FakeRoundState:
    draft_token_ids: list[int] = field(default_factory=list)
    draft_q_values: list[float] = field(default_factory=list)
    draft_logits_rows: list[torch.Tensor] = field(default_factory=list)

    def q_dist_at(self, index: int) -> torch.Tensor:
        return self.draft_logits_rows[index]


@dataclass
class _FakeEdgeSession:
    req_id: str
    prompt_len: int
    token_ids: list[int]
    round_state: _FakeRoundState = field(default_factory=_FakeRoundState)

    def committed_output_ids(self) -> list[int]:
        return self.token_ids[self.prompt_len :]


class _FakeEdgeDecodeEngine:
    def __init__(self) -> None:
        self.sessions: dict[str, _FakeEdgeSession] = {}

    def open_session(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params,
        lora_request=None,
    ) -> _FakeEdgeSession:
        session = _FakeEdgeSession(
            req_id=req_id,
            prompt_len=len(prompt_token_ids),
            token_ids=list(prompt_token_ids),
        )
        self.sessions[req_id] = session
        return session

    def prefill(self, session: _FakeEdgeSession, bootstrap_token_id: int) -> int:
        session.token_ids.append(bootstrap_token_id)
        return bootstrap_token_id

    def draft(
        self,
        session: _FakeEdgeSession,
        first_token_id: int,
        gamma: int,
    ) -> _FakeRoundState:
        import torch

        session.round_state = _FakeRoundState(
            draft_token_ids=[19, 20][:gamma],
            draft_q_values=[0.6, 0.4][:gamma],
            draft_logits_rows=[
                torch.zeros(32, dtype=torch.float32),
                torch.zeros(32, dtype=torch.float32),
            ][:gamma],
        )
        session.token_ids.extend(session.round_state.draft_token_ids)
        return session.round_state

    def rollback(self, session: _FakeEdgeSession, rejected_count: int) -> None:
        if rejected_count > 0:
            session.token_ids = session.token_ids[:-rejected_count]

    def commit_external_token(self, session: _FakeEdgeSession, token_id: int) -> int:
        session.token_ids.append(token_id)
        session.round_state = _FakeRoundState()
        return token_id

    def close_session(self, session: _FakeEdgeSession) -> None:
        self.sessions.pop(session.req_id, None)


class _FakePersistentEdgeService:
    def __init__(self) -> None:
        self.calls = 0

    def generate(
        self,
        *,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params,
        lora_request=None,
    ) -> list[int]:
        self.calls += 1
        return [17, self.calls]


class _BlockingEdgeService:
    def __init__(self) -> None:
        self.calls = 0
        self.started_file = os.environ["DSSD_EDGE_STARTED_FILE"]
        self.release_file = os.environ["DSSD_EDGE_RELEASE_FILE"]

    def generate(
        self,
        *,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params,
        lora_request=None,
    ) -> list[int]:
        self.calls += 1
        if self.calls == 1:
            with open(self.started_file, "w", encoding="utf-8") as handle:
                handle.write("started")
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if os.path.exists(self.release_file):
                    break
                time.sleep(0.05)
            else:
                raise RuntimeError("timed out waiting for release file")
            return [9001]
        return [9002]


class _FailingEdgeService:
    def generate(
        self,
        *,
        req_id: str,
        prompt_token_ids: list[int],
        sampling_params,
        lora_request=None,
    ) -> list[int]:
        raise RuntimeError("edge generate failed")


def build_verifier_service(_args):
    from vllm.dssd.service import DSSDVerifierService

    return DSSDVerifierService(decode_engine=_FakeVerifierEngine())


def build_failing_verify_verifier_service(_args):
    from vllm.dssd.service import DSSDVerifierService

    return DSSDVerifierService(decode_engine=_FailingVerifyVerifierEngine())


def build_rejecting_verify_verifier_service(_args):
    from vllm.dssd.service import DSSDVerifierService

    return DSSDVerifierService(decode_engine=_RejectingVerifyVerifierEngine())


def build_edge_service(args):
    from vllm.dssd.service import DSSDEdgeService
    from vllm.dssd.transport.http_verifier_transport import HTTPVerifierTransport

    return DSSDEdgeService(
        decode_engine=_FakeEdgeDecodeEngine(),
        verifier=HTTPVerifierTransport(server_url=args.verifier_url),
        eos_token_id=args.eos_token_id,
        gamma=args.gamma,
    )


def build_persistent_edge_service(_args):
    return _FakePersistentEdgeService()


def build_blocking_edge_service(_args):
    return _BlockingEdgeService()


def build_failing_edge_service(_args):
    return _FailingEdgeService()
