# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import socket
import time
from threading import Thread
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import torch
import uvicorn

from vllm import SamplingParams
from vllm.entrypoints.spec_decode import verifier_server as verifier_server_mod
from vllm.v1.engine import EngineCoreRequest, FinishReason
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.spec_decode.distributed import edge_core_client as edge_client_mod
from vllm.v1.spec_decode.distributed.protocol import (
    DraftProposal,
    OpenSessionResponse,
    ResyncSessionResponse,
    VerificationResult,
)
from vllm.v1.spec_decode.distributed.runtime import (
    DraftProposalOutput,
    serialize_probs,
)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_request(
    request_id: str = "request-1",
    max_tokens: int = 5,
) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        external_req_id=f"{request_id}-external",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0.0, seed=7),
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


class FakeSpeculativeConfig:
    def __init__(self, verifier_url: str, verifier_timeout_s: float = 5.0):
        self.verifier_url = verifier_url
        self.verifier_timeout_s = verifier_timeout_s

    def uses_distributed_draft_model(self) -> bool:
        return True


def make_fake_vllm_config(base_url: str) -> SimpleNamespace:
    return SimpleNamespace(
        speculative_config=FakeSpeculativeConfig(base_url),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_external_lb=False,
        ),
    )


class FakeDraftRunner:
    vocab_size = 128

    def __init__(self, _vllm_config: Any) -> None:
        pass

    async def propose(
        self,
        session_id: str,
        proposal_id: int,
        base_version: int,
        accepted_prefix_token_ids: list[int],
        prompt_len: int,
        sampling: Any,
        generator: torch.Generator,
    ) -> DraftProposalOutput:
        del sampling, generator
        proposal_tokens = {
            0: ([11, 12], False),
            1: ([13], False),
            2: ([88], True),
        }
        draft_token_ids, draft_stopped = proposal_tokens[proposal_id]
        return DraftProposalOutput(
            proposal=DraftProposal(
                session_id=session_id,
                proposal_id=proposal_id,
                base_version=base_version,
                accepted_prefix_len=len(accepted_prefix_token_ids),
                draft_token_ids=draft_token_ids,
                draft_token_probs=[1.0] * len(draft_token_ids),
                draft_stopped=draft_stopped,
            ),
            stopped=draft_stopped,
        )


def _make_fake_target_runner(call_log: list[tuple[str, Any]]) -> type:
    class FakeTargetVerificationRunner:
        def __init__(
            self,
            model_name: str,
            device: str | None,
            dtype: str,
            trust_remote_code: bool,
        ) -> None:
            del model_name, device, dtype, trust_remote_code
            self.sessions: dict[str, list[int]] = {}
            self.vocab_size = FakeDraftRunner.vocab_size

        async def open_session(self, request) -> OpenSessionResponse:
            self.sessions[request.session_id] = list(request.prompt_token_ids)
            call_log.append(("open_session", request.session_id))
            return OpenSessionResponse(
                session_id=request.session_id,
                session_version=request.initial_version,
                vocab_size=self.vocab_size,
            )

        async def close_session(self, request) -> None:
            self.sessions.pop(request.session_id, None)
            call_log.append(("close_session", request.session_id))

        async def resync_session(self, request) -> ResyncSessionResponse:
            self.sessions[request.session_id] = list(request.accepted_prefix_token_ids)
            call_log.append(
                ("resync_session", request.session_id, list(request.accepted_prefix_token_ids))
            )
            return ResyncSessionResponse(
                session_id=request.session_id,
                session_version=request.edge_version,
            )

        async def verify_proposal(self, proposal: DraftProposal) -> VerificationResult:
            call_log.append(
                ("verify_proposal", proposal.proposal_id, list(proposal.draft_token_ids))
            )
            if proposal.proposal_id == 0:
                self.sessions[proposal.session_id].extend(proposal.draft_token_ids + [99])
                return VerificationResult(
                    session_id=proposal.session_id,
                    proposal_id=proposal.proposal_id,
                    base_version=proposal.base_version,
                    accepted_len=len(proposal.draft_token_ids),
                    accepted_token_ids=list(proposal.draft_token_ids),
                    verifier_version=proposal.base_version + len(proposal.draft_token_ids) + 1,
                    bonus_token_id=99,
                )

            if proposal.proposal_id == 1:
                probs = torch.zeros(self.vocab_size, dtype=torch.float32)
                probs[77] = 1.0
                return VerificationResult(
                    session_id=proposal.session_id,
                    proposal_id=proposal.proposal_id,
                    base_version=proposal.base_version,
                    accepted_len=0,
                    accepted_token_ids=[],
                    verifier_version=proposal.base_version,
                    reject_pos=0,
                    target_probs_at_reject_pos=serialize_probs(probs),
                )

            self.sessions[proposal.session_id].extend(proposal.draft_token_ids)
            return VerificationResult(
                session_id=proposal.session_id,
                proposal_id=proposal.proposal_id,
                base_version=proposal.base_version,
                accepted_len=len(proposal.draft_token_ids),
                accepted_token_ids=list(proposal.draft_token_ids),
                verifier_version=proposal.base_version + len(proposal.draft_token_ids),
            )

    return FakeTargetVerificationRunner


class VerifierServerHarness:
    def __init__(self, app, port: int):
        self.port = port
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        )
        self._thread = Thread(target=self._server.run, daemon=True)

    def start(self) -> None:
        self._thread.start()
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                response = httpx.get(f"http://127.0.0.1:{self.port}/health", timeout=0.2)
                if response.status_code == 200:
                    return
            except Exception:
                time.sleep(0.05)
        raise RuntimeError("Verifier server failed to start in time.")

    def stop(self) -> None:
        self._server.should_exit = True
        self._thread.join(timeout=10)


@pytest.fixture
def verifier_server(monkeypatch: pytest.MonkeyPatch):
    call_log: list[tuple[str, Any]] = []
    monkeypatch.setattr(
        verifier_server_mod,
        "TargetVerificationRunner",
        _make_fake_target_runner(call_log),
    )
    port = find_free_port()
    app = verifier_server_mod.build_app(
        verifier_server_mod.VerifierServerArgs(
            model="fake-model",
            host="127.0.0.1",
            port=port,
        )
    )
    harness = VerifierServerHarness(app, port)
    harness.start()
    try:
        yield f"http://127.0.0.1:{port}", call_log
    finally:
        harness.stop()


def _collect_outputs_sync(client: EngineCoreClient) -> tuple[list[int], FinishReason | None]:
    tokens: list[int] = []
    finish_reason = None
    while finish_reason is None:
        outputs = client.get_output()
        assert len(outputs.outputs) == 1
        output = outputs.outputs[0]
        tokens.extend(output.new_token_ids)
        finish_reason = output.finish_reason
    return tokens, finish_reason


async def _collect_outputs_async(
    client: EngineCoreClient,
) -> tuple[list[int], FinishReason | None]:
    tokens: list[int] = []
    finish_reason = None
    while finish_reason is None:
        outputs = await asyncio.wait_for(client.get_output_async(), timeout=5)
        assert len(outputs.outputs) == 1
        output = outputs.outputs[0]
        tokens.extend(output.new_token_ids)
        finish_reason = output.finish_reason
    return tokens, finish_reason


def _assert_common_call_log(call_log: list[tuple[str, Any]]) -> None:
    call_names = [item[0] for item in call_log]
    assert call_names == [
        "open_session",
        "verify_proposal",
        "verify_proposal",
        "resync_session",
        "verify_proposal",
        "close_session",
    ]
    resync_call = call_log[3]
    assert resync_call[2][-1] == 77


def test_distributed_client_supports_sync_generate_path(
    monkeypatch: pytest.MonkeyPatch,
    verifier_server,
) -> None:
    base_url, call_log = verifier_server
    monkeypatch.setattr(edge_client_mod, "EdgeDraftRunner", FakeDraftRunner)
    client = EngineCoreClient.make_client(
        multiprocess_mode=True,
        asyncio_mode=False,
        vllm_config=make_fake_vllm_config(base_url),
        executor_class=object,
        log_stats=False,
    )
    try:
        client.add_request(make_request(max_tokens=5))
        tokens, finish_reason = _collect_outputs_sync(client)
    finally:
        client.shutdown()

    assert tokens == [11, 12, 99, 77, 88]
    assert finish_reason == FinishReason.LENGTH
    _assert_common_call_log(call_log)


@pytest.mark.asyncio
async def test_distributed_client_supports_async_generate_path(
    monkeypatch: pytest.MonkeyPatch,
    verifier_server,
) -> None:
    base_url, call_log = verifier_server
    monkeypatch.setattr(edge_client_mod, "EdgeDraftRunner", FakeDraftRunner)
    client = EngineCoreClient.make_client(
        multiprocess_mode=True,
        asyncio_mode=True,
        vllm_config=make_fake_vllm_config(base_url),
        executor_class=object,
        log_stats=False,
    )
    try:
        await client.add_request_async(make_request(max_tokens=5))
        tokens, finish_reason = await _collect_outputs_async(client)
    finally:
        await client.shutdown_async()  # type: ignore[attr-defined]

    assert tokens == [11, 12, 99, 77, 88]
    assert finish_reason == FinishReason.LENGTH
    _assert_common_call_log(call_log)
