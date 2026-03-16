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
from vllm.sampling_params import StructuredOutputsParams
from vllm.entrypoints.spec_decode import verifier_server as verifier_server_mod
from vllm.v1.engine import EngineCoreRequest, FinishReason
from vllm.v1.engine.core_client import EngineCoreClient
from vllm.v1.spec_decode.distributed import edge_core_client as edge_client_mod
from vllm.v1.spec_decode.distributed.errors import VerifierSessionMissingError
from vllm.v1.spec_decode.distributed.logprobs import pack_sample_logprobs
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
from vllm.v1.spec_decode.distributed.structured_output import StructuredOutputSession


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_request(
    request_id: str = "request-1",
    max_tokens: int = 5,
    logprobs: int | None = None,
    prompt_logprobs: int | None = None,
    structured_outputs: StructuredOutputsParams | None = None,
) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=request_id,
        external_req_id=f"{request_id}-external",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            seed=7,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            structured_outputs=structured_outputs,
        ),
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


class FakeSpeculativeConfig:
    def __init__(
        self,
        verifier_url: str,
        verifier_timeout_s: float = 5.0,
        num_speculative_tokens: int = 2,
        draft_model_config: Any | None = None,
        distributed_runtime_backend: str = "hf",
    ):
        self.verifier_url = verifier_url
        self.verifier_timeout_s = verifier_timeout_s
        self.num_speculative_tokens = num_speculative_tokens
        self.draft_model_config = draft_model_config
        self.distributed_runtime_backend = distributed_runtime_backend

    def uses_distributed_draft_model(self) -> bool:
        return True


def make_fake_vllm_config(
    base_url: str,
    *,
    with_draft_model: bool = False,
    structured_outputs_backend: str | None = None,
) -> SimpleNamespace:
    draft_model_config = None
    if with_draft_model:
        draft_model_config = SimpleNamespace(
            model="fake-model",
            trust_remote_code=False,
        )
    return SimpleNamespace(
        speculative_config=FakeSpeculativeConfig(
            base_url,
            draft_model_config=draft_model_config,
        ),
        structured_outputs_config=(
            SimpleNamespace(
                backend=structured_outputs_backend,
                disable_any_whitespace=False,
                disable_additional_properties=False,
            )
            if structured_outputs_backend is not None
            else None
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_external_lb=False,
        ),
    )


class FakeDraftRunner:
    vocab_size = 128

    def __init__(self, _vllm_config: Any) -> None:
        self.closed_sessions: list[str] = []
        self.structured_output_sessions: list[Any | None] = []

    async def propose(
        self,
        session_id: str,
        proposal_id: int,
        base_version: int,
        accepted_prefix_token_ids: list[int],
        prompt_len: int,
        sampling: Any,
        generator: torch.Generator,
        structured_output_session: Any | None = None,
    ) -> DraftProposalOutput:
        del sampling, generator
        self.structured_output_sessions.append(structured_output_session)
        proposal_tokens = {
            0: ([11, 12], False),
            1: ([13], False),
            2: ([88], True),
        }
        draft_token_ids, draft_stopped = proposal_tokens[proposal_id]
        draft_token_distributions = []
        for token_id in draft_token_ids:
            probs = torch.zeros(self.vocab_size, dtype=torch.float32)
            probs[token_id] = 1.0
            draft_token_distributions.append(probs)
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
            draft_token_distributions=draft_token_distributions,
        )

    def close_session(self, session_id: str) -> None:
        self.closed_sessions.append(session_id)

    def clear_sessions(self) -> None:
        self.closed_sessions.clear()


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
            call_log.append(
                (
                    "open_session",
                    request.session_id,
                    request.sampling_metadata.structured_output_backend,
                )
            )
            prompt_logprobs = []
            if request.sampling_metadata.prompt_logprobs is not None:
                probs_2 = torch.zeros(self.vocab_size, dtype=torch.float32)
                probs_2[2] = 1.0
                probs_3 = torch.zeros(self.vocab_size, dtype=torch.float32)
                probs_3[3] = 1.0
                packed_2 = pack_sample_logprobs(
                    probs_2, 2, request.sampling_metadata.prompt_logprobs
                )
                packed_3 = pack_sample_logprobs(
                    probs_3, 3, request.sampling_metadata.prompt_logprobs
                )
                assert packed_2 is not None
                assert packed_3 is not None
                prompt_logprobs = [packed_2, packed_3]
            return OpenSessionResponse(
                session_id=request.session_id,
                session_version=request.initial_version,
                vocab_size=self.vocab_size,
                prompt_logprobs=prompt_logprobs,
            )

        async def close_session(self, request) -> None:
            self.sessions.pop(request.session_id, None)
            call_log.append(("close_session", request.session_id))

        async def resync_session(self, request) -> ResyncSessionResponse:
            self.sessions[request.session_id] = list(request.accepted_prefix_token_ids)
            call_log.append(
                (
                    "resync_session",
                    request.session_id,
                    list(request.accepted_prefix_token_ids),
                    request.prompt_len,
                    request.sampling_metadata is not None,
                )
            )
            return ResyncSessionResponse(
                session_id=request.session_id,
                session_version=request.edge_version,
            )

        async def verify_proposal(self, proposal: DraftProposal) -> VerificationResult:
            if (
                proposal.session_id == "request-session-loss"
                and proposal.proposal_id == 0
                and not getattr(self, "_missing_once_done", False)
            ):
                self._missing_once_done = True
                call_log.append(("verify_missing_session", proposal.proposal_id))
                raise VerifierSessionMissingError("Unknown verifier session")
            call_log.append(
                ("verify_proposal", proposal.proposal_id, list(proposal.draft_token_ids))
            )
            if proposal.proposal_id == 0:
                probs_11 = torch.zeros(self.vocab_size, dtype=torch.float32)
                probs_11[11] = 1.0
                probs_12 = torch.zeros(self.vocab_size, dtype=torch.float32)
                probs_12[12] = 1.0
                probs_99 = torch.zeros(self.vocab_size, dtype=torch.float32)
                probs_99[99] = 1.0
                packed_11 = pack_sample_logprobs(probs_11, 11, 1)
                packed_12 = pack_sample_logprobs(probs_12, 12, 1)
                packed_99 = pack_sample_logprobs(probs_99, 99, 1)
                assert packed_11 is not None
                assert packed_12 is not None
                assert packed_99 is not None
                self.sessions[proposal.session_id].extend(proposal.draft_token_ids + [99])
                return VerificationResult(
                    session_id=proposal.session_id,
                    proposal_id=proposal.proposal_id,
                    base_version=proposal.base_version,
                    accepted_len=len(proposal.draft_token_ids),
                    accepted_token_ids=list(proposal.draft_token_ids),
                    verifier_version=proposal.base_version + len(proposal.draft_token_ids) + 1,
                    bonus_token_id=99,
                    accepted_logprobs=[packed_11, packed_12],
                    bonus_logprobs=packed_99,
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

            probs_88 = torch.zeros(self.vocab_size, dtype=torch.float32)
            probs_88[88] = 1.0
            packed_88 = pack_sample_logprobs(probs_88, 88, 1)
            assert packed_88 is not None
            self.sessions[proposal.session_id].extend(proposal.draft_token_ids)
            return VerificationResult(
                session_id=proposal.session_id,
                proposal_id=proposal.proposal_id,
                base_version=proposal.base_version,
                accepted_len=len(proposal.draft_token_ids),
                accepted_token_ids=list(proposal.draft_token_ids),
                verifier_version=proposal.base_version + len(proposal.draft_token_ids),
                accepted_logprobs=[packed_88],
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
            runtime_backend="hf",
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


def _collect_outputs_sync_detailed(client: EngineCoreClient):
    collected = []
    finish_reason = None
    while finish_reason is None:
        outputs = client.get_output()
        assert len(outputs.outputs) == 1
        output = outputs.outputs[0]
        collected.append(output)
        finish_reason = output.finish_reason
    return collected


def _collect_outputs_sync_multi(
    client: EngineCoreClient,
    request_ids: list[str],
) -> tuple[dict[str, list[int]], dict[str, FinishReason | None]]:
    tokens_by_request = {request_id: [] for request_id in request_ids}
    finish_by_request: dict[str, FinishReason | None] = {
        request_id: None for request_id in request_ids
    }
    unfinished = set(request_ids)
    while unfinished:
        outputs = client.get_output()
        assert len(outputs.outputs) == 1
        output = outputs.outputs[0]
        tokens_by_request[output.request_id].extend(output.new_token_ids)
        if output.finish_reason is not None:
            finish_by_request[output.request_id] = output.finish_reason
            unfinished.discard(output.request_id)
    return tokens_by_request, finish_by_request


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
    assert resync_call[3] == 3
    assert resync_call[4] is True


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


def test_distributed_client_supports_multiple_sync_requests(
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
    request_ids = ["request-a", "request-b"]
    try:
        for request_id in request_ids:
            client.add_request(make_request(request_id=request_id, max_tokens=5))
        tokens_by_request, finish_by_request = _collect_outputs_sync_multi(
            client,
            request_ids,
        )
    finally:
        client.shutdown()

    assert tokens_by_request == {
        "request-a": [11, 12, 99, 77, 88],
        "request-b": [11, 12, 99, 77, 88],
    }
    assert finish_by_request == {
        "request-a": FinishReason.LENGTH,
        "request-b": FinishReason.LENGTH,
    }
    assert [name for name, *_ in call_log].count("open_session") == 2
    assert [name for name, *_ in call_log].count("verify_proposal") == 6
    assert [name for name, *_ in call_log].count("resync_session") == 2
    assert [name for name, *_ in call_log].count("close_session") == 2


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


def test_distributed_client_streams_logprobs_from_verifier(
    monkeypatch: pytest.MonkeyPatch,
    verifier_server,
) -> None:
    base_url, _call_log = verifier_server
    monkeypatch.setattr(edge_client_mod, "EdgeDraftRunner", FakeDraftRunner)
    client = EngineCoreClient.make_client(
        multiprocess_mode=True,
        asyncio_mode=False,
        vllm_config=make_fake_vllm_config(base_url),
        executor_class=object,
        log_stats=False,
    )
    try:
        client.add_request(make_request(request_id="request-logprobs", logprobs=1))
        outputs = _collect_outputs_sync_detailed(client)
    finally:
        client.shutdown()

    assert [output.new_token_ids for output in outputs] == [[11, 12, 99], [77], [88]]
    assert outputs[0].new_logprobs is not None
    assert outputs[0].new_logprobs.logprob_token_ids.tolist() == [
        [11, 11],
        [12, 12],
        [99, 99],
    ]
    assert outputs[1].new_logprobs is not None
    assert outputs[1].new_logprobs.logprob_token_ids.tolist() == [[77, 77]]


def test_distributed_client_streams_prompt_logprobs_from_verifier(
    monkeypatch: pytest.MonkeyPatch,
    verifier_server,
) -> None:
    base_url, _call_log = verifier_server
    monkeypatch.setattr(edge_client_mod, "EdgeDraftRunner", FakeDraftRunner)
    client = EngineCoreClient.make_client(
        multiprocess_mode=True,
        asyncio_mode=False,
        vllm_config=make_fake_vllm_config(base_url),
        executor_class=object,
        log_stats=False,
    )
    try:
        client.add_request(
            make_request(
                request_id="request-prompt-logprobs",
                prompt_logprobs=1,
            )
        )
        outputs = _collect_outputs_sync_detailed(client)
    finally:
        client.shutdown()

    assert outputs[0].new_token_ids == []
    assert outputs[0].new_prompt_logprobs_tensors is not None
    assert outputs[0].new_prompt_logprobs_tensors.logprob_token_ids.tolist() == [
        [2, 2],
        [3, 3],
    ]
    assert [output.new_token_ids for output in outputs[1:]] == [
        [11, 12, 99],
        [77],
        [88],
    ]


class FakeStructuredOutputFactory:
    def __init__(
        self,
        model_name: str,
        trust_remote_code: bool,
        *,
        num_speculative_tokens: int = 0,
        **_: Any,
    ) -> None:
        self.model_name = model_name
        self.trust_remote_code = trust_remote_code
        self.num_speculative_tokens = num_speculative_tokens
        self.closed = False
        self.created_sessions: list[StructuredOutputSession] = []
        self.resolve_calls: list[str] = []

    def resolve_sampling_params(self, params: SamplingParams, structured_outputs_config: Any) -> None:
        assert structured_outputs_config is not None
        assert params.structured_outputs is not None
        params.structured_outputs._backend = structured_outputs_config.backend
        self.resolve_calls.append(structured_outputs_config.backend)

    def create_session(
        self,
        request_id: str,
        sampling: Any,
    ) -> StructuredOutputSession | None:
        if sampling.structured_output_backend is None:
            return None
        session = StructuredOutputSession(
            request_id=request_id,
            grammar=FakeStructuredOutputGrammar(),
            bitmask=torch.zeros((1, 2), dtype=torch.int32),
        )
        self.created_sessions.append(session)
        return session

    def close(self) -> None:
        self.closed = True


class FakeStructuredOutputGrammar:
    def __init__(self) -> None:
        self.accepted: list[int] = []

    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        del request_id
        self.accepted.extend(tokens)
        return True

    def validate_tokens(self, tokens: list[int]) -> list[int]:
        return list(tokens)

    def rollback(self, num_tokens: int) -> None:
        if num_tokens > 0:
            del self.accepted[-num_tokens:]

    def fill_bitmask(self, bitmask: torch.Tensor, batch_index: int) -> None:
        bitmask[batch_index].fill_(-1)

    def is_terminated(self) -> bool:
        return False

    def reset(self) -> None:
        self.accepted = []


def test_distributed_client_supports_structured_outputs(
    monkeypatch: pytest.MonkeyPatch,
    verifier_server,
) -> None:
    base_url, call_log = verifier_server
    monkeypatch.setattr(edge_client_mod, "EdgeDraftRunner", FakeDraftRunner)
    monkeypatch.setattr(
        edge_client_mod,
        "StructuredOutputFactory",
        FakeStructuredOutputFactory,
    )
    client = EngineCoreClient.make_client(
        multiprocess_mode=True,
        asyncio_mode=False,
        vllm_config=make_fake_vllm_config(
            base_url,
            with_draft_model=True,
            structured_outputs_backend="outlines",
        ),
        executor_class=object,
        log_stats=False,
    )
    fake_factory = client._structured_output_factory
    assert isinstance(fake_factory, FakeStructuredOutputFactory)
    try:
        client.add_request(
            make_request(
                request_id="request-structured",
                max_tokens=5,
                structured_outputs=StructuredOutputsParams(choice=["yes", "no"]),
            )
        )
        tokens, finish_reason = _collect_outputs_sync(client)
    finally:
        draft_runner = client._draft_runner
        client.shutdown()

    assert tokens == [11, 12, 99, 77, 88]
    assert finish_reason == FinishReason.LENGTH
    assert fake_factory.resolve_calls == ["outlines"]
    assert len(fake_factory.created_sessions) == 1
    assert fake_factory.created_sessions[0].request_id == "request-structured"
    assert draft_runner.structured_output_sessions == fake_factory.created_sessions * 3
    assert fake_factory.created_sessions[0].grammar.accepted == [11, 12, 99, 77, 88]
    assert call_log[0] == ("open_session", "request-structured", "outlines")
    assert fake_factory.closed


def test_distributed_client_resyncs_after_verifier_session_loss(
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
        client.add_request(make_request(request_id="request-session-loss", max_tokens=5))
        tokens, finish_reason = _collect_outputs_sync(client)
    finally:
        client.shutdown()

    assert tokens == [11, 12, 99, 77, 88]
    assert finish_reason == FinishReason.LENGTH
    assert call_log[:4] == [
        ("open_session", "request-session-loss", None),
        ("verify_missing_session", 0),
        ("resync_session", "request-session-loss", [1, 2, 3], 3, True),
        ("verify_proposal", 0, [11, 12]),
    ]
