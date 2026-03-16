# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass

import msgspec.msgpack
import uvicorn
from fastapi import FastAPI, Request, Response

from vllm.logger import init_logger
from vllm.v1.spec_decode.distributed.protocol import (
    CloseSessionRequest,
    DraftProposal,
    MSGPACK_ENCODER,
    OpenSessionRequest,
    OpenSessionResponse,
    ResyncSessionRequest,
    ResyncSessionResponse,
    VerificationResult,
)
from vllm.v1.spec_decode.distributed.verifier import TargetVerificationRunner
from vllm.v1.spec_decode.distributed.verification_core import VerificationCore

logger = init_logger(__name__)


@dataclass
class VerifierServerArgs:
    model: str
    host: str = "0.0.0.0"
    port: int = 9000
    device: str | None = None
    dtype: str = "auto"
    trust_remote_code: bool = False
    log_level: str = "info"
    scheduler_max_batch_size: int = 8
    scheduler_batch_wait_ms: float = 1.0
    scheduler_max_batch_tokens: int | None = None
    scheduler_queue_timeout_ms: float | None = None
    session_idle_timeout_s: float | None = None


def build_app(args: VerifierServerArgs) -> FastAPI:
    runner = TargetVerificationRunner(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    core = VerificationCore(
        runner,
        scheduler_max_batch_size=args.scheduler_max_batch_size,
        scheduler_batch_wait_ms=args.scheduler_batch_wait_ms,
        scheduler_max_batch_tokens=args.scheduler_max_batch_tokens,
        scheduler_queue_timeout_ms=args.scheduler_queue_timeout_ms,
        session_idle_timeout_s=args.session_idle_timeout_s,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        await core.shutdown()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/open_session")
    async def open_session(request: Request) -> Response:
        payload = msgspec.msgpack.decode(
            await request.body(), type=OpenSessionRequest
        )
        result: OpenSessionResponse = await core.open_session(payload)
        return Response(
            content=MSGPACK_ENCODER.encode(result),
            media_type="application/msgpack",
        )

    @app.post("/verify_proposal")
    async def verify_proposal(request: Request) -> Response:
        payload = msgspec.msgpack.decode(await request.body(), type=DraftProposal)
        result: VerificationResult = await core.verify_proposal(payload)
        return Response(
            content=MSGPACK_ENCODER.encode(result),
            media_type="application/msgpack",
        )

    @app.post("/resync_session")
    async def resync_session(request: Request) -> Response:
        payload = msgspec.msgpack.decode(
            await request.body(), type=ResyncSessionRequest
        )
        result: ResyncSessionResponse = await core.resync_session(payload)
        return Response(
            content=MSGPACK_ENCODER.encode(result),
            media_type="application/msgpack",
        )

    @app.post("/close_session")
    async def close_session(request: Request) -> Response:
        payload = msgspec.msgpack.decode(
            await request.body(), type=CloseSessionRequest
        )
        await core.close_session(payload)
        return Response(status_code=204)

    return app


def run_verifier_server(args: VerifierServerArgs) -> None:
    logger.info(
        "Starting distributed speculative verifier for model %s on %s:%d",
        args.model,
        args.host,
        args.port,
    )
    uvicorn.run(
        build_app(args),
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
