# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from http import HTTPStatus

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import msgspec

from vllm.v1.dssd.protocol import (
    BindVerifierRequest,
    CloseSessionRequest,
    CreateSessionRequest,
    VerifyRoundRequest,
)

router = APIRouter(prefix="/server/dssd", tags=["dssd"])


def _get_service(raw_request: Request):
    service = getattr(raw_request.app.state, "dssd_verifier_service", None)
    if service is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="DSSD verifier service is not available",
        )
    return service


def _response_payload(payload):
    if hasattr(payload, "__struct_fields__"):
        return JSONResponse(content=msgspec.to_builtins(payload))
    return payload


def _convert_request(payload, request_type):
    return msgspec.convert(payload, type=request_type)


@router.post("/bind", response_model=None)
async def bind_verifier(payload: dict, raw_request: Request):
    service = _get_service(raw_request)
    return _response_payload(
        await service.bind_verifier(
            _convert_request(payload, BindVerifierRequest),
        )
    )


@router.post("/sessions", response_model=None)
async def create_session(payload: dict, raw_request: Request):
    service = _get_service(raw_request)
    return _response_payload(
        await service.create_session(
            _convert_request(payload, CreateSessionRequest),
        )
    )


@router.post("/verify_round", response_model=None)
async def verify_round(payload: dict, raw_request: Request):
    service = _get_service(raw_request)
    return _response_payload(
        await service.verify_round(
            _convert_request(payload, VerifyRoundRequest),
        )
    )


@router.post("/close_session", response_model=None)
async def close_session(payload: dict, raw_request: Request):
    service = _get_service(raw_request)
    return _response_payload(
        await service.close_session(
            _convert_request(payload, CloseSessionRequest),
        )
    )


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
