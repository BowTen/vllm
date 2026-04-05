# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from http import HTTPStatus

from fastapi import APIRouter, FastAPI, HTTPException, Request

router = APIRouter(prefix="/server/dssd", tags=["dssd"])


@router.post("/bind")
async def bind_verifier(raw_request: Request):
    service = getattr(raw_request.app.state, "dssd_verifier_service", None)
    if service is not None and hasattr(service, "bind_verifier"):
        return await service.bind_verifier()
    raise HTTPException(
        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        detail="DSSD verifier service is not available",
    )


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
