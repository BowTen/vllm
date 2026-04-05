# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from fastapi import APIRouter, FastAPI, Request

router = APIRouter(prefix="/server/dssd", tags=["dssd"])


@router.post("/bind")
async def bind_verifier(raw_request: Request):
    service = getattr(raw_request.app.state, "dssd_verifier_service", None)
    if service is not None and hasattr(service, "bind_verifier"):
        return await service.bind_verifier()
    return {"ok": True}


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
