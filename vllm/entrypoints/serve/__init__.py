# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fastapi import FastAPI

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)


def _is_dssd_verifier_enabled(args: object) -> bool:
    dssd_config = getattr(args, "dssd_config", None)
    if dssd_config is None:
        return False

    if isinstance(dssd_config, dict):
        enabled = dssd_config.get("enabled", False)
        role = dssd_config.get("role")
    else:
        enabled = getattr(dssd_config, "enabled", False)
        role = getattr(dssd_config, "role", None)

    return bool(enabled and role == "verifier")


def register_vllm_serve_api_routers(app: FastAPI):
    if envs.VLLM_SERVER_DEV_MODE:
        logger.warning(
            "SECURITY WARNING: Development endpoints are enabled! "
            "This should NOT be used in production!"
        )

    from vllm.entrypoints.serve.lora.api_router import (
        attach_router as attach_lora_router,
    )

    attach_lora_router(app)

    from vllm.entrypoints.serve.profile.api_router import (
        attach_router as attach_profile_router,
    )

    attach_profile_router(app)

    from vllm.entrypoints.serve.sleep.api_router import (
        attach_router as attach_sleep_router,
    )

    attach_sleep_router(app)

    from vllm.entrypoints.serve.rpc.api_router import (
        attach_router as attach_rpc_router,
    )

    attach_rpc_router(app)

    from vllm.entrypoints.serve.cache.api_router import (
        attach_router as attach_cache_router,
    )

    attach_cache_router(app)

    from vllm.entrypoints.serve.tokenize.api_router import (
        attach_router as attach_tokenize_router,
    )

    attach_tokenize_router(app)

    if _is_dssd_verifier_enabled(app.state.args):
        from vllm.entrypoints.serve.dssd.api_router import (
            attach_router as attach_dssd_router,
        )

        attach_dssd_router(app)

    from .instrumentator import register_instrumentator_api_routers

    register_instrumentator_api_routers(app)
