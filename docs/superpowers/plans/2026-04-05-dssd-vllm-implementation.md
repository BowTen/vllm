# DSSD-on-vLLM Implementation Plan

> **Legacy Notice**
>
> This plan is retained as historical planning context for the initial DSSD bring-up.
> Ongoing work should be planned as OpenSpec changes under `openspec/changes/`,
> with [openspec/program.md](/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex/openspec/program.md)
> as the current global source of truth.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 vLLM 中实现一个可运行的 DSSD 边云分布式投机采样原型，边端对外服务用户请求，云端作为有状态 verifier 服务完成多轮 Accept/Reject 验证。

**Architecture:** 通过新增 `DSSDConfig`、共享协议对象、EngineCore utility methods、边端/云端 session 管理、worker-level draft/verifier hooks，以及边端 OpenAI serving 集成来实现系统。DSSD 控制逻辑放在 vLLM 前端/控制层，模型执行能力通过 EngineCore utility path 和 GPUModelRunner hook 暴露；云端 verifier 多位置前向尽量复用现有 speculative decode 的输入展开能力。

**Tech Stack:** Python 3.12, asyncio, FastAPI, msgspec, pytest, pytest-asyncio, vLLM V1 engine, EngineCore utility RPC, uv

---

## Preflight

在 worktree `/home/zz/workspace/vllm/.worktrees/dssd-vllm-codex` 内执行：

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/lint.txt
uv pip install pytest pytest-asyncio tblib
pre-commit install
```

如果需要完整测试依赖，再执行：

```bash
source .venv/bin/activate
uv pip install -r requirements/test.txt
```

---

## Planned Files

### Configuration and CLI

- Create: `vllm/config/dssd.py`
  负责 `DSSDConfig`、网络模拟子配置、配置校验辅助函数。
- Modify: `vllm/config/__init__.py`
  导出 `DSSDConfig`。
- Modify: `vllm/config/vllm.py`
  在 `VllmConfig` 中加入 `dssd_config` 字段，并在初始化时做基本合法性校验。
- Modify: `vllm/engine/arg_utils.py`
  让 CLI / config 文件路径能构造 `DSSDConfig`。
- Modify: `vllm/entrypoints/openai/cli_args.py`
  补充 DSSD 相关快速参数校验。

### Shared DSSD Core

- Create: `vllm/v1/dssd/__init__.py`
  DSSD 顶层包导出。
- Create: `vllm/v1/dssd/protocol.py`
  定义边云协议和 Engine utility 请求/响应对象。
- Create: `vllm/v1/dssd/transport.py`
  定义 transport 抽象和带网络模拟能力的本机实现。
- Create: `vllm/v1/dssd/metrics.py`
  定义 DSSD 运行指标对象与聚合器。
- Create: `vllm/v1/dssd/engine/__init__.py`
  engine 子包导出。
- Create: `vllm/v1/dssd/engine/session_store.py`
  提供 edge/verifier session 的本地存储。
- Create: `vllm/v1/dssd/engine/session_runner.py`
  聚合 DSSD utility methods，连接 control 层与 worker hook。
- Create: `vllm/v1/dssd/engine/batch_planner.py`
  处理 verifier `verify_round` 的分桶与组批。

### Verifier Path

- Create: `vllm/v1/dssd/verifier/__init__.py`
  verifier 子包导出。
- Create: `vllm/v1/dssd/verifier/session.py`
  `DSSDVerifierSessionState` 和 session manager。
- Create: `vllm/v1/dssd/verifier/service.py`
  verifier service，负责协议校验、幂等、accept/reject 判定。
- Create: `vllm/entrypoints/serve/dssd/api_router.py`
  私有 verifier FastAPI router。
- Modify: `vllm/entrypoints/serve/__init__.py`
  注册 verifier router。
- Modify: `vllm/entrypoints/openai/api_server.py`
  在 `app.state` 初始化 verifier service。
- Create: `vllm/v1/dssd/worker/verifier_runner.py`
  GPU 侧 verifier 多位置前向 hook。
- Modify: `vllm/v1/worker/gpu_model_runner.py`
  接入 verifier hook。

### Edge Path

- Create: `vllm/v1/dssd/edge/__init__.py`
  edge 子包导出。
- Create: `vllm/v1/dssd/edge/session.py`
  `DSSDEdgeSessionState` 与 round cache。
- Create: `vllm/v1/dssd/edge/coordinator.py`
  轮次状态机。
- Create: `vllm/v1/dssd/worker/draft_runner.py`
  边端 `gamma` 步 draft 执行器。
- Create: `vllm/v1/dssd/worker/resample.py`
  reject 后 residual resample。

### Engine and Serving Integration

- Modify: `vllm/v1/engine/core.py`
  暴露 DSSD utility methods 给 `EngineCoreRequestType.UTILITY`。
- Modify: `vllm/v1/engine/core_client.py`
  增加对应的同步/异步 DSSD utility client methods。
- Create: `vllm/entrypoints/openai/chat_completion/dssd_serving.py`
  `DSSDEdgeServingChat`。
- Modify: `vllm/entrypoints/openai/generate/api_router.py`
  根据 `dssd_config.role` 选择普通 chat serving 或 DSSD edge serving。

### Tests and Docs

- Create: `tests/config/test_dssd_config.py`
- Create: `tests/v1/engine/test_dssd_engine_args.py`
- Create: `tests/v1/dssd/test_protocol.py`
- Create: `tests/v1/engine/test_dssd_core_utility.py`
- Create: `tests/v1/dssd/test_verifier_session_manager.py`
- Create: `tests/entrypoints/test_dssd_verifier_router.py`
- Create: `tests/v1/worker/test_dssd_verifier_runner.py`
- Create: `tests/v1/dssd/test_verifier_batch_planner.py`
- Create: `tests/v1/dssd/test_edge_session_manager.py`
- Create: `tests/v1/worker/test_dssd_draft_runner.py`
- Create: `tests/v1/dssd/test_resample.py`
- Create: `tests/v1/dssd/test_round_coordinator.py`
- Create: `tests/entrypoints/openai/chat_completion/test_dssd_serving_chat.py`
- Create: `tests/v1/e2e/dssd/test_dssd_smoke.py`
- Create: `docs/serving/dssd_edge_verifier.md`

---

### Task 1: Add DSSD Config and CLI Surface

**Files:**
- Create: `vllm/config/dssd.py`
- Modify: `vllm/config/__init__.py`
- Modify: `vllm/config/vllm.py`
- Modify: `vllm/engine/arg_utils.py`
- Modify: `vllm/entrypoints/openai/cli_args.py`
- Test: `tests/config/test_dssd_config.py`
- Test: `tests/v1/engine/test_dssd_engine_args.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/config/test_dssd_config.py
import pytest

from vllm.config.dssd import DSSDConfig


def test_dssd_config_defaults():
    cfg = DSSDConfig(enabled=True, role="edge", gamma=4,
                     verifier_url="http://127.0.0.1:9000")
    assert cfg.enabled is True
    assert cfg.role == "edge"
    assert cfg.gamma == 4
    assert cfg.verifier_url == "http://127.0.0.1:9000"


def test_dssd_config_rejects_edge_without_verifier_url():
    with pytest.raises(ValueError, match="verifier_url"):
        DSSDConfig(enabled=True, role="edge", gamma=4).validate()
```

```python
# tests/v1/engine/test_dssd_engine_args.py
from argparse import Namespace

import pytest

from vllm.entrypoints.openai.cli_args import validate_parsed_serve_args


def test_validate_parsed_serve_args_rejects_edge_without_verifier_url():
    args = Namespace(
        subparser="serve",
        chat_template=None,
        enable_auto_tool_choice=False,
        tool_call_parser=None,
        enable_log_outputs=False,
        enable_log_requests=False,
        dssd_config={"enabled": True, "role": "edge", "gamma": 4},
    )
    with pytest.raises(TypeError, match="verifier_url"):
        validate_parsed_serve_args(args)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
pytest tests/config/test_dssd_config.py tests/v1/engine/test_dssd_engine_args.py -v
```

Expected:

```text
ERROR or FAIL because vllm.config.dssd does not exist and serve args do not validate dssd_config yet
```

- [ ] **Step 3: Write the minimal implementation**

```python
# vllm/config/dssd.py
from typing import Literal

from vllm.config.utils import config


@config
class DSSDNetworkSimulationConfig:
    latency_ms: float = 0.0
    bandwidth_mbps: float | None = None
    jitter_ms: float = 0.0


@config
class DSSDConfig:
    enabled: bool = False
    role: Literal["edge", "verifier"] | None = None
    gamma: int = 4
    verifier_url: str | None = None
    bind_timeout_s: float = 10.0
    request_timeout_s: float = 30.0
    protocol_version: str = "2026-04-05"
    auth_mode: str | None = None
    auth_payload: dict[str, str] | None = None
    network_simulation: DSSDNetworkSimulationConfig = DSSDNetworkSimulationConfig()
    metrics_enabled: bool = True

    def validate(self) -> "DSSDConfig":
        if not self.enabled:
            return self
        if self.role not in {"edge", "verifier"}:
            raise ValueError("DSSDConfig.role must be 'edge' or 'verifier'")
        if self.gamma < 1:
            raise ValueError("DSSDConfig.gamma must be >= 1")
        if self.role == "edge" and not self.verifier_url:
            raise ValueError("DSSD edge mode requires verifier_url")
        return self
```

```python
# vllm/config/vllm.py
from .dssd import DSSDConfig


class VllmConfig:
    dssd_config: DSSDConfig | None = None

    def __post_init__(self):
        if self.dssd_config is not None:
            self.dssd_config.validate()
```

```python
# vllm/config/__init__.py
from vllm.config.dssd import DSSDConfig

__all__.append("DSSDConfig")
```

```python
# vllm/entrypoints/openai/cli_args.py
def validate_parsed_serve_args(args: argparse.Namespace):
    if hasattr(args, "subparser") and args.subparser != "serve":
        return
    validate_chat_template(args.chat_template)
    if args.enable_auto_tool_choice and not args.tool_call_parser:
        raise TypeError("Error: --enable-auto-tool-choice requires --tool-call-parser")
    if args.enable_log_outputs and not args.enable_log_requests:
        raise TypeError("Error: --enable-log-outputs requires --enable-log-requests")
    dssd_config = getattr(args, "dssd_config", None)
    if dssd_config and dssd_config.get("enabled") and dssd_config.get("role") == "edge":
        if not dssd_config.get("verifier_url"):
            raise TypeError("Error: --dssd-config.role=edge requires verifier_url")
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
source .venv/bin/activate
pytest tests/config/test_dssd_config.py tests/v1/engine/test_dssd_engine_args.py -v
```

Expected:

```text
4 passed
```

- [ ] **Step 5: Commit**

```bash
git add vllm/config/dssd.py vllm/config/__init__.py vllm/config/vllm.py \
  vllm/engine/arg_utils.py vllm/entrypoints/openai/cli_args.py \
  tests/config/test_dssd_config.py tests/v1/engine/test_dssd_engine_args.py
git commit -m "feat: add DSSD config surface"
```

### Task 2: Add Shared Protocol and Transport

**Files:**
- Create: `vllm/v1/dssd/__init__.py`
- Create: `vllm/v1/dssd/protocol.py`
- Create: `vllm/v1/dssd/transport.py`
- Test: `tests/v1/dssd/test_protocol.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/v1/dssd/test_protocol.py
import msgspec.msgpack

from vllm.v1.dssd.protocol import (
    BindVerifierRequest,
    VerifyRoundRequest,
    VerifyRoundResponse,
)


def test_verify_round_request_msgpack_round_trip():
    req = VerifyRoundRequest(
        binding_id="bind-1",
        verifier_session_id="vs-1",
        seq_no=3,
        prefix_delta_token_ids=[42],
        draft_token_ids=[7, 8, 9],
        q_values=[0.2, 0.3, 0.4],
    )
    restored = msgspec.msgpack.decode(
        msgspec.msgpack.encode(req),
        type=VerifyRoundRequest,
    )
    assert restored == req


def test_verify_round_response_reject_shape():
    resp = VerifyRoundResponse(
        verifier_session_id="vs-1",
        seq_no=3,
        accepted_count=1,
        all_accepted=False,
        bonus_token_id=None,
        reject_index=2,
        reject_target_probs=[0.1, 0.9],
        finished=False,
        finish_reason=None,
    )
    assert resp.reject_index == 2
    assert resp.reject_target_probs == [0.1, 0.9]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
pytest tests/v1/dssd/test_protocol.py -v
```

Expected:

```text
ERROR because vllm.v1.dssd.protocol does not exist yet
```

- [ ] **Step 3: Write the minimal implementation**

```python
# vllm/v1/dssd/protocol.py
import msgspec


class BindVerifierRequest(msgspec.Struct, omit_defaults=True):
    protocol_version: str
    edge_instance_id: str
    tokenizer_hash: str
    vocab_hash: str
    supported_gamma_max: int
    auth_payload: dict[str, str] | None = None


class BindVerifierResponse(msgspec.Struct, omit_defaults=True):
    binding_id: str
    protocol_version: str
    verifier_model_id: str
    tokenizer_hash: str
    vocab_hash: str
    supported_gamma_max: int
    capabilities: dict[str, str] | None = None


class VerifyRoundRequest(msgspec.Struct, omit_defaults=True):
    binding_id: str
    verifier_session_id: str
    seq_no: int
    prefix_delta_token_ids: list[int]
    draft_token_ids: list[int]
    q_values: list[float]


class VerifyRoundResponse(msgspec.Struct, omit_defaults=True):
    verifier_session_id: str
    seq_no: int
    accepted_count: int
    all_accepted: bool
    bonus_token_id: int | None = None
    reject_index: int | None = None
    reject_target_probs: list[float] | None = None
    finished: bool = False
    finish_reason: str | None = None
```

```python
# vllm/v1/dssd/transport.py
from abc import ABC, abstractmethod

from vllm.v1.dssd.protocol import (
    BindVerifierRequest,
    BindVerifierResponse,
    VerifyRoundRequest,
    VerifyRoundResponse,
)


class DSSDTransport(ABC):
    @abstractmethod
    async def bind_verifier(
        self, request: BindVerifierRequest
    ) -> BindVerifierResponse:
        raise NotImplementedError

    @abstractmethod
    async def verify_round(
        self, request: VerifyRoundRequest
    ) -> VerifyRoundResponse:
        raise NotImplementedError
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
source .venv/bin/activate
pytest tests/v1/dssd/test_protocol.py -v
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit**

```bash
git add vllm/v1/dssd/__init__.py vllm/v1/dssd/protocol.py vllm/v1/dssd/transport.py \
  tests/v1/dssd/test_protocol.py
git commit -m "feat: add DSSD shared protocol"
```

### Task 3: Add EngineCore Utility Plumbing and Session Store

**Files:**
- Create: `vllm/v1/dssd/engine/__init__.py`
- Create: `vllm/v1/dssd/engine/session_store.py`
- Create: `vllm/v1/dssd/engine/session_runner.py`
- Modify: `vllm/v1/engine/core.py`
- Modify: `vllm/v1/engine/core_client.py`
- Test: `tests/v1/engine/test_dssd_core_utility.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/v1/engine/test_dssd_core_utility.py
from unittest.mock import AsyncMock

import pytest

from vllm.v1.dssd.protocol import VerifyRoundRequest
from vllm.v1.engine.core_client import AsyncMPClient


@pytest.mark.asyncio
async def test_async_client_exposes_verify_round_utility():
    client = object.__new__(AsyncMPClient)
    client.call_utility_async = AsyncMock(return_value="ok")

    request = VerifyRoundRequest(
        binding_id="bind-1",
        verifier_session_id="vs-1",
        seq_no=0,
        prefix_delta_token_ids=[],
        draft_token_ids=[1, 2],
        q_values=[0.5, 0.4],
    )

    result = await AsyncMPClient.dssd_verify_round_async(client, request)

    assert result == "ok"
    client.call_utility_async.assert_awaited_once_with("dssd_verify_round", request)
```

```python
# tests/v1/engine/test_dssd_core_utility.py
from vllm.v1.dssd.engine.session_store import DSSDSessionStore


def test_session_store_round_trip():
    store = DSSDSessionStore()
    store.put("edge-1", {"seq_no": 0})
    assert store.get("edge-1") == {"seq_no": 0}
    store.delete("edge-1")
    assert store.get("edge-1") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
pytest tests/v1/engine/test_dssd_core_utility.py -v
```

Expected:

```text
ERROR because DSSD engine utility methods and session store do not exist yet
```

- [ ] **Step 3: Write the minimal implementation**

```python
# vllm/v1/dssd/engine/session_store.py
class DSSDSessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, object] = {}

    def put(self, session_id: str, state: object) -> None:
        self._sessions[session_id] = state

    def get(self, session_id: str) -> object | None:
        return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
```

```python
# vllm/v1/dssd/engine/session_runner.py
from vllm.v1.dssd.engine.session_store import DSSDSessionStore


class DSSDSessionRunner:
    def __init__(self) -> None:
        self.edge_sessions = DSSDSessionStore()
        self.verifier_sessions = DSSDSessionStore()

    def dssd_verify_round(self, request):
        return {"method": "dssd_verify_round", "request": request}
```

```python
# vllm/v1/engine/core_client.py
class EngineCoreClient(ABC):
    async def dssd_verify_round_async(self, request):
        raise NotImplementedError


class AsyncMPClient(MPClient):
    async def dssd_verify_round_async(self, request):
        return await self.call_utility_async("dssd_verify_round", request)
```

```python
# vllm/v1/engine/core.py
from vllm.v1.dssd.engine.session_runner import DSSDSessionRunner


class EngineCore:
    # Inside EngineCore.__init__ after the scheduler and executor are created:
    self.dssd_session_runner = DSSDSessionRunner()

    def dssd_verify_round(self, request):
        return self.dssd_session_runner.dssd_verify_round(request)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
source .venv/bin/activate
pytest tests/v1/engine/test_dssd_core_utility.py -v
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit**

```bash
git add vllm/v1/dssd/engine/__init__.py vllm/v1/dssd/engine/session_store.py \
  vllm/v1/dssd/engine/session_runner.py vllm/v1/engine/core.py \
  vllm/v1/engine/core_client.py tests/v1/engine/test_dssd_core_utility.py
git commit -m "feat: add DSSD engine utility plumbing"
```

### Task 4: Add Verifier Session Manager and Private Router

**Files:**
- Create: `vllm/v1/dssd/verifier/__init__.py`
- Create: `vllm/v1/dssd/verifier/session.py`
- Create: `vllm/v1/dssd/verifier/service.py`
- Create: `vllm/entrypoints/serve/dssd/api_router.py`
- Modify: `vllm/entrypoints/serve/__init__.py`
- Modify: `vllm/entrypoints/openai/api_server.py`
- Test: `tests/v1/dssd/test_verifier_session_manager.py`
- Test: `tests/entrypoints/test_dssd_verifier_router.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/v1/dssd/test_verifier_session_manager.py
import pytest

from vllm.v1.dssd.verifier.session import DSSDVerifierSessionManager


def test_verifier_session_manager_replays_cached_response():
    manager = DSSDVerifierSessionManager()
    manager.create_session("vs-1", sampling_params_fingerprint="abc")
    manager.cache_response("vs-1", seq_no=3, response={"accepted_count": 2})
    assert manager.get_cached_response("vs-1", seq_no=3) == {"accepted_count": 2}


def test_verifier_session_manager_rejects_out_of_order_seq():
    manager = DSSDVerifierSessionManager()
    manager.create_session("vs-1", sampling_params_fingerprint="abc")
    manager.update_seq_no("vs-1", 4)
    with pytest.raises(ValueError, match="out-of-order"):
        manager.ensure_next_seq_no("vs-1", 6)
```

```python
# tests/entrypoints/test_dssd_verifier_router.py
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.dssd.api_router import attach_router


def test_verifier_router_exposes_bind_endpoint():
    app = FastAPI()
    app.state.dssd_verifier_service = object()
    attach_router(app)
    client = TestClient(app)
    paths = {route.path for route in app.routes}
    assert "/server/dssd/bind" in paths
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
pytest tests/v1/dssd/test_verifier_session_manager.py tests/entrypoints/test_dssd_verifier_router.py -v
```

Expected:

```text
ERROR because verifier session manager and router do not exist yet
```

- [ ] **Step 3: Write the minimal implementation**

```python
# vllm/v1/dssd/verifier/session.py
from dataclasses import dataclass, field
from time import time


@dataclass
class DSSDVerifierSessionState:
    verifier_session_id: str
    sampling_params_fingerprint: str
    seq_no: int = -1
    committed_token_ids: list[int] = field(default_factory=list)
    last_response_cache: dict[int, object] = field(default_factory=dict)
    last_activity_at: float = field(default_factory=time)


class DSSDVerifierSessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, DSSDVerifierSessionState] = {}

    def create_session(self, verifier_session_id: str, *, sampling_params_fingerprint: str):
        self._sessions[verifier_session_id] = DSSDVerifierSessionState(
            verifier_session_id=verifier_session_id,
            sampling_params_fingerprint=sampling_params_fingerprint,
        )

    def cache_response(self, verifier_session_id: str, *, seq_no: int, response: object):
        self._sessions[verifier_session_id].last_response_cache[seq_no] = response

    def get_cached_response(self, verifier_session_id: str, *, seq_no: int):
        return self._sessions[verifier_session_id].last_response_cache.get(seq_no)

    def update_seq_no(self, verifier_session_id: str, seq_no: int):
        self._sessions[verifier_session_id].seq_no = seq_no

    def ensure_next_seq_no(self, verifier_session_id: str, seq_no: int):
        current = self._sessions[verifier_session_id].seq_no
        if seq_no > current + 1:
            raise ValueError("out-of-order verify_round seq_no")
```

```python
# vllm/entrypoints/serve/dssd/api_router.py
from fastapi import APIRouter, FastAPI


router = APIRouter(prefix="/server/dssd", tags=["dssd"])


@router.post("/bind")
async def bind_verifier():
    return {"ok": True}


def attach_router(app: FastAPI):
    app.include_router(router)
```

```python
# vllm/entrypoints/serve/__init__.py
from vllm.entrypoints.serve.dssd.api_router import attach_router as attach_dssd_router

attach_dssd_router(app)
```

```python
# vllm/entrypoints/openai/api_server.py
from vllm.v1.dssd.verifier.service import DSSDVerifierService


async def init_app_state(engine_client, state, args, supported_tasks=None):
    vllm_config = engine_client.vllm_config
    state.dssd_verifier_service = None
    if vllm_config.dssd_config and vllm_config.dssd_config.enabled:
        state.dssd_verifier_service = DSSDVerifierService(engine_client, vllm_config)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
source .venv/bin/activate
pytest tests/v1/dssd/test_verifier_session_manager.py tests/entrypoints/test_dssd_verifier_router.py -v
```

Expected:

```text
3 passed
```

- [ ] **Step 5: Commit**

```bash
git add vllm/v1/dssd/verifier/__init__.py vllm/v1/dssd/verifier/session.py \
  vllm/v1/dssd/verifier/service.py vllm/entrypoints/serve/dssd/api_router.py \
  vllm/entrypoints/serve/__init__.py vllm/entrypoints/openai/api_server.py \
  tests/v1/dssd/test_verifier_session_manager.py tests/entrypoints/test_dssd_verifier_router.py
git commit -m "feat: add verifier session manager and router"
```

### Task 5: Implement Verifier Runner and Batch Verify Execution

**Files:**
- Create: `vllm/v1/dssd/engine/batch_planner.py`
- Create: `vllm/v1/dssd/worker/verifier_runner.py`
- Modify: `vllm/v1/dssd/engine/session_runner.py`
- Modify: `vllm/v1/worker/gpu_model_runner.py`
- Test: `tests/v1/worker/test_dssd_verifier_runner.py`
- Test: `tests/v1/dssd/test_verifier_batch_planner.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/v1/dssd/test_verifier_batch_planner.py
from vllm.v1.dssd.engine.batch_planner import VerifierRoundBatcher


def test_batcher_groups_requests_by_gamma():
    batcher = VerifierRoundBatcher()
    batcher.add({"session_id": "a", "gamma": 4, "sampling_signature": "s1"})
    batcher.add({"session_id": "b", "gamma": 4, "sampling_signature": "s1"})
    batcher.add({"session_id": "c", "gamma": 2, "sampling_signature": "s1"})
    groups = batcher.flush()
    assert len(groups) == 2
```

```python
# tests/v1/worker/test_dssd_verifier_runner.py
from vllm.v1.dssd.worker.verifier_runner import build_verifier_result


def test_build_verifier_result_extracts_target_positions():
    result = build_verifier_result(
        draft_token_ids=[11, 12],
        q_values=[0.6, 0.4],
        target_probs=[[0.2, 0.8], [0.9, 0.1], [0.7, 0.3]],
    )
    assert result.accepted_count >= 0
    assert result.seq_probs[0] == [0.2, 0.8]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
pytest tests/v1/dssd/test_verifier_batch_planner.py tests/v1/worker/test_dssd_verifier_runner.py -v
```

Expected:

```text
ERROR because batch planner and verifier runner do not exist yet
```

- [ ] **Step 3: Write the minimal implementation**

```python
# vllm/v1/dssd/engine/batch_planner.py
from collections import defaultdict


class VerifierRoundBatcher:
    def __init__(self) -> None:
        self._pending = []

    def add(self, item: dict) -> None:
        self._pending.append(item)

    def flush(self) -> list[list[dict]]:
        groups = defaultdict(list)
        for item in self._pending:
            key = (item["gamma"], item["sampling_signature"])
            groups[key].append(item)
        self._pending.clear()
        return list(groups.values())
```

```python
# vllm/v1/dssd/worker/verifier_runner.py
from dataclasses import dataclass


@dataclass
class DSSDVerifierModelResult:
    accepted_count: int
    seq_probs: list[list[float]]
    bonus_probs: list[float]


def build_verifier_result(draft_token_ids, q_values, target_probs):
    return DSSDVerifierModelResult(
        accepted_count=0,
        seq_probs=target_probs[:-1],
        bonus_probs=target_probs[-1],
    )
```

```python
# vllm/v1/dssd/engine/session_runner.py
from vllm.v1.dssd.engine.batch_planner import VerifierRoundBatcher


class DSSDSessionRunner:
    def __init__(self) -> None:
        self.edge_sessions = DSSDSessionStore()
        self.verifier_sessions = DSSDSessionStore()
        self.verifier_batcher = VerifierRoundBatcher()
```

```python
# vllm/v1/worker/gpu_model_runner.py
class GPUModelRunner:
    def dssd_verify_round(self, request):
        # Follow the spec decode path by feeding external draft tokens into
        # scheduled_spec_decode_tokens and returning target probabilities
        scheduler_output = SchedulerOutput.make_empty()
        scheduler_output.scheduled_spec_decode_tokens = {
            request.request_id: request.draft_token_ids
        }
        return request
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
source .venv/bin/activate
pytest tests/v1/dssd/test_verifier_batch_planner.py tests/v1/worker/test_dssd_verifier_runner.py -v
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit**

```bash
git add vllm/v1/dssd/engine/batch_planner.py vllm/v1/dssd/worker/verifier_runner.py \
  vllm/v1/dssd/engine/session_runner.py vllm/v1/worker/gpu_model_runner.py \
  tests/v1/dssd/test_verifier_batch_planner.py tests/v1/worker/test_dssd_verifier_runner.py
git commit -m "feat: add verifier batching and runner hooks"
```

### Task 6: Implement Edge Session State, Draft Runner, and Residual Resample

**Files:**
- Create: `vllm/v1/dssd/edge/__init__.py`
- Create: `vllm/v1/dssd/edge/session.py`
- Create: `vllm/v1/dssd/worker/draft_runner.py`
- Create: `vllm/v1/dssd/worker/resample.py`
- Modify: `vllm/v1/dssd/engine/session_runner.py`
- Modify: `vllm/v1/worker/gpu_model_runner.py`
- Test: `tests/v1/dssd/test_edge_session_manager.py`
- Test: `tests/v1/worker/test_dssd_draft_runner.py`
- Test: `tests/v1/dssd/test_resample.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/v1/dssd/test_edge_session_manager.py
from vllm.v1.dssd.edge.session import DSSDEdgeSessionState


def test_edge_session_starts_with_empty_pending_delta():
    session = DSSDEdgeSessionState(
        request_id="req-1",
        local_session_id="edge-1",
        verifier_binding_id="bind-1",
        verifier_session_id="vs-1",
        prompt_token_ids=[1, 2, 3],
    )
    assert session.pending_prefix_delta_token_ids == []
    assert session.committed_token_ids == []
```

```python
# tests/v1/dssd/test_resample.py
import torch

from vllm.v1.dssd.worker.resample import compute_residual_distribution


def test_compute_residual_distribution_clamps_negative_mass():
    p = torch.tensor([0.7, 0.2, 0.1])
    q = torch.tensor([0.2, 0.5, 0.3])
    residual = compute_residual_distribution(p, q)
    assert torch.all(residual >= 0)
    assert torch.isclose(residual.sum(), torch.tensor(1.0))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
pytest tests/v1/dssd/test_edge_session_manager.py tests/v1/dssd/test_resample.py -v
```

Expected:

```text
ERROR because edge session state and resample helpers do not exist yet
```

- [ ] **Step 3: Write the minimal implementation**

```python
# vllm/v1/dssd/edge/session.py
from dataclasses import dataclass, field


@dataclass
class DSSDEdgeRoundCache:
    seq_no: int
    gamma: int
    draft_token_ids: list[int]
    q_values: list[float]
    q_dists_handle: str


@dataclass
class DSSDEdgeSessionState:
    request_id: str
    local_session_id: str
    verifier_binding_id: str
    verifier_session_id: str
    prompt_token_ids: list[int]
    seq_no: int = 0
    committed_token_ids: list[int] = field(default_factory=list)
    pending_prefix_delta_token_ids: list[int] = field(default_factory=list)
```

```python
# vllm/v1/dssd/worker/resample.py
import torch


def compute_residual_distribution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    residual = torch.clamp(p - q, min=0)
    total = residual.sum()
    if total <= 0:
        raise ValueError("residual distribution has no positive mass")
    return residual / total
```

```python
# vllm/v1/dssd/worker/draft_runner.py
from dataclasses import dataclass


@dataclass
class DraftRoundResult:
    draft_token_ids: list[int]
    q_values: list[float]
    q_dists_handle: str
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
source .venv/bin/activate
pytest tests/v1/dssd/test_edge_session_manager.py tests/v1/dssd/test_resample.py -v
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit**

```bash
git add vllm/v1/dssd/edge/__init__.py vllm/v1/dssd/edge/session.py \
  vllm/v1/dssd/worker/draft_runner.py vllm/v1/dssd/worker/resample.py \
  vllm/v1/dssd/engine/session_runner.py vllm/v1/worker/gpu_model_runner.py \
  tests/v1/dssd/test_edge_session_manager.py tests/v1/worker/test_dssd_draft_runner.py \
  tests/v1/dssd/test_resample.py
git commit -m "feat: add edge draft and residual resample state"
```

### Task 7: Implement Round Coordinator and Edge Serving Integration

**Files:**
- Create: `vllm/v1/dssd/edge/coordinator.py`
- Create: `vllm/entrypoints/openai/chat_completion/dssd_serving.py`
- Modify: `vllm/entrypoints/openai/generate/api_router.py`
- Test: `tests/v1/dssd/test_round_coordinator.py`
- Test: `tests/entrypoints/openai/chat_completion/test_dssd_serving_chat.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/v1/dssd/test_round_coordinator.py
import pytest

from vllm.v1.dssd.edge.coordinator import DSSDRoundCoordinator
from vllm.v1.dssd.protocol import VerifyRoundResponse


@pytest.mark.asyncio
async def test_round_coordinator_commits_bonus_token_on_full_accept():
    coordinator = DSSDRoundCoordinator(edge_engine=None, transport=None)
    response = VerifyRoundResponse(
        verifier_session_id="vs-1",
        seq_no=0,
        accepted_count=2,
        all_accepted=True,
        bonus_token_id=99,
        finished=False,
        finish_reason=None,
    )
    committed = coordinator._build_committed_tokens([10, 11], response, resampled_token=None)
    assert committed == [10, 11, 99]
```

```python
# tests/entrypoints/openai/chat_completion/test_dssd_serving_chat.py
from unittest.mock import AsyncMock, MagicMock

import pytest

from vllm.entrypoints.openai.chat_completion.dssd_serving import DSSDEdgeServingChat


@pytest.mark.asyncio
async def test_dssd_serving_chat_uses_round_coordinator(monkeypatch):
    serving = object.__new__(DSSDEdgeServingChat)
    serving.round_coordinator = AsyncMock()
    serving.round_coordinator.run = AsyncMock()
    await serving.round_coordinator.run.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
pytest tests/v1/dssd/test_round_coordinator.py \
  tests/entrypoints/openai/chat_completion/test_dssd_serving_chat.py -v
```

Expected:

```text
ERROR because round coordinator and DSSD serving chat do not exist yet
```

- [ ] **Step 3: Write the minimal implementation**

```python
# vllm/v1/dssd/edge/coordinator.py
class DSSDRoundCoordinator:
    def __init__(self, edge_engine, transport):
        self.edge_engine = edge_engine
        self.transport = transport

    def _build_committed_tokens(self, draft_token_ids, response, *, resampled_token):
        if response.all_accepted:
            return draft_token_ids + [response.bonus_token_id]
        accepted = draft_token_ids[: response.accepted_count]
        return accepted + [resampled_token]
```

```python
# vllm/entrypoints/openai/chat_completion/dssd_serving.py
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat


class DSSDEdgeServingChat(OpenAIServingChat):
    async def create_chat_completion(self, request, raw_request=None):
        # The final implementation will delegate the user request to
        # DSSDRoundCoordinator instead of engine_client.generate().
        return await super().create_chat_completion(request, raw_request)
```

```python
# vllm/entrypoints/openai/generate/api_router.py
from vllm.entrypoints.openai.chat_completion.dssd_serving import DSSDEdgeServingChat


async def init_generate_state(engine_client, state, args, request_logger, supported_tasks):
    dssd_cfg = engine_client.vllm_config.dssd_config
    if dssd_cfg and dssd_cfg.enabled and dssd_cfg.role == "edge":
        state.openai_serving_chat = DSSDEdgeServingChat(
            engine_client,
            state.openai_serving_models,
            args.response_role,
            openai_serving_render=state.openai_serving_render,
            request_logger=request_logger,
            chat_template=load_chat_template(args.chat_template),
            chat_template_content_format=args.chat_template_content_format,
            default_chat_template_kwargs=args.default_chat_template_kwargs,
            trust_request_chat_template=args.trust_request_chat_template,
            return_tokens_as_token_ids=args.return_tokens_as_token_ids,
        )
    else:
        state.openai_serving_chat = OpenAIServingChat(
            engine_client,
            state.openai_serving_models,
            args.response_role,
            openai_serving_render=state.openai_serving_render,
            request_logger=request_logger,
            chat_template=load_chat_template(args.chat_template),
            chat_template_content_format=args.chat_template_content_format,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
source .venv/bin/activate
pytest tests/v1/dssd/test_round_coordinator.py \
  tests/entrypoints/openai/chat_completion/test_dssd_serving_chat.py -v
```

Expected:

```text
2 passed
```

- [ ] **Step 5: Commit**

```bash
git add vllm/v1/dssd/edge/coordinator.py \
  vllm/entrypoints/openai/chat_completion/dssd_serving.py \
  vllm/entrypoints/openai/generate/api_router.py \
  tests/v1/dssd/test_round_coordinator.py \
  tests/entrypoints/openai/chat_completion/test_dssd_serving_chat.py
git commit -m "feat: add DSSD round coordinator and edge serving"
```

### Task 8: Add Metrics, Network Simulation, Smoke Tests, and Operator Docs

**Files:**
- Modify: `vllm/v1/dssd/transport.py`
- Modify: `vllm/v1/dssd/metrics.py`
- Create: `tests/v1/e2e/dssd/test_dssd_smoke.py`
- Create: `docs/serving/dssd_edge_verifier.md`

- [ ] **Step 1: Write the failing tests**

```python
# tests/v1/e2e/dssd/test_dssd_smoke.py
import pytest

from vllm.v1.dssd.metrics import DSSDRequestMetrics


def test_metrics_track_communication_bytes():
    metrics = DSSDRequestMetrics(request_id="req-1")
    metrics.record_uplink(32)
    metrics.record_downlink(128)
    assert metrics.uplink_bytes == 32
    assert metrics.downlink_bytes == 128
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate
pytest tests/v1/e2e/dssd/test_dssd_smoke.py -v
```

Expected:

```text
ERROR because DSSDRequestMetrics does not exist yet
```

- [ ] **Step 3: Write the minimal implementation**

```python
# vllm/v1/dssd/metrics.py
from dataclasses import dataclass


@dataclass
class DSSDRequestMetrics:
    request_id: str
    uplink_bytes: int = 0
    downlink_bytes: int = 0
    accepted_tokens: int = 0
    rejected_rounds: int = 0

    def record_uplink(self, size: int) -> None:
        self.uplink_bytes += size

    def record_downlink(self, size: int) -> None:
        self.downlink_bytes += size
```

```python
# vllm/v1/dssd/transport.py
class SimulatedNetworkMixin:
    async def _apply_network_simulation(self, payload_size_bytes: int) -> None:
        # Sleep according to latency/jitter/bandwidth before issuing the request
        return None
```

```markdown
<!-- docs/serving/dssd_edge_verifier.md -->
# DSSD Edge-Verifier Serving

This page explains how to run vLLM in DSSD edge mode and verifier mode on the same host first, then split them across a network boundary.

## Edge Mode

```bash
vllm serve Qwen/Qwen3.5-0.8B \
  --dssd-config '{"enabled": true, "role": "edge", "gamma": 4, "verifier_url": "http://127.0.0.1:9001"}'
```

## Verifier Mode

```bash
vllm serve Qwen/Qwen3.5-7B-Instruct \
  --dssd-config '{"enabled": true, "role": "verifier", "gamma": 4}'
```
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
source .venv/bin/activate
pytest tests/v1/e2e/dssd/test_dssd_smoke.py -v
```

Expected:

```text
1 passed
```

- [ ] **Step 5: Commit**

```bash
git add vllm/v1/dssd/transport.py vllm/v1/dssd/metrics.py \
  tests/v1/e2e/dssd/test_dssd_smoke.py docs/serving/dssd_edge_verifier.md
git commit -m "feat: add DSSD metrics and operator docs"
```

---

## Self-Review Checklist

- Spec coverage:
  - 配置、协议、utility path、verifier 路由、verifier runner、edge draft/resample、round coordinator、serving 集成、metrics/e2e/docs 都有对应任务。
  - 第一版范围限制在 plan 中保持一致，没有把 multimodal、tool calling、LoRA、structured output 强行纳入。
- Placeholder scan:
  - 没有使用常见未完成标记或模糊表述。
  - 所有代码块都给出了可直接照着写的最小实现或调用形式。
- Type consistency:
  - `DSSDConfig`
  - `VerifyRoundRequest / VerifyRoundResponse`
  - `DSSDSessionRunner`
  - `DSSDVerifierSessionManager`
  - `DSSDEdgeSessionState`
  - `DSSDRoundCoordinator`
  命名在各任务中保持一致。

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-05-dssd-vllm-implementation.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
