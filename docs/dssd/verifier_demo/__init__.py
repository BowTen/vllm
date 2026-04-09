"""DSSD verifier 端示意代码。

这套代码的目标不是当前可运行，而是：
1. 让算法逻辑可以在编辑器里顺着方法跳转阅读。
2. 让后续真实实现时，能快速映射到 vLLM 的实际组件和状态。
"""

from .engine import VerifierDecodeEngine
from .sampler import DSSDVerifierSampler
from .scheduler import VerifierSchedulerAdapter
from .service import DSSDVerifierService
from .state_bridge import VerifierStateBridge
from .types import (
    VerifierOpenSessionResult,
    VerifierRoundRequest,
    VerifierRoundResult,
    VerifierSession,
)

__all__ = [
    "DSSDVerifierSampler",
    "DSSDVerifierService",
    "VerifierDecodeEngine",
    "VerifierOpenSessionResult",
    "VerifierRoundRequest",
    "VerifierRoundResult",
    "VerifierSchedulerAdapter",
    "VerifierSession",
    "VerifierStateBridge",
]
