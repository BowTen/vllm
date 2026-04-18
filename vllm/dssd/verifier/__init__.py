from importlib import import_module
from typing import TYPE_CHECKING, Any

from .types import (
    VerifierOpenSessionResult,
    VerifierRoundRequest,
    VerifierRoundResult,
    VerifierRoundState,
    VerifierSession,
)

if TYPE_CHECKING:
    from .engine import VerifierDecodeEngine
    from .engine_v1 import VerifierDecodeEngineV1
    from .sampler import DSSDVerifierSampler
    from .sampler_v1 import DSSDVerifierSamplerV1
    from .scheduler import VerifierSchedulerAdapter
    from .state_bridge import VerifierStateBridge
    from .state_bridge_v1 import VerifierStateBridgeV1

__all__ = [
    "DSSDVerifierSampler",
    "DSSDVerifierSamplerV1",
    "VerifierDecodeEngine",
    "VerifierDecodeEngineV1",
    "VerifierOpenSessionResult",
    "VerifierRoundRequest",
    "VerifierRoundResult",
    "VerifierRoundState",
    "VerifierSchedulerAdapter",
    "VerifierSession",
    "VerifierStateBridge",
    "VerifierStateBridgeV1",
]

_LAZY_EXPORTS = {
    "DSSDVerifierSampler": (".sampler", "DSSDVerifierSampler"),
    "DSSDVerifierSamplerV1": (".sampler_v1", "DSSDVerifierSamplerV1"),
    "VerifierDecodeEngine": (".engine", "VerifierDecodeEngine"),
    "VerifierDecodeEngineV1": (".engine_v1", "VerifierDecodeEngineV1"),
    "VerifierSchedulerAdapter": (
        ".scheduler",
        "VerifierSchedulerAdapter",
    ),
    "VerifierStateBridge": (".state_bridge", "VerifierStateBridge"),
    "VerifierStateBridgeV1": (".state_bridge_v1", "VerifierStateBridgeV1"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
