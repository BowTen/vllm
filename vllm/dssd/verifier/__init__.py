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
    from .sampler import DSSDVerifierSampler
    from .scheduler import VerifierSchedulerAdapter
    from .state_bridge import VerifierStateBridge

__all__ = [
    "DSSDVerifierSampler",
    "VerifierDecodeEngine",
    "VerifierOpenSessionResult",
    "VerifierRoundRequest",
    "VerifierRoundResult",
    "VerifierRoundState",
    "VerifierSchedulerAdapter",
    "VerifierSession",
    "VerifierStateBridge",
]

_LAZY_EXPORTS = {
    "DSSDVerifierSampler": (".sampler", "DSSDVerifierSampler"),
    "VerifierDecodeEngine": (".engine", "VerifierDecodeEngine"),
    "VerifierSchedulerAdapter": (
        ".scheduler",
        "VerifierSchedulerAdapter",
    ),
    "VerifierStateBridge": (".state_bridge", "VerifierStateBridge"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
