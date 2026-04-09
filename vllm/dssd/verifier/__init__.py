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

__all__ = [
    "VerifierOpenSessionResult",
    "VerifierRoundRequest",
    "VerifierRoundResult",
    "VerifierRoundState",
    "VerifierSession",
]


def __getattr__(name: str) -> Any:
    if name == "VerifierDecodeEngine":
        from .engine import VerifierDecodeEngine

        return VerifierDecodeEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
