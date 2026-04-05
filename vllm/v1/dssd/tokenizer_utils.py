# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import hashlib
import json
from typing import Any


def compute_tokenizer_fingerprints(tokenizer: Any) -> tuple[str, str]:
    vocab = _extract_vocab(tokenizer)
    if vocab is None:
        return "unavailable", "unavailable"

    tokenizer_payload = {
        "class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}",
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }
    tokenizer_hash = hashlib.sha256(
        json.dumps(tokenizer_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    vocab_hash = hashlib.sha256(
        json.dumps(sorted(vocab.items()), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    return tokenizer_hash, vocab_hash


def _extract_vocab(tokenizer: Any) -> dict[str, int] | None:
    getter = getattr(tokenizer, "get_vocab", None)
    if callable(getter):
        try:
            vocab = getter()
        except Exception:
            vocab = None
        if isinstance(vocab, dict):
            return {str(key): int(value) for key, value in vocab.items()}
    vocab = getattr(tokenizer, "vocab", None)
    if isinstance(vocab, dict):
        return {str(key): int(value) for key, value in vocab.items()}
    return None
