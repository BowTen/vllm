# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
VLLM_DIR = ROOT / "vllm"
DSSD_DIR = VLLM_DIR / "v1" / "dssd"


def _install_package_stub(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_install_package_stub("vllm", VLLM_DIR)
_install_package_stub("vllm.v1", VLLM_DIR / "v1")
_install_package_stub("vllm.v1.dssd", DSSD_DIR)

tokenizer_utils_module = _load_module(
    "vllm.v1.dssd.tokenizer_utils",
    DSSD_DIR / "tokenizer_utils.py",
)

compute_tokenizer_fingerprints = tokenizer_utils_module.compute_tokenizer_fingerprints


class _Tokenizer:
    def __init__(self, vocab: dict[str, int]) -> None:
        self._vocab = vocab
        self.vocab_size = len(vocab)
        self.eos_token_id = max(vocab.values())
        self.bos_token_id = min(vocab.values())

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)


def test_compute_tokenizer_fingerprints_uses_vocab_content():
    left = _Tokenizer({"a": 0, "b": 1})
    right = _Tokenizer({"a": 0, "c": 1})

    left_tokenizer_hash, left_vocab_hash = compute_tokenizer_fingerprints(left)
    right_tokenizer_hash, right_vocab_hash = compute_tokenizer_fingerprints(right)

    assert left_tokenizer_hash != "unavailable"
    assert left_vocab_hash != "unavailable"
    assert left_vocab_hash != right_vocab_hash
    assert left_tokenizer_hash == right_tokenizer_hash
