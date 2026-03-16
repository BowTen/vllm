# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from vllm.logger import init_logger
from vllm.v1.spec_decode.distributed.protocol import DraftProposal, SamplingMetadata
from vllm.v1.spec_decode.distributed.sampling import (
    SampleResult,
    is_terminal_token,
    sample_from_logits,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


def resolve_runtime_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_torch_dtype(dtype: str | torch.dtype) -> torch.dtype | None:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype == "auto":
        return None
    aliases = {
        "fp16": torch.float16,
        "half": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype in aliases:
        return aliases[dtype]
    return getattr(torch, dtype)


def serialize_probs(probs: torch.Tensor) -> bytes:
    return np.asarray(probs.detach().cpu().numpy(), dtype=np.float32).tobytes()


def deserialize_probs(raw: bytes, vocab_size: int) -> torch.Tensor:
    array = np.frombuffer(raw, dtype=np.float32, count=vocab_size)
    return torch.from_numpy(array.copy())


@dataclass
class IncrementalRuntimeState:
    token_ids: list[int]
    cache: Any | None
    next_logits: torch.Tensor


def _common_prefix_len(lhs: list[int], rhs: list[int]) -> int:
    matched = 0
    for left, right in zip(lhs, rhs):
        if left != right:
            break
        matched += 1
    return matched


def clone_runtime_cache(cache: Any | None) -> Any | None:
    if cache is None:
        return None
    if hasattr(cache, "to_legacy_cache") and hasattr(type(cache), "from_legacy_cache"):
        return type(cache).from_legacy_cache(cache.to_legacy_cache())
    if isinstance(cache, tuple):
        return DynamicCache.from_legacy_cache(cache)
    raise TypeError(f"Unsupported cache type for cloning: {type(cache)!r}")


def normalize_runtime_cache(cache: Any | None) -> Any | None:
    if cache is None:
        return None
    if isinstance(cache, tuple):
        return DynamicCache.from_legacy_cache(cache)
    return cache


class BaseCausalLMRuntime:
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        dtype: torch.dtype | None,
        trust_remote_code: bool,
    ) -> None:
        self.model_name = model_name
        self.device = device
        logger.info("Loading %s on %s", model_name, device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        self.model.to(device)
        self.model.eval()
        self.vocab_size = int(self.model.config.vocab_size)

    def sample_next_token_from_state(
        self,
        state: IncrementalRuntimeState,
        prompt_len: int,
        sampling: SamplingMetadata,
        generator: torch.Generator,
    ) -> SampleResult:
        output_token_ids = state.token_ids[prompt_len:]
        suppress_stops = len(output_token_ids) < sampling.min_tokens
        return sample_from_logits(
            state.next_logits,
            sampling,
            prompt_token_ids=state.token_ids[:prompt_len],
            output_token_ids=output_token_ids,
            generator=generator,
            suppress_stops=suppress_stops,
        )

    def sample_next_token(
        self,
        prefix_token_ids: list[int],
        prompt_len: int,
        sampling: SamplingMetadata,
        generator: torch.Generator,
    ) -> SampleResult:
        return self.sample_next_token_from_state(
            self.build_runtime_state(prefix_token_ids),
            prompt_len=prompt_len,
            sampling=sampling,
            generator=generator,
        )

    def build_runtime_state(
        self, token_ids: list[int]
    ) -> IncrementalRuntimeState:
        cache, next_logits = self._run_tokens(token_ids, cache=None)
        return IncrementalRuntimeState(
            token_ids=list(token_ids),
            cache=cache,
            next_logits=next_logits,
        )

    def sync_runtime_state(
        self,
        state: IncrementalRuntimeState | None,
        token_ids: list[int],
    ) -> IncrementalRuntimeState:
        if state is None:
            return self.build_runtime_state(token_ids)
        if state.token_ids == token_ids:
            return state

        common_prefix_len = _common_prefix_len(state.token_ids, token_ids)
        if common_prefix_len == len(state.token_ids):
            missing = token_ids[common_prefix_len:]
            if missing:
                self.extend_runtime_state(state, missing)
            return state

        return self.build_runtime_state(token_ids)

    def clone_runtime_state(
        self, state: IncrementalRuntimeState
    ) -> IncrementalRuntimeState:
        return IncrementalRuntimeState(
            token_ids=list(state.token_ids),
            cache=clone_runtime_cache(state.cache),
            next_logits=state.next_logits.clone(),
        )

    def extend_runtime_state(
        self,
        state: IncrementalRuntimeState,
        token_ids: list[int],
    ) -> None:
        if not token_ids:
            return
        cache, next_logits = self._run_tokens(token_ids, cache=state.cache)
        state.token_ids.extend(token_ids)
        state.cache = cache
        state.next_logits = next_logits

    def advance_runtime_state(
        self,
        state: IncrementalRuntimeState,
        token_id: int,
    ) -> None:
        self.extend_runtime_state(state, [token_id])

    def _run_tokens(
        self,
        token_ids: list[int],
        cache: Any | None,
    ) -> tuple[Any | None, torch.Tensor]:
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        inputs = torch.tensor([token_ids], device=self.device, dtype=torch.long)
        kwargs: dict[str, Any] = {
            "input_ids": inputs,
            "use_cache": True,
        }
        if cache is not None:
            kwargs["past_key_values"] = cache
        with torch.inference_mode():
            outputs = self.model(**kwargs)
        next_logits = outputs.logits[0, -1].detach()
        next_cache = normalize_runtime_cache(outputs.past_key_values)
        return next_cache, next_logits


@dataclass
class DraftProposalOutput:
    proposal: DraftProposal
    stopped: bool


class EdgeDraftRunner(BaseCausalLMRuntime):
    def __init__(self, vllm_config: "VllmConfig") -> None:
        spec_config = vllm_config.speculative_config
        assert spec_config is not None
        draft_model_config = spec_config.draft_model_config
        assert draft_model_config is not None
        super().__init__(
            model_name=draft_model_config.model,
            device=resolve_runtime_device(spec_config.draft_device),
            dtype=resolve_torch_dtype(draft_model_config.dtype),
            trust_remote_code=draft_model_config.trust_remote_code,
        )
        self.num_speculative_tokens = spec_config.num_speculative_tokens
        self._lock = asyncio.Lock()
        self._sessions: dict[str, IncrementalRuntimeState] = {}

    def close_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def clear_sessions(self) -> None:
        self._sessions.clear()

    async def propose(
        self,
        session_id: str,
        proposal_id: int,
        base_version: int,
        accepted_prefix_token_ids: list[int],
        prompt_len: int,
        sampling: SamplingMetadata,
        generator: torch.Generator,
    ) -> DraftProposalOutput:
        async with self._lock:
            return await asyncio.to_thread(
                self._propose_sync,
                session_id,
                proposal_id,
                base_version,
                accepted_prefix_token_ids,
                prompt_len,
                sampling,
                generator,
            )

    def _propose_sync(
        self,
        session_id: str,
        proposal_id: int,
        base_version: int,
        accepted_prefix_token_ids: list[int],
        prompt_len: int,
        sampling: SamplingMetadata,
        generator: torch.Generator,
    ) -> DraftProposalOutput:
        draft_token_ids: list[int] = []
        draft_token_probs: list[float] = []
        remaining = None
        current_output_len = len(accepted_prefix_token_ids) - prompt_len
        if sampling.max_tokens is not None:
            remaining = max(0, sampling.max_tokens - current_output_len)
        max_steps = self.num_speculative_tokens
        if remaining is not None:
            max_steps = min(max_steps, remaining)

        stopped = max_steps < self.num_speculative_tokens
        runtime_state = self.sync_runtime_state(
            self._sessions.get(session_id),
            accepted_prefix_token_ids,
        )
        self._sessions[session_id] = runtime_state
        temp_state = self.clone_runtime_state(runtime_state)
        for _ in range(max_steps):
            result = self.sample_next_token_from_state(
                temp_state, prompt_len, sampling, generator
            )
            draft_token_ids.append(result.token_id)
            draft_token_probs.append(result.token_prob)
            self.advance_runtime_state(temp_state, result.token_id)
            output_len_after = len(temp_state.token_ids) - prompt_len
            if is_terminal_token(result.token_id, sampling, output_len_after):
                stopped = True
                break

        return DraftProposalOutput(
            proposal=DraftProposal(
                session_id=session_id,
                proposal_id=proposal_id,
                base_version=base_version,
                accepted_prefix_len=len(accepted_prefix_token_ids),
                draft_token_ids=draft_token_ids,
                draft_token_probs=draft_token_probs,
                draft_stopped=stopped,
            ),
            stopped=stopped,
        )
