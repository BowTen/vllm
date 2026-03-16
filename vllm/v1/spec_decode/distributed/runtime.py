# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from transformers import AutoModelForCausalLM

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

    def sample_next_token(
        self,
        prefix_token_ids: list[int],
        prompt_len: int,
        sampling: SamplingMetadata,
        generator: torch.Generator,
    ) -> SampleResult:
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        inputs = torch.tensor([prefix_token_ids], device=self.device, dtype=torch.long)
        with torch.inference_mode():
            outputs = self.model(input_ids=inputs, use_cache=False)
        logits = outputs.logits[0, -1]
        output_token_ids = prefix_token_ids[prompt_len:]
        suppress_stops = len(output_token_ids) < sampling.min_tokens
        return sample_from_logits(
            logits,
            sampling,
            prompt_token_ids=prefix_token_ids[:prompt_len],
            output_token_ids=output_token_ids,
            generator=generator,
            suppress_stops=suppress_stops,
        )


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
        prefix = list(accepted_prefix_token_ids)
        for _ in range(max_steps):
            result = self.sample_next_token(prefix, prompt_len, sampling, generator)
            draft_token_ids.append(result.token_id)
            draft_token_probs.append(result.token_prob)
            prefix.append(result.token_id)
            output_len_after = len(prefix) - prompt_len
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
