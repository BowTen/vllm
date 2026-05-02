from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class VllmLogprobsModel:
    """Full-prefix logits backend built from vLLM full-vocab logprobs.

    The CLI configures vLLM's logprobs API to return raw logits by default.
    raw_logprobs also work as logits-like rows for sampling, but raw logits
    avoid an extra log-softmax round before the reference applies DSSD's
    sampling processors.
    """

    llm: Any
    device: torch.device | str = "cuda"

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        self.vocab_size = int(self.llm.llm_engine.model_config.get_vocab_size())
        self._next_params = self._make_sampling_params(
            logprobs=-1,
            prompt_logprobs=None,
        )
        self._verify_params = self._make_sampling_params(
            logprobs=-1,
            prompt_logprobs=-1,
        )

    @classmethod
    def from_model(
        cls,
        model: str,
        *,
        tokenizer: str | None = None,
        dtype: str = "auto",
        device: str = "cuda",
        enforce_eager: bool = True,
        async_scheduling: bool = False,
        gpu_memory_utilization: float = 0.9,
        kv_cache_memory_bytes: int | None = None,
        max_model_len: int = 1024,
        max_num_batched_tokens: int | None = None,
        max_num_seqs: int | None = None,
        trust_remote_code: bool = False,
        logprobs_mode: str = "raw_logits",
    ) -> "VllmLogprobsModel":
        from vllm import LLM

        kwargs: dict[str, Any] = {}
        if max_num_batched_tokens is not None:
            kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        if max_num_seqs is not None:
            kwargs["max_num_seqs"] = max_num_seqs

        llm = LLM(
            model,
            tokenizer=tokenizer,
            dtype=dtype,
            enforce_eager=enforce_eager,
            async_scheduling=async_scheduling,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
            max_model_len=max_model_len,
            max_logprobs=-1,
            logprobs_mode=logprobs_mode,
            enable_prefix_caching=False,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        return cls(llm=llm, device=device)

    @staticmethod
    def _make_sampling_params(*, logprobs: int, prompt_logprobs: int | None):
        from vllm import SamplingParams

        return SamplingParams(
            max_tokens=1,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            ignore_eos=True,
            detokenize=False,
            skip_special_tokens=False,
        )

    def next_logits(self, input_ids: list[int]) -> torch.Tensor:
        if not input_ids:
            raise ValueError("input_ids must contain at least one token")
        output = self._generate(input_ids, self._next_params)
        sample_logprobs = output.outputs[0].logprobs
        if sample_logprobs is None or len(sample_logprobs) < 1:
            raise RuntimeError("vLLM did not return sample logprobs")
        return self._row_to_tensor(sample_logprobs[0]).unsqueeze(0)

    def verify_logits(self, token_ids: list[int], draft_len: int) -> torch.Tensor:
        if draft_len < 1:
            raise ValueError("draft_len must be at least 1")
        if len(token_ids) < draft_len + 1:
            raise ValueError(
                "token_ids must include a non-empty confirmed prefix plus draft tokens"
            )

        output = self._generate(token_ids, self._verify_params)
        prompt_logprobs = output.prompt_logprobs
        if prompt_logprobs is None:
            raise RuntimeError("vLLM did not return prompt logprobs")
        sample_logprobs = output.outputs[0].logprobs
        if sample_logprobs is None or len(sample_logprobs) < 1:
            raise RuntimeError("vLLM did not return bonus logprobs")

        start = len(token_ids) - draft_len
        rows = []
        for position in range(start, len(token_ids)):
            row = prompt_logprobs[position]
            if row is None:
                raise RuntimeError(
                    f"missing prompt logprobs at token position {position}"
                )
            rows.append(self._row_to_tensor(row))
        rows.append(self._row_to_tensor(sample_logprobs[0]))
        return torch.stack(rows, dim=0).unsqueeze(0)

    def _generate(self, token_ids: list[int], sampling_params: Any):
        from vllm import TokensPrompt

        outputs = self.llm.generate(
            [TokensPrompt(prompt_token_ids=list(token_ids))],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        if len(outputs) != 1:
            raise RuntimeError(f"expected one vLLM output, got {len(outputs)}")
        return outputs[0]

    def _row_to_tensor(self, row: dict[int, Any]) -> torch.Tensor:
        if len(row) != self.vocab_size:
            raise RuntimeError(
                "expected full-vocabulary logprobs; "
                f"got {len(row)} entries for vocab size {self.vocab_size}"
            )
        values = torch.empty(
            self.vocab_size,
            dtype=torch.float32,
            device=self.device,
        )
        for token_id, logprob in row.items():
            values[int(token_id)] = float(logprob.logprob)
        return values
