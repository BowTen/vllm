from __future__ import annotations

import torch

from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler

from .types import VerifierRoundRequest, VerifierRoundResult


class DSSDVerifierSampler:
    """DSSD verifier 端的自定义采样尾段。

    目标不是替换 vLLM 整条采样链，而是在 verifier round 里：
    1. 复用 vLLM 的 processed logits 语义。
    2. 自己做 accept / reject。
    3. 只把 committed_token + accepted prefix 交给 postprocess。
    4. 把 bonus token 或 rejected logits 旁路返回给 edge。
    """

    def __init__(
        self,
        sampler: Sampler,
        num_speculative_steps: int,
    ) -> None:
        self.sampler = sampler
        self.num_speculative_steps = num_speculative_steps

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
        request: VerifierRoundRequest,
    ) -> tuple[SamplerOutput, VerifierRoundResult]:
        processed_logits = self.apply_sampling_params(logits, input_batch)
        accepted_len = self.compute_accept_length(
            processed_logits, request, input_batch
        )

        sampled_token_ids = self.build_postprocess_tokens(request, accepted_len)
        sampler_output = SamplerOutput(
            sampled_token_ids=sampled_token_ids,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=self.build_num_sampled(input_batch, accepted_len),
        )

        if accepted_len == len(request.draft_token_ids):
            bonus_token_id = self.sample_bonus_token(
                processed_logits, request, input_batch
            )
            result = VerifierRoundResult(
                req_id=request.req_id,
                accepted_len=accepted_len,
                bonus_token_id=bonus_token_id,
            )
        else:
            result = VerifierRoundResult(
                req_id=request.req_id,
                accepted_len=accepted_len,
                rejected_step=accepted_len,
                rejected_target_logits=self.extract_rejected_target_logits(
                    processed_logits,
                    request,
                    accepted_len,
                    input_batch,
                ),
            )

        return sampler_output, result

    def apply_sampling_params(
        self,
        logits: torch.Tensor,
        input_batch: InputBatch,
    ) -> torch.Tensor:
        pos = input_batch.positions[input_batch.logits_indices]
        input_ids = input_batch.input_ids[input_batch.logits_indices]
        return self.sampler.apply_sampling_params(
            logits,
            input_batch.expanded_idx_mapping,
            input_batch.idx_mapping_np,
            pos,
            input_ids,
            input_batch.expanded_local_pos,
        )

    def compute_accept_length(
        self,
        processed_logits: torch.Tensor,
        request: VerifierRoundRequest,
        input_batch: InputBatch,
    ) -> int:
        _ = input_batch
        accepted_len = 0
        eps = 1e-20
        for step, token_id in enumerate(request.draft_token_ids):
            logits_row = processed_logits[step]
            log_p = logits_row[token_id] - torch.logsumexp(logits_row, dim=-1)
            p_value = torch.exp(log_p).item()
            q_value = max(request.draft_q_values[step], eps)
            accept_prob = min(1.0, p_value / q_value)
            rand_u = self._sample_accept_uniform(input_batch, step)
            if rand_u < accept_prob:
                accepted_len += 1
                continue
            break
        return accepted_len

    def _sample_accept_uniform(
        self,
        input_batch: InputBatch,
        step: int,
    ) -> float:
        # 真实 vLLM 在 Triton kernel 里走的是 tl.rand(seed, pos) 语义。
        # demo 里用一个可读的 CPU fallback 保留“同一 request、同一 position
        # -> 同一个随机数”的核心约束。
        pos = int(input_batch.positions[input_batch.logits_indices][step].item())
        req_idx = int(input_batch.expanded_idx_mapping[step].item())
        seed = int(self.sampler.sampling_states.seeds.gpu[req_idx].item())

        mixed_seed = (
            (seed ^ 0x9E3779B97F4A7C15) + (pos + 1) * 0xBF58476D1CE4E5B9
        ) & 0xFFFFFFFFFFFFFFFF
        generator = torch.Generator(device="cpu")
        generator.manual_seed(mixed_seed)
        return float(torch.rand((), generator=generator).item())

    def build_postprocess_tokens(
        self,
        request: VerifierRoundRequest,
        accepted_len: int,
    ) -> torch.Tensor:
        # postprocess() 只应该看到“本轮新生成的输出 token”，
        # 也就是 accepted draft prefix；committed_token 是本轮输入，
        # 不应再出现在 sampled_token_ids 里。
        sampled = torch.full(
            (1, self.num_speculative_steps + 1),
            fill_value=-1,
            dtype=torch.int64,
        )
        if accepted_len > 0:
            accepted = torch.tensor(
                request.draft_token_ids[:accepted_len],
                dtype=torch.int64,
            )
            sampled[0, :accepted_len] = accepted
        return sampled

    def sample_bonus_token(
        self,
        processed_logits: torch.Tensor,
        request: VerifierRoundRequest,
        input_batch: InputBatch,
    ) -> int:
        bonus_start = len(request.draft_token_ids)
        bonus_logits = processed_logits[bonus_start : bonus_start + 1]
        if bonus_logits.shape[0] != 1:
            raise RuntimeError("bonus logits 行数不正确")

        # 这里仍复用 vLLM 的 gumbel_sample，而不是自己写 multinomial。
        # 这样更贴近真实采样语义。
        pos = input_batch.positions[input_batch.logits_indices][
            bonus_start : bonus_start + 1
        ]
        req_idx = input_batch.expanded_idx_mapping[
            bonus_start : bonus_start + 1
        ]
        sampled = gumbel_sample(
            bonus_logits,
            req_idx,
            self.sampler.sampling_states.temperature.gpu,
            self.sampler.sampling_states.seeds.gpu,
            pos,
            apply_temperature=False,
        )
        return int(sampled.item())

    def extract_rejected_target_logits(
        self,
        processed_logits: torch.Tensor,
        request: VerifierRoundRequest,
        accepted_len: int,
        input_batch: InputBatch,
    ) -> torch.Tensor | None:
        _ = request
        _ = input_batch
        if accepted_len >= processed_logits.shape[0]:
            return None
        return processed_logits[accepted_len].detach().clone()

    def build_num_sampled(
        self,
        input_batch: InputBatch,
        accepted_len: int,
    ) -> torch.Tensor:
        return torch.tensor(
            [accepted_len],
            dtype=torch.int32,
            device=input_batch.seq_lens.device,
        )

    def build_num_rejected(
        self,
        input_batch: InputBatch,
        accepted_len: int,
        draft_len: int,
    ) -> torch.Tensor:
        return torch.tensor(
            [draft_len - accepted_len],
            dtype=torch.int32,
            device=input_batch.seq_lens.device,
        )
