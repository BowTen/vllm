from __future__ import annotations

from dataclasses import replace

import torch

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

from .types import VerifierRoundRequest, VerifierRoundResult

_GREEDY_TEMPERATURE_EPS = 1e-5


class DSSDVerifierSamplerV1:
    def __init__(self, sampler) -> None:
        self.sampler = sampler

    def sample_bootstrap(
        self,
        *,
        logits: torch.Tensor,
        sampling_metadata,
    ) -> int:
        sampler_output = self.sampler(
            logits=logits,
            sampling_metadata=replace(sampling_metadata, max_num_logprobs=None),
        )
        return int(sampler_output.sampled_token_ids[0, 0].item())

    def verify_round(
        self,
        *,
        logits: torch.Tensor,
        spec_decode_metadata,
        sampling_metadata,
        request: VerifierRoundRequest,
    ) -> VerifierRoundResult:
        if len(request.draft_token_ids) != len(request.draft_q_values):
            raise ValueError("draft_token_ids 和 draft_q_values 长度不一致")

        draft_len = len(request.draft_token_ids)
        if draft_len == 0:
            bonus_logits = (logits if spec_decode_metadata is None else
                            logits[spec_decode_metadata.bonus_logits_indices])
            bonus_output = self.sampler(
                logits=bonus_logits,
                sampling_metadata=replace(sampling_metadata, max_num_logprobs=None),
                predict_bonus_token=True,
            )
            return VerifierRoundResult(
                req_id=request.req_id,
                accepted_len=0,
                bonus_token_id=int(bonus_output.sampled_token_ids[0, 0].item()),
            )

        target_logits = logits[spec_decode_metadata.target_logits_indices].to(
            torch.float32)
        processed_target_logits = self.sampler.apply_logits_processors(
            target_logits.clone(),
            sampling_metadata,
            False,
        )

        draft_token_ids = torch.tensor(
            request.draft_token_ids,
            device=processed_target_logits.device,
            dtype=torch.int64,
        )
        if self._is_greedy_request(sampling_metadata):
            target_argmax = processed_target_logits[:draft_len].argmax(dim=-1)
            mismatches = torch.nonzero(
                target_argmax != draft_token_ids,
                as_tuple=False,
            ).flatten()
            if mismatches.numel() > 0:
                reject_idx = int(mismatches[0].item())
                return VerifierRoundResult(
                    req_id=request.req_id,
                    accepted_len=reject_idx,
                    rejected_token_id=int(target_argmax[reject_idx].item()),
                )

            bonus_output = self.sampler(
                logits=logits[spec_decode_metadata.bonus_logits_indices],
                sampling_metadata=replace(sampling_metadata, max_num_logprobs=None),
                predict_bonus_token=True,
            )
            return VerifierRoundResult(
                req_id=request.req_id,
                accepted_len=draft_len,
                bonus_token_id=int(bonus_output.sampled_token_ids[0, 0].item()),
            )

        sampling_target_logits = self._apply_random_sampling_processors(
            processed_target_logits,
            sampling_metadata,
        )

        q_values = torch.tensor(
            request.draft_q_values,
            device=sampling_target_logits.device,
            dtype=sampling_target_logits.dtype,
        ).clamp_min_(torch.finfo(sampling_target_logits.dtype).tiny)
        probs = torch.softmax(sampling_target_logits[:draft_len], dim=-1)
        draft_probs = probs.gather(1, draft_token_ids.view(-1, 1)).squeeze(1)

        generator = sampling_metadata.generators.get(0)
        uniforms = torch.rand(
            draft_len,
            device=processed_target_logits.device,
            generator=generator,
        )
        rejected = draft_probs <= (uniforms * q_values)
        reject_indices = torch.nonzero(rejected, as_tuple=False).flatten()

        if reject_indices.numel() > 0:
            reject_idx = int(reject_indices[0].item())
            return VerifierRoundResult(
                req_id=request.req_id,
                accepted_len=reject_idx,
                rejected_target_logits=sampling_target_logits[reject_idx].detach(
                ).clone(),
            )

        bonus_output = self.sampler(
            logits=logits[spec_decode_metadata.bonus_logits_indices],
            sampling_metadata=replace(sampling_metadata, max_num_logprobs=None),
            predict_bonus_token=True,
        )
        return VerifierRoundResult(
            req_id=request.req_id,
            accepted_len=draft_len,
            bonus_token_id=int(bonus_output.sampled_token_ids[0, 0].item()),
        )

    @staticmethod
    def _is_greedy_request(sampling_metadata) -> bool:
        if sampling_metadata.all_greedy:
            return True
        temperature = sampling_metadata.temperature
        if temperature is None:
            return False
        return bool(torch.all(temperature < _GREEDY_TEMPERATURE_EPS).item())

    def _apply_random_sampling_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata,
    ) -> torch.Tensor:
        assert sampling_metadata.temperature is not None
        logits = self.sampler.apply_temperature(
            logits,
            sampling_metadata.temperature,
            sampling_metadata.all_random,
        )

        for processor in sampling_metadata.logitsprocs.argmax_invariant:
            logits = processor.apply(logits)

        top_k = self._expand_sampling_tensor(
            sampling_metadata.top_k,
            logits.shape[0],
        )
        top_p = self._expand_sampling_tensor(
            sampling_metadata.top_p,
            logits.shape[0],
        )
        return apply_top_k_top_p(
            logits,
            top_k,
            top_p,
        )

    @staticmethod
    def _expand_sampling_tensor(
        value: torch.Tensor | None,
        num_rows: int,
    ) -> torch.Tensor | None:
        if value is None or value.shape[0] == num_rows:
            return value
        if value.shape[0] != 1:
            raise ValueError(
                "sampling metadata tensor must have one value or one per row"
            )
        return value.expand(num_rows)
