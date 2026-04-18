from __future__ import annotations

from dataclasses import replace

import torch

from .types import VerifierRoundRequest, VerifierRoundResult


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
        q_values = torch.tensor(
            request.draft_q_values,
            device=processed_target_logits.device,
            dtype=processed_target_logits.dtype,
        ).clamp_min_(torch.finfo(processed_target_logits.dtype).tiny)
        probs = torch.softmax(processed_target_logits[:draft_len], dim=-1)
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
                rejected_target_logits=processed_target_logits[reject_idx].detach(
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
