# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch

from vllm.v1.sample.ops.bad_words import _apply_bad_words_single_batch
from vllm.v1.spec_decode.distributed.protocol import SamplingMetadata

_NEG_INF = torch.finfo(torch.float32).min


@dataclass
class SampleResult:
    token_id: int
    token_prob: float
    probs: torch.Tensor


def is_terminal_token(
    token_id: int,
    sampling: SamplingMetadata,
    output_len_after: int,
) -> bool:
    if output_len_after < sampling.min_tokens:
        return False

    if (
        sampling.eos_token_id is not None
        and not sampling.ignore_eos
        and token_id == sampling.eos_token_id
    ):
        return True

    return token_id in sampling.stop_token_ids


def suppress_terminal_tokens(
    logits: torch.Tensor,
    sampling: SamplingMetadata,
) -> None:
    if sampling.eos_token_id is not None and not sampling.ignore_eos:
        logits[sampling.eos_token_id] = _NEG_INF
    if sampling.stop_token_ids:
        logits[torch.tensor(sampling.stop_token_ids, device=logits.device)] = _NEG_INF


def apply_sampling_transforms(
    logits: torch.Tensor,
    sampling: SamplingMetadata,
    prompt_token_ids: list[int],
    output_token_ids: list[int],
) -> torch.Tensor:
    logits = logits.to(dtype=torch.float32).clone()

    if output_token_ids:
        output_counts = Counter(output_token_ids)
        if sampling.presence_penalty:
            present_ids = torch.tensor(
                list(output_counts.keys()), device=logits.device, dtype=torch.long
            )
            logits[present_ids] -= sampling.presence_penalty

        if sampling.frequency_penalty:
            freq_ids = torch.tensor(
                list(output_counts.keys()), device=logits.device, dtype=torch.long
            )
            freq_vals = torch.tensor(
                list(output_counts.values()), device=logits.device, dtype=logits.dtype
            )
            logits[freq_ids] -= sampling.frequency_penalty * freq_vals

    if sampling.repetition_penalty != 1.0:
        repeated_ids = set(prompt_token_ids)
        repeated_ids.update(output_token_ids)
        if repeated_ids:
            rep_ids = torch.tensor(
                sorted(repeated_ids), device=logits.device, dtype=torch.long
            )
            rep_logits = logits[rep_ids]
            penalty = sampling.repetition_penalty
            rep_logits = torch.where(
                rep_logits > 0, rep_logits / penalty, rep_logits * penalty
            )
            logits[rep_ids] = rep_logits

    if sampling.logit_bias:
        bias_ids = torch.tensor(
            list(sampling.logit_bias.keys()),
            device=logits.device,
            dtype=torch.long,
        )
        bias_vals = torch.tensor(
            list(sampling.logit_bias.values()),
            device=logits.device,
            dtype=logits.dtype,
        )
        logits[bias_ids] += bias_vals

    if sampling.allowed_token_ids is not None:
        allowed = torch.full_like(logits, _NEG_INF)
        allow_ids = torch.tensor(
            sampling.allowed_token_ids, device=logits.device, dtype=torch.long
        )
        allowed[allow_ids] = logits[allow_ids]
        logits = allowed

    if sampling.bad_words_token_ids:
        _apply_bad_words_single_batch(logits, sampling.bad_words_token_ids,
                                      output_token_ids)

    if sampling.temperature > 0:
        logits = logits / sampling.temperature

    return logits


def probs_from_logits(
    logits: torch.Tensor,
    sampling: SamplingMetadata,
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    *,
    suppress_stops: bool,
) -> torch.Tensor:
    transformed = apply_sampling_transforms(
        logits, sampling, prompt_token_ids, output_token_ids
    )
    if suppress_stops:
        suppress_terminal_tokens(transformed, sampling)

    greedy = sampling.temperature <= 0
    filtered = transformed.clone()

    if not greedy and sampling.min_p > 0:
        base_probs = torch.softmax(filtered, dim=-1)
        threshold = base_probs.max() * sampling.min_p
        filtered[base_probs < threshold] = _NEG_INF

    if not greedy and sampling.top_k > 0 and sampling.top_k < filtered.numel():
        threshold = torch.topk(filtered, sampling.top_k).values[-1]
        filtered[filtered < threshold] = _NEG_INF

    if not greedy and sampling.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > sampling.top_p
        if remove.numel() > 0:
            remove[0] = False
        filtered[sorted_indices[remove]] = _NEG_INF

    probs = torch.softmax(filtered, dim=-1)
    if not torch.isfinite(probs).all() or float(probs.sum().item()) == 0.0:
        raise ValueError("Sampling produced an invalid probability distribution.")
    return probs


def sample_from_logits(
    logits: torch.Tensor,
    sampling: SamplingMetadata,
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    generator: torch.Generator,
    suppress_stops: bool,
) -> SampleResult:
    probs = probs_from_logits(
        logits,
        sampling,
        prompt_token_ids,
        output_token_ids,
        suppress_stops=suppress_stops,
    )

    greedy = sampling.temperature <= 0

    if greedy:
        token_id = int(torch.argmax(probs).item())
        return SampleResult(
            token_id=token_id,
            token_prob=float(probs[token_id].item()),
            probs=probs,
        )

    token_id = int(torch.multinomial(probs, 1, generator=generator).item())
    return SampleResult(
        token_id=token_id,
        token_prob=float(probs[token_id].item()),
        probs=probs,
    )


def sample_from_probs(
    probs: torch.Tensor,
    generator: torch.Generator,
    *,
    greedy: bool = False,
) -> int:
    probs = probs.to(dtype=torch.float32)
    total = float(probs.sum().item())
    if total <= 0:
        raise ValueError("Received an empty probability distribution to sample from.")
    if abs(total - 1.0) > 1e-4:
        probs = probs / total
    if greedy:
        return int(torch.argmax(probs).item())
    return int(torch.multinomial(probs, 1, generator=generator).item())


def should_accept_draft_token(
    target_probs: torch.Tensor,
    draft_token_id: int,
    draft_token_prob: float,
    generator: torch.Generator,
    *,
    greedy: bool = False,
) -> bool:
    if draft_token_id < 0 or draft_token_id >= target_probs.numel():
        raise ValueError(
            f"Draft token id {draft_token_id} is outside the target vocabulary."
        )

    if greedy:
        return draft_token_id == int(torch.argmax(target_probs).item())

    if draft_token_prob <= 0:
        return False

    target_prob = float(target_probs[draft_token_id].item())
    accept_prob = max(0.0, min(1.0, target_prob / draft_token_prob))
    if accept_prob <= 0.0:
        return False
    if accept_prob >= 1.0:
        return True
    uniform = float(
        torch.rand((), dtype=torch.float64, generator=generator).item()
    )
    return uniform <= accept_prob


def residual_probs_from_distributions(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> torch.Tensor:
    target_probs = target_probs.to(dtype=torch.float32)
    draft_probs = draft_probs.to(dtype=torch.float32, device=target_probs.device)
    if target_probs.shape != draft_probs.shape:
        raise ValueError(
            "Target and draft probability distributions must share the same shape."
        )
    return (target_probs - draft_probs).clamp_min(0.0)


def sample_recovered_token(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    generator: torch.Generator,
    *,
    greedy: bool = False,
) -> int:
    if greedy:
        return int(torch.argmax(target_probs).item())

    residual = residual_probs_from_distributions(target_probs, draft_probs)
    total = float(residual.sum().item())
    if total <= 0.0:
        raise ValueError(
            "Recovered-token residual distribution is empty; "
            "cannot sample after speculative rejection."
        )
    return sample_from_probs(residual, generator, greedy=False)
