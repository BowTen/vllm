from dataclasses import dataclass
from typing import Any

import torch

from experiments.dssd_correctness.sampling import (
    DSSDReferenceSamplingConfig,
    apply_reference_logits_processors,
    sample_from_processed_logits,
    sample_vllm_v1_random,
)


@dataclass(frozen=True)
class DSSDReferenceRound:
    draft_token_ids: list[int]
    accepted_len: int
    committed_token_ids: list[int]


@dataclass(frozen=True)
class DSSDReferenceOutput:
    case_id: str
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    output_text: str
    seed: int
    sampling: DSSDReferenceSamplingConfig
    rounds: list[DSSDReferenceRound]


def _model_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device


def _normalize_eos_token_ids(eos_token_id: Any) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    if isinstance(eos_token_id, (list, tuple, set)):
        return {int(token_id) for token_id in eos_token_id}
    return {int(eos_token_id)}


def _eos_token_ids(tokenizer: Any, model: Any | None = None) -> set[int]:
    eos_token_ids = _normalize_eos_token_ids(
        getattr(tokenizer, "eos_token_id", None)
    )
    generation_config = getattr(model, "generation_config", None)
    eos_token_ids.update(
        _normalize_eos_token_ids(
            getattr(generation_config, "eos_token_id", None)
        )
    )
    return eos_token_ids


def _first_eos_index(token_ids: list[int], eos_token_ids: set[int]) -> int | None:
    return next(
        (
            index
            for index, token_id in enumerate(token_ids)
            if token_id in eos_token_ids
        ),
        None,
    )


def _logits_for_sampling(
    logits: torch.Tensor, config: DSSDReferenceSamplingConfig
) -> torch.Tensor:
    if config.temperature <= 0:
        return logits.to(dtype=torch.float32).clone()
    return apply_reference_logits_processors(logits, config)


def _next_logits(
    model: Any, input_ids: list[int], device: torch.device | str | None = None
) -> torch.Tensor:
    if not input_ids:
        raise ValueError("input_ids must contain at least one token")
    if hasattr(model, "next_logits"):
        return model.next_logits(input_ids)
    model_device = torch.device(device) if device is not None else _model_device(model)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=model_device)
    with torch.inference_mode():
        return model(input_ids=input_tensor).logits[:, -1, :]


def _verify_logits(
    model: Any,
    token_ids: list[int],
    draft_len: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    if draft_len < 1:
        raise ValueError("draft_len must be at least 1")
    if len(token_ids) < draft_len + 1:
        raise ValueError(
            "token_ids must include a non-empty confirmed prefix plus draft tokens"
        )
    if hasattr(model, "verify_logits"):
        return model.verify_logits(token_ids, draft_len)

    model_device = torch.device(device) if device is not None else _model_device(model)
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=model_device)
    with torch.inference_mode():
        logits = model(input_ids=input_tensor).logits

    # Causal LM logits at position i predict token i + 1. For
    # confirmed + draft, the final draft_len + 1 rows predict draft_0,
    # draft_1, ..., draft_n, and the bonus token after the full draft.
    return logits[:, -(draft_len + 1) :, :]


def _build_output(
    *,
    case_id: str,
    tokenizer: Any,
    prompt_token_ids: list[int],
    output_ids: list[int],
    seed: int,
    config: DSSDReferenceSamplingConfig,
    rounds: list[DSSDReferenceRound],
) -> DSSDReferenceOutput:
    return DSSDReferenceOutput(
        case_id=case_id,
        prompt_token_ids=list(prompt_token_ids),
        output_token_ids=list(output_ids),
        output_text=tokenizer.decode(output_ids, skip_special_tokens=True),
        seed=seed,
        sampling=config,
        rounds=rounds,
    )


def generate_dssd_reference(
    *,
    case_id: str,
    target_model: Any,
    draft_model: Any,
    tokenizer: Any,
    prompt_token_ids: list[int],
    config: DSSDReferenceSamplingConfig,
    seed: int,
) -> DSSDReferenceOutput:
    if config.max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    if config.gamma < 1:
        raise ValueError("gamma must be at least 1")
    if not prompt_token_ids:
        raise ValueError("prompt_token_ids must contain at least one token")

    device = _model_device(target_model)
    target_generator = torch.Generator(device=device).manual_seed(seed)
    draft_generator = torch.Generator(device=device).manual_seed(seed)
    eos_token_ids = _eos_token_ids(tokenizer, target_model)

    confirmed = list(prompt_token_ids)
    output_ids: list[int] = []
    rounds: list[DSSDReferenceRound] = []

    target_logits = _next_logits(target_model, confirmed, device)
    target_processed = _logits_for_sampling(target_logits, config)
    bootstrap_token_id, _ = sample_from_processed_logits(
        target_processed, config, target_generator
    )
    confirmed.append(bootstrap_token_id)
    output_ids.append(bootstrap_token_id)

    if (
        len(output_ids) >= config.max_tokens
        or (
            not config.ignore_eos
            and bootstrap_token_id in eos_token_ids
        )
    ):
        return _build_output(
            case_id=case_id,
            tokenizer=tokenizer,
            prompt_token_ids=prompt_token_ids,
            output_ids=output_ids,
            seed=seed,
            config=config,
            rounds=rounds,
        )

    while len(output_ids) < config.max_tokens:
        draft_prefix = list(confirmed)
        draft_token_ids: list[int] = []
        draft_q_values: list[float] = []
        draft_processed_rows: list[torch.Tensor] = []

        for _ in range(config.gamma):
            draft_logits = _next_logits(draft_model, draft_prefix, device)
            draft_processed = _logits_for_sampling(draft_logits, config)
            draft_token_id, draft_q_value = sample_from_processed_logits(
                draft_processed, config, draft_generator
            )
            draft_token_ids.append(draft_token_id)
            draft_q_values.append(draft_q_value)
            draft_processed_rows.append(draft_processed[0])
            draft_prefix.append(draft_token_id)

        target_verify_logits = _verify_logits(
            target_model,
            confirmed + draft_token_ids,
            len(draft_token_ids),
            device,
        )
        target_processed_rows = _logits_for_sampling(
            target_verify_logits[0], config
        )

        accepted_len = 0
        if config.temperature <= 0:
            for index, draft_token_id in enumerate(draft_token_ids):
                target_token_id = int(
                    target_processed_rows[index].argmax(dim=-1).item()
                )
                if target_token_id != draft_token_id:
                    break
                accepted_len += 1
        else:
            uniforms = torch.rand(
                len(draft_token_ids),
                device=device,
                generator=target_generator,
            )
            for index, draft_token_id in enumerate(draft_token_ids):
                p_probs = target_processed_rows[index].softmax(
                    dim=-1, dtype=torch.float32
                )
                p_value = float(p_probs[draft_token_id].item())
                q_value = max(draft_q_values[index], torch.finfo(torch.float32).tiny)
                uniform = float(uniforms[index].item())
                if p_value <= uniform * q_value:
                    break
                accepted_len += 1

        committed_this_round = draft_token_ids[:accepted_len]
        if accepted_len == len(draft_token_ids):
            bonus_token_id, _ = sample_from_processed_logits(
                target_processed_rows[-1], config, target_generator
            )
            committed_this_round.append(bonus_token_id)
        elif config.temperature <= 0:
            rejected_row = target_processed_rows[accepted_len]
            committed_this_round.append(int(rejected_row.argmax(dim=-1).item()))
        else:
            rejected_row = target_processed_rows[accepted_len]
            p_probs = rejected_row.softmax(dim=-1, dtype=torch.float32)
            q_probs = draft_processed_rows[accepted_len].softmax(
                dim=-1, dtype=torch.float32
            )
            residual_probs = torch.clamp(p_probs - q_probs, min=0)
            residual_norm = residual_probs.sum()
            if float(residual_norm.item()) > 0:
                recovery_probs = residual_probs / residual_norm
            else:
                recovery_probs = p_probs
            recovery_token_id = int(
                sample_vllm_v1_random(
                    recovery_probs.unsqueeze(0), draft_generator
                ).item()
            )
            committed_this_round.append(recovery_token_id)

        remaining_tokens = config.max_tokens - len(output_ids)
        committed_this_round = committed_this_round[:remaining_tokens]
        should_stop = False
        if not config.ignore_eos:
            eos_index = _first_eos_index(committed_this_round, eos_token_ids)
            if eos_index is not None:
                committed_this_round = committed_this_round[: eos_index + 1]
                should_stop = True

        recorded_accepted_len = min(accepted_len, len(committed_this_round))
        confirmed.extend(committed_this_round)
        output_ids.extend(committed_this_round)
        rounds.append(
            DSSDReferenceRound(
                draft_token_ids=draft_token_ids,
                accepted_len=recorded_accepted_len,
                committed_token_ids=list(committed_this_round),
            )
        )

        if should_stop:
            break

    return _build_output(
        case_id=case_id,
        tokenizer=tokenizer,
        prompt_token_ids=prompt_token_ids,
        output_ids=output_ids,
        seed=seed,
        config=config,
        rounds=rounds,
    )
