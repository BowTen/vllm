from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DSSDReferenceSamplingConfig:
    max_tokens: int = 0
    gamma: int = 0
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    ignore_eos: bool = True


def apply_reference_logits_processors(
    logits: torch.Tensor, config: DSSDReferenceSamplingConfig
) -> torch.Tensor:
    original_shape = logits.shape
    processed = logits.to(dtype=torch.float32).clone()
    if processed.dim() == 0:
        raise ValueError(
            "logits must have at least one dimension for the vocabulary axis"
        )

    if config.temperature > 0 and config.temperature != 1:
        processed = processed / config.temperature

    vocab_size = processed.shape[-1]
    rows = processed.reshape(-1, vocab_size)

    if 0 < config.top_k < vocab_size:
        top_k_threshold = rows.topk(config.top_k, dim=-1).values[:, -1]
        rows = rows.masked_fill(
            rows < top_k_threshold.unsqueeze(dim=-1), -float("inf")
        )

    if config.top_p < 1.0:
        sorted_logits, sorted_indices = rows.sort(dim=-1, descending=False)
        sorted_probs = sorted_logits.softmax(dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        top_p_mask = cumulative_probs <= 1 - config.top_p
        top_p_mask[:, -1] = False
        sorted_logits = sorted_logits.masked_fill(top_p_mask, -float("inf"))
        rows = torch.empty_like(rows).scatter(
            dim=-1, index=sorted_indices, src=sorted_logits
        )

    return rows.reshape(original_shape)


def sample_vllm_v1_random(
    probs: torch.Tensor, generator: torch.Generator | None
) -> torch.Tensor:
    q = torch.empty_like(probs)
    if generator is None:
        q.exponential_()
    else:
        for row in range(probs.shape[0]):
            q[row].exponential_(generator=generator)
    return (probs / q).argmax(dim=-1).view(-1)


def sample_from_processed_logits(
    processed_logits: torch.Tensor,
    config: DSSDReferenceSamplingConfig,
    generator: torch.Generator | None,
) -> tuple[int, float]:
    if processed_logits.dim() == 1:
        logits_row = processed_logits
    elif processed_logits.dim() == 2 and processed_logits.shape[0] == 1:
        logits_row = processed_logits[0]
    else:
        raise ValueError(
            "processed_logits must have shape [vocab] or [1, vocab], "
            f"got {tuple(processed_logits.shape)}"
        )

    if config.temperature <= 0:
        return int(logits_row.argmax(dim=-1).item()), 1.0

    probs = logits_row.softmax(dim=-1, dtype=torch.float32)
    token_id = int(sample_vllm_v1_random(probs.unsqueeze(0), generator).item())
    return token_id, float(probs[token_id].item())
