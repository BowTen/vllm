import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from experiments.dssd_correctness.generate_reference import (
    _case_id,
    _iter_cases,
    _sampling_config,
)
from experiments.dssd_correctness.reference import generate_dssd_reference
from experiments.dssd_correctness.vllm_logprobs_backend import VllmLogprobsModel


def _prompt_token_ids(
    case: dict[str, Any],
    tokenizer: Any,
    case_id: str,
) -> list[int]:
    if "prompt_token_ids" in case:
        return [int(token_id) for token_id in case["prompt_token_ids"]]
    if "prompt" not in case:
        raise ValueError(
            f"case {case_id} must define prompt_token_ids or prompt"
        )
    return tokenizer.encode(str(case["prompt"]), add_special_tokens=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DSSD reference JSONL using vLLM full-vocab logprobs."
    )
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    parser.add_argument(
        "--logprobs-mode",
        default="raw_logits",
        choices=("raw_logits", "raw_logprobs"),
        help=(
            "Use raw_logits to match DSSD's internal sampler inputs. "
            "raw_logprobs is mathematically equivalent for sampling but can "
            "introduce extra floating-point noise before residual recovery."
        ),
    )
    parser.add_argument(
        "--no-enforce-eager",
        dest="enforce_eager",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--async-scheduling",
        action="store_true",
        default=False,
    )
    return parser.parse_args()


def _build_model(args: argparse.Namespace, model_path: str) -> VllmLogprobsModel:
    return VllmLogprobsModel.from_model(
        model_path,
        tokenizer=args.tokenizer,
        dtype=args.dtype,
        device=args.device,
        enforce_eager=args.enforce_eager,
        async_scheduling=args.async_scheduling,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=args.trust_remote_code,
        logprobs_mode=args.logprobs_mode,
    )


def main() -> None:
    args = parse_args()
    target_model = _build_model(args, args.target_model)
    draft_model = _build_model(args, args.draft_model)
    tokenizer = target_model.llm.get_tokenizer()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for line_number, case in _iter_cases(args.cases):
            case_id = _case_id(case, line_number)
            config = _sampling_config(case, case_id)
            output = generate_dssd_reference(
                case_id=case_id,
                target_model=target_model,
                draft_model=draft_model,
                tokenizer=tokenizer,
                prompt_token_ids=_prompt_token_ids(case, tokenizer, case_id),
                config=config,
                seed=int(case.get("seed", 0)),
            )
            f.write(json.dumps(dataclasses.asdict(output)) + "\n")


if __name__ == "__main__":
    main()
