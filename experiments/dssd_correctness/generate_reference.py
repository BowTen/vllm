import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.dssd_correctness.reference import generate_dssd_reference
from experiments.dssd_correctness.sampling import DSSDReferenceSamplingConfig


def _torch_dtype(dtype: str) -> torch.dtype | str:
    if dtype == "auto":
        return "auto"
    try:
        value = getattr(torch, dtype)
    except AttributeError as err:
        raise ValueError(f"unsupported torch dtype: {dtype}") from err
    if not isinstance(value, torch.dtype):
        raise ValueError(f"unsupported torch dtype: {dtype}")
    return value


def _case_id(case: dict[str, Any], line_number: int) -> str:
    return str(case.get("case_id", f"case-{line_number}"))


def _sampling_config(case: dict[str, Any], case_id: str) -> DSSDReferenceSamplingConfig:
    missing = [field for field in ("max_tokens", "gamma") if field not in case]
    if missing:
        raise ValueError(
            f"case {case_id} is missing required sampling field(s): "
            + ", ".join(missing)
        )

    # Keep experiment cases limited to vLLM-aligned logits processing fields.
    # Transformers-specific penalties are intentionally unsupported here.
    return DSSDReferenceSamplingConfig(
        max_tokens=int(case["max_tokens"]),
        gamma=int(case["gamma"]),
        temperature=float(case.get("temperature", 0.0)),
        top_p=float(case.get("top_p", 1.0)),
        top_k=int(case.get("top_k", -1)),
        ignore_eos=bool(case.get("ignore_eos", True)),
    )


def _prompt_token_ids(
    case: dict[str, Any], tokenizer: Any, case_id: str
) -> list[int]:
    if "prompt_token_ids" in case:
        return [int(token_id) for token_id in case["prompt_token_ids"]]
    if "prompt" not in case:
        raise ValueError(
            f"case {case_id} must define prompt_token_ids or prompt"
        )
    return tokenizer.encode(str(case["prompt"]), add_special_tokens=False)


def _iter_cases(path: Path) -> list[tuple[int, dict[str, Any]]]:
    cases: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            cases.append((line_number, json.loads(stripped)))
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Transformers DSSD reference JSONL outputs."
    )
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--trace-top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_name = args.tokenizer or args.target_model
    dtype = _torch_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, trust_remote_code=args.trust_remote_code
    )
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(args.device)
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).to(args.device)
    target_model.eval()
    draft_model.eval()

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
                trace_path=args.trace_output,
                trace_top_k=args.trace_top_k,
            )
            f.write(json.dumps(dataclasses.asdict(output)) + "\n")


if __name__ == "__main__":
    main()
