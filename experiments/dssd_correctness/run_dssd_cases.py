import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


_SAMPLING_FIELDS = (
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "seed",
    "ignore_eos",
)


_SAMPLING_DEFAULTS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": -1,
    "seed": 0,
    "ignore_eos": True,
}


def _iter_cases(path: Path) -> list[tuple[int, dict[str, Any]]]:
    cases: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            cases.append((line_number, json.loads(stripped)))
    return cases


def _case_id(case: dict[str, Any], line_number: int) -> str:
    return str(case.get("case_id", f"case-{line_number}"))


def _prompt_token_ids(case: dict[str, Any], case_id: str) -> list[int]:
    if "prompt_token_ids" in case:
        return [int(token_id) for token_id in case["prompt_token_ids"]]
    if "prompt" in case:
        raise ValueError(
            "run_dssd_cases needs prompt_token_ids because the edge /generate "
            "endpoint is token-id based. The reference CLI can tokenize prompt; "
            "for exact DSSD compare, users should reuse those prompt ids/cases."
        )
    raise ValueError(f"case {case_id} must define prompt_token_ids")


def _sampling_source(case: dict[str, Any]) -> dict[str, Any]:
    sampling = case.get("sampling")
    if isinstance(sampling, dict):
        return sampling
    return case


def _sampling_params(case: dict[str, Any], case_id: str) -> dict[str, Any]:
    sampling_source = _sampling_source(case)
    if "max_tokens" not in sampling_source:
        raise ValueError(f"case {case_id} is missing required field: max_tokens")
    sampling_params = {"max_tokens": int(sampling_source["max_tokens"])}
    for field, default in _SAMPLING_DEFAULTS.items():
        sampling_params[field] = case.get(field, sampling_source.get(field, default))
    return sampling_params


def _request_payload(
    case: dict[str, Any],
    *,
    case_id: str,
    prompt_token_ids: list[int],
    sampling_params: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "req_id": str(case.get("req_id", case_id)),
        "prompt_token_ids": prompt_token_ids,
        "sampling_params": sampling_params,
    }
    if "lora_request" in case:
        payload["lora_request"] = case["lora_request"]
    return payload


def _post_json(endpoint: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"DSSD edge request failed with HTTP {exc.code}: {error_body}"
        ) from exc
    return json.loads(body.decode("utf-8"))


def _output_record(
    case: dict[str, Any],
    *,
    case_id: str,
    payload: dict[str, Any],
    prompt_token_ids: list[int],
    sampling_params: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "case_id": case_id,
        "req_id": payload["req_id"],
        "prompt_token_ids": prompt_token_ids,
        "output_token_ids": list(response["output_ids"]),
        "seed": case.get("seed"),
        "sampling": sampling_params,
        "raw_response": response,
    }
    sampling_source = _sampling_source(case)
    if "gamma" in sampling_source:
        record["gamma"] = sampling_source["gamma"]
    elif "gamma" in case:
        record["gamma"] = case["gamma"]
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DSSD correctness cases against an edge /generate endpoint."
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8000/generate",
    )
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", default=600.0, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for line_number, case in _iter_cases(args.cases):
            case_id = _case_id(case, line_number)
            prompt_token_ids = _prompt_token_ids(case, case_id)
            sampling_params = _sampling_params(case, case_id)
            payload = _request_payload(
                case,
                case_id=case_id,
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
            )
            response = _post_json(args.endpoint, payload, args.timeout)
            record = _output_record(
                case,
                case_id=case_id,
                payload=payload,
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
                response=response,
            )
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
