from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from vllm.sampling_params import SamplingParams
from vllm.utils.import_utils import resolve_obj_by_qualname

from .runtime_factory import add_runtime_args

_DEFAULT_SERVICE_FACTORY = (
    "vllm.dssd.entrypoints.runtime_factory.build_real_edge_service"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a DSSD edge request.")
    parser.add_argument("--service-factory", default=_DEFAULT_SERVICE_FACTORY)
    parser.add_argument("--verifier-url", required=True)
    parser.add_argument("--req-id", required=True)
    parser.add_argument("--prompt-token-ids", required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--eos-token-id", type=int, required=True)
    parser.add_argument("--gamma", type=int, required=True)
    add_runtime_args(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    service_factory = resolve_obj_by_qualname(args.service_factory)
    edge_service, cleanup = _normalize_service_factory_result(service_factory(args))
    exit_code = 1
    try:
        prompt_token_ids = _parse_token_ids(args.prompt_token_ids)
        output_ids = edge_service.generate(
            req_id=args.req_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=SamplingParams(
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            ),
        )
        print(json.dumps({"req_id": args.req_id, "output_ids": list(output_ids)}))
        exit_code = 0
    except Exception:
        traceback.print_exc()
    finally:
        if cleanup is not None:
            cleanup()
    _hard_exit(exit_code)
    return exit_code


def _parse_token_ids(raw_value: str) -> list[int]:
    if not raw_value:
        return []
    return [int(part) for part in raw_value.split(",") if part]


def _normalize_service_factory_result(result):
    if isinstance(result, tuple):
        return result
    return result, None


def _hard_exit(code: int) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    raise SystemExit(main())
