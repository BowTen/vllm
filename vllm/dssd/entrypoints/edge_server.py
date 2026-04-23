from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_DEFAULT_SERVICE_FACTORY = (
    "vllm.dssd.entrypoints.runtime_factory.build_real_edge_service"
)
_RAW_SAMPLING_PARAMS_ENV = "VLLM_DSSD_TEST_EDGE_SERVER_RAW_SAMPLING_PARAMS"
_DEFAULT_MAX_MODEL_LEN = 64
_DEFAULT_GPU_MEMORY_UTILIZATION = 0.01
_DEFAULT_KV_CACHE_MEMORY_BYTES = None
_DEFAULT_MAX_NUM_BATCHED_TOKENS = 64
_DEFAULT_MAX_NUM_SEQS = 2
_allow_raw_sampling_params = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a persistent DSSD edge server."
    )
    parser.add_argument("--service-factory", default=_DEFAULT_SERVICE_FACTORY)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ready-file")
    parser.add_argument("--verifier-url", required=True)
    parser.add_argument("--eos-token-id", type=int, required=True)
    parser.add_argument("--gamma", type=int, required=True)
    _add_runtime_args(parser)
    parser.add_argument(
        "--model-runner-version",
        choices=("v1", "v2"),
        default="v1",
    )
    return parser


def main() -> int:
    global _allow_raw_sampling_params

    args = build_parser().parse_args()
    _allow_raw_sampling_params = _should_decode_raw_sampling_params(
        args.service_factory)
    service_factory = _resolve_obj_by_qualname(args.service_factory)
    edge_service, cleanup = _normalize_service_factory_result(service_factory(args))
    server = _build_server(
        host=args.host,
        port=args.port,
        edge_service=edge_service,
    )
    exit_code = None
    try:
        if args.ready_file:
            Path(args.ready_file).write_text(
                json.dumps({
                    "server_url":
                    f"http://{server.server_address[0]}:{server.server_address[1]}"
                })
            )
        server.serve_forever()
        exit_code = 0
    except KeyboardInterrupt:
        exit_code = 0
    finally:
        server.server_close()
        if cleanup is not None:
            cleanup()
    if exit_code is not None:
        _hard_exit(exit_code)
    return 0


def _build_server(*, host: str, port: int, edge_service) -> ThreadingHTTPServer:
    execution_lock = threading.Lock()

    class EdgeHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            try:
                if self.path != "/generate":
                    self.send_error(404, "unknown path")
                    return
                payload = _load_json(self.rfile.read(_content_length(self)))
                request_kwargs = _edge_generate_request_from_payload(payload)
                with execution_lock:
                    output_ids = edge_service.generate(**request_kwargs)
                self._send_json(
                    _edge_generate_response_to_payload(
                        req_id=request_kwargs["req_id"],
                        output_ids=list(output_ids),
                    ))
            except Exception as exc:
                self._send_json(
                    {
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                    status=500,
                )

        def log_message(self, format: str, *args) -> None:
            return None

        def _send_json(self, payload: dict, *, status: int = 200) -> None:
            body = _dump_json(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer((host, port), EdgeHandler)


def _content_length(handler: BaseHTTPRequestHandler) -> int:
    return int(handler.headers.get("Content-Length", "0"))


def _add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-model-len", type=int, default=_DEFAULT_MAX_MODEL_LEN)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=_DEFAULT_GPU_MEMORY_UTILIZATION,
    )
    parser.add_argument(
        "--kv-cache-memory-bytes",
        type=int,
        default=_DEFAULT_KV_CACHE_MEMORY_BYTES,
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=_DEFAULT_MAX_NUM_BATCHED_TOKENS,
    )
    parser.add_argument("--max-num-seqs", type=int, default=_DEFAULT_MAX_NUM_SEQS)
    parser.add_argument(
        "--enforce-eager",
        dest="enforce_eager",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-enforce-eager",
        dest="enforce_eager",
        action="store_false",
    )
    parser.add_argument(
        "--async-scheduling",
        action="store_true",
        default=False,
    )


def _resolve_obj_by_qualname(qualname: str):
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


def _dump_json(payload: dict) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _load_json(payload: bytes) -> dict:
    return json.loads(payload.decode("utf-8"))


def _edge_generate_request_from_payload(payload: dict) -> dict:
    if _allow_raw_sampling_params:
        sampling_params = SimpleNamespace(**dict(payload["sampling_params"]))
        return {
            "req_id": payload["req_id"],
            "prompt_token_ids": list(payload["prompt_token_ids"]),
            "sampling_params": sampling_params,
            "lora_request": payload.get("lora_request"),
        }

    from vllm.sampling_params import SamplingParams
    import msgspec

    return {
        "req_id": payload["req_id"],
        "prompt_token_ids": list(payload["prompt_token_ids"]),
        "sampling_params": msgspec.convert(
            payload["sampling_params"],
            type=SamplingParams,
        ),
        "lora_request": _decode_lora_request(payload.get("lora_request")),
    }


def _edge_generate_response_to_payload(*, req_id: str, output_ids: list[int]) -> dict:
    return {
        "req_id": req_id,
        "output_ids": list(output_ids),
    }


def _decode_lora_request(payload: Any):
    if payload is None:
        return None

    import msgspec
    from vllm.lora.request import LoRARequest

    return msgspec.convert(payload, type=LoRARequest)


def _should_decode_raw_sampling_params(service_factory: str) -> bool:
    return (os.environ.get(_RAW_SAMPLING_PARAMS_ENV) == "1"
            and service_factory.startswith("tests."))


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
