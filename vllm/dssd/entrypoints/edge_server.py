from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_DEFAULT_SERVICE_FACTORY = (
    "vllm.dssd.entrypoints.runtime_factory.build_real_edge_service"
)


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
    args = build_parser().parse_args()
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
    from .runtime_factory import add_runtime_args

    add_runtime_args(parser)


def _resolve_obj_by_qualname(qualname: str):
    from vllm.utils.import_utils import resolve_obj_by_qualname

    return resolve_obj_by_qualname(qualname)


def _dump_json(payload: dict) -> bytes:
    from vllm.dssd.transport.http_utils import dump_json

    return dump_json(payload)


def _load_json(payload: bytes) -> dict:
    from vllm.dssd.transport.http_utils import load_json

    return load_json(payload)


def _edge_generate_request_from_payload(payload: dict) -> dict:
    from vllm.dssd.transport.http_utils import edge_generate_request_from_payload

    return edge_generate_request_from_payload(payload)


def _edge_generate_response_to_payload(*, req_id: str, output_ids: list[int]) -> dict:
    from vllm.dssd.transport.http_utils import edge_generate_response_to_payload

    return edge_generate_response_to_payload(req_id=req_id, output_ids=output_ids)


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
