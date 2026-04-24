from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from vllm.utils.import_utils import resolve_obj_by_qualname

from .runtime_args import add_runtime_args
from vllm.dssd.transport.http_utils import (
    close_session_ack_to_payload,
    close_session_request_from_payload,
    dump_json,
    edge_generate_response_to_payload,
    load_json,
    open_session_request_from_payload,
    open_session_response_to_payload,
    verify_round_request_from_payload,
    verify_round_response_to_http_payload,
)

_DEFAULT_SERVICE_FACTORY = (
    "vllm.dssd.entrypoints.runtime_factory.build_real_verifier_service"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a DSSD verifier server.")
    parser.add_argument("--service-factory", default=_DEFAULT_SERVICE_FACTORY)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--ready-file")
    add_runtime_args(parser)
    parser.add_argument("--gamma", type=int, default=0)
    parser.add_argument(
        "--model-runner-version",
        choices=("v1", "v2"),
        default="v1",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    service_factory = resolve_obj_by_qualname(args.service_factory)
    verifier_service, cleanup = _normalize_service_factory_result(
        service_factory(args)
    )
    server = _build_server(
        host=args.host,
        port=args.port,
        verifier_service=verifier_service,
    )
    exit_code = None
    try:
        if args.ready_file:
            Path(args.ready_file).write_text(
                json.dumps(
                    {
                        "server_url": (
                            f"http://{server.server_address[0]}:{server.server_address[1]}"
                        )
                    }
                )
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


def _build_server(
    *,
    host: str,
    port: int,
    verifier_service,
) -> ThreadingHTTPServer:
    class VerifierHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            try:
                if self.path.startswith("/calibrate"):
                    self.rfile.read(_content_length(self))
                    response_bytes = _parse_calibration_response_bytes(self.path)
                    self._send_raw(b"0" * response_bytes)
                    return
                payload = load_json(self.rfile.read(_content_length(self)))
                if self.path == "/open_session":
                    request = open_session_request_from_payload(payload)
                    response = verifier_service.open_session(request)
                    self._send_json(open_session_response_to_payload(response))
                    return
                if self.path == "/generate":
                    request = open_session_request_from_payload(payload)
                    output_ids = verifier_service.generate(request)
                    self._send_json(
                        edge_generate_response_to_payload(
                            req_id=request.req_id,
                            output_ids=list(output_ids),
                        )
                    )
                    return
                if self.path == "/verify_round":
                    request = verify_round_request_from_payload(payload)
                    response = verifier_service.verify_round(request)
                    content_type, body = verify_round_response_to_http_payload(response)
                    self._send_raw(body, content_type=content_type)
                    return
                if self.path == "/close_session":
                    request = close_session_request_from_payload(payload)
                    response = verifier_service.close_session(request.req_id)
                    self._send_json(close_session_ack_to_payload(response))
                    return
                self.send_error(404, "unknown path")
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
            body = dump_json(payload)
            self._send_raw(body, status=status, content_type="application/json")

        def _send_raw(
            self,
            body: bytes,
            *,
            status: int = 200,
            content_type: str = "application/octet-stream",
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer((host, port), VerifierHandler)


def _content_length(handler: BaseHTTPRequestHandler) -> int:
    return int(handler.headers.get("Content-Length", "0"))


def _parse_calibration_response_bytes(path: str) -> int:
    query = parse_qs(urlparse(path).query)
    values = query.get("response_bytes", ["1"])
    response_bytes = int(values[0])
    if response_bytes < 0:
        raise ValueError("response_bytes must be non-negative")
    return response_bytes


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
