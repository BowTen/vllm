from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

_DEFAULT_SERVICE_FACTORY = (
    "vllm.dssd.entrypoints.runtime_factory.build_real_edge_service"
)
_RAW_SAMPLING_PARAMS_ENV = "VLLM_DSSD_TEST_EDGE_SERVER_RAW_SAMPLING_PARAMS"
_allow_raw_sampling_params = False


def _load_runtime_arg_helper():
    if __package__:
        from .runtime_args import add_runtime_args

        return add_runtime_args

    module_path = Path(__file__).with_name("runtime_args.py")
    spec = importlib.util.spec_from_file_location(
        "dssd_edge_runtime_args",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.add_runtime_args


_add_runtime_args = _load_runtime_arg_helper()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a persistent DSSD edge server."
    )
    parser.add_argument("--service-factory", default=_DEFAULT_SERVICE_FACTORY)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6006)
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


def _build_server(
    *,
    host: str,
    port: int,
    edge_service,
) -> ThreadingHTTPServer:
    execution_lock = threading.Lock()

    class EdgeHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            try:
                if self.path != "/":
                    self.send_error(404, "unknown path")
                    return
                self._send_raw(
                    _load_benchmark_page(),
                    content_type="text/html; charset=utf-8",
                )
            except Exception as exc:
                self._send_json(
                    {
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                    status=500,
                )

        def do_POST(self) -> None:  # noqa: N802
            try:
                if self.path not in {
                    "/generate",
                    "/complete",
                    "/benchmark_complete",
                }:
                    self.send_error(404, "unknown path")
                    return
                payload = _load_json(self.rfile.read(_content_length(self)))
                if self.path == "/generate":
                    request_kwargs = _edge_generate_request_from_payload(payload)
                    with execution_lock:
                        output_ids = edge_service.generate(**request_kwargs)
                    self._send_json(
                        _edge_generate_response_to_payload(
                            req_id=request_kwargs["req_id"],
                            output_ids=list(output_ids),
                        ))
                    return

                request_kwargs = _edge_complete_request_from_payload(payload)
                if self.path == "/benchmark_complete":
                    with execution_lock:
                        benchmark_response = _run_benchmark_complete(
                            edge_service=edge_service,
                            request_kwargs=request_kwargs,
                        )
                    self._send_json(benchmark_response)
                    return

                with execution_lock:
                    text = edge_service.complete(**request_kwargs)
                self._send_json(
                    _edge_complete_response_to_payload(
                        req_id=request_kwargs["req_id"],
                        text=text,
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
            self._send_raw(body, status=status, content_type="application/json")

        def _send_raw(
            self,
            body: bytes,
            *,
            status: int = 200,
            content_type: str,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer((host, port), EdgeHandler)


def _content_length(handler: BaseHTTPRequestHandler) -> int:
    return int(handler.headers.get("Content-Length", "0"))


def _resolve_obj_by_qualname(qualname: str):
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


def _dump_json(payload: dict) -> bytes:
    from vllm.dssd.transport.http_utils import dump_json

    return dump_json(payload)


def _load_json(payload: bytes) -> dict:
    from vllm.dssd.transport.http_utils import load_json

    return load_json(payload)


def _edge_generate_request_from_payload(payload: dict) -> dict:
    if _allow_raw_sampling_params:
        sampling_params = SimpleNamespace(**dict(payload["sampling_params"]))
        return {
            "req_id": payload["req_id"],
            "prompt_token_ids": list(payload["prompt_token_ids"]),
            "sampling_params": sampling_params,
            "lora_request": payload.get("lora_request"),
        }

    from vllm.dssd.transport.http_utils import edge_generate_request_from_payload

    return edge_generate_request_from_payload(payload)


def _edge_generate_response_to_payload(*, req_id: str, output_ids: list[int]) -> dict:
    from vllm.dssd.transport.http_utils import edge_generate_response_to_payload

    return edge_generate_response_to_payload(req_id=req_id, output_ids=output_ids)


def _edge_complete_request_from_payload(payload: dict) -> dict:
    if _allow_raw_sampling_params:
        sampling_params = SimpleNamespace(**dict(payload["sampling_params"]))
        return {
            "req_id": payload["req_id"],
            "prompt": payload["prompt"],
            "sampling_params": sampling_params,
            "lora_request": payload.get("lora_request"),
        }

    from vllm.dssd.transport.http_utils import edge_complete_request_from_payload

    return edge_complete_request_from_payload(payload)


def _edge_complete_response_to_payload(*, req_id: str, text: str) -> dict:
    from vllm.dssd.transport.http_utils import edge_complete_response_to_payload

    return edge_complete_response_to_payload(req_id=req_id, text=text)


def _load_benchmark_page() -> bytes:
    page_path = Path(__file__).with_name("static") / "edge_benchmark.html"
    return page_path.read_bytes()


def _run_benchmark_complete(*, edge_service, request_kwargs: dict) -> dict:
    tokenizer = getattr(edge_service, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("benchmark route requires edge_service.tokenizer")

    prompt_token_ids = list(tokenizer(request_kwargs["prompt"]).input_ids)
    output_ids, stats = edge_service.generate_with_stats(
        req_id=request_kwargs["req_id"],
        prompt_token_ids=prompt_token_ids,
        sampling_params=request_kwargs["sampling_params"],
        lora_request=request_kwargs.get("lora_request"),
    )
    return {
        "req_id": request_kwargs["req_id"],
        "text": tokenizer.decode(list(output_ids), skip_special_tokens=True),
        "prompt_token_count": len(prompt_token_ids),
        "output_token_count": len(output_ids),
        "total_rounds": stats.total_rounds,
        "total_draft_tokens": stats.total_draft_tokens,
        "total_accepted_tokens": stats.total_accepted_tokens,
        "draft_acceptance_rate": stats.draft_acceptance_rate,
        "all_accept_round_rate": stats.all_accept_round_rate,
        "avg_accepted_len_per_round": stats.avg_accepted_len_per_round,
    }


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
