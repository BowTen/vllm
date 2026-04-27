# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_DEFAULT_SERVICE_FACTORY = (
    "vllm.dssd.entrypoints.native_spec_server.build_native_spec_service"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a native vLLM speculative decoding benchmark server."
    )
    parser.add_argument("--service-factory", default=_DEFAULT_SERVICE_FACTORY)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6007)
    parser.add_argument("--ready-file")
    parser.add_argument("--model", required=True, help="Target model path.")
    parser.add_argument(
        "--draft-model",
        required=True,
        help="Draft model path used in vLLM speculative_config.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer path. Defaults to --draft-model.",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=4,
        help="Default native speculative draft length.",
    )
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    service_factory = _resolve_obj_by_qualname(args.service_factory)
    service, cleanup = _normalize_service_factory_result(service_factory(args))
    server = _build_server(host=args.host, port=args.port, service=service)
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


class NativeSpecBenchmarkService:

    def __init__(
        self,
        *,
        llm,
        tokenizer,
        default_num_speculative_tokens: int,
        seed: int,
    ) -> None:
        self.llm = llm
        self.tokenizer = tokenizer
        self.default_num_speculative_tokens = int(default_num_speculative_tokens)
        self.seed = int(seed)

    def complete(
        self,
        *,
        req_id: str,
        prompt: str,
        sampling_params: dict,
    ) -> dict:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        prompt_token_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )
        max_tokens = int(sampling_params.get("max_tokens", 128))
        seed_value = sampling_params.get("seed", self.seed)
        seed = None if seed_value in (None, "") else int(seed_value)
        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=float(sampling_params.get("temperature", 0.0)),
            ignore_eos=bool(sampling_params.get("ignore_eos", True)),
            seed=seed,
        )
        before = self.llm.get_metrics()
        _synchronize_if_needed()
        start = time.perf_counter()
        outputs = self.llm.generate(
            [TokensPrompt(prompt_token_ids=prompt_token_ids)],
            sampling_params=params,
            use_tqdm=False,
        )
        _synchronize_if_needed()
        elapsed = time.perf_counter() - start
        after = self.llm.get_metrics()

        completion = outputs[0].outputs[0]
        token_ids = list(completion.token_ids)
        text = getattr(completion, "text", None)
        if not text:
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)

        num_drafts = _metric_delta(
            before,
            after,
            "vllm:spec_decode_num_drafts",
        )
        num_draft_tokens = _metric_delta(
            before,
            after,
            "vllm:spec_decode_num_draft_tokens",
        )
        num_accepted_tokens = _metric_delta(
            before,
            after,
            "vllm:spec_decode_num_accepted_tokens",
        )
        draft_acceptance_rate = (
            num_accepted_tokens / num_draft_tokens
            if num_draft_tokens > 0
            else math.nan
        )
        acceptance_length = 1.0 + (
            num_accepted_tokens / num_drafts if num_drafts > 0 else 0.0
        )
        return {
            "req_id": req_id,
            "text": text,
            "server_inference_seconds": elapsed,
            "prompt_token_count": len(prompt_token_ids),
            "output_token_count": len(token_ids),
            "tokens_per_second": (
                len(token_ids) / elapsed if elapsed > 0 else math.nan
            ),
            "num_drafts": num_drafts,
            "num_draft_tokens": num_draft_tokens,
            "num_accepted_tokens": num_accepted_tokens,
            "draft_acceptance_rate": draft_acceptance_rate,
            "acceptance_length": acceptance_length,
            "num_speculative_tokens": self.default_num_speculative_tokens,
        }

    def close(self) -> None:
        del self.llm
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            return


def build_native_spec_service(args):
    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer_name = args.tokenizer or args.draft_model
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        local_files_only=Path(tokenizer_name).exists(),
    )
    llm = LLM(
        model=args.model,
        tokenizer=tokenizer_name,
        speculative_config={
            "model": args.draft_model,
            "method": "draft_model",
            "num_speculative_tokens": args.num_speculative_tokens,
            "draft_tensor_parallel_size": 1,
        },
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        dtype=args.dtype,
        enforce_eager=args.enforce_eager,
        disable_log_stats=False,
        enable_prefix_caching=False,
    )
    service = NativeSpecBenchmarkService(
        llm=llm,
        tokenizer=tokenizer,
        default_num_speculative_tokens=args.num_speculative_tokens,
        seed=args.seed,
    )
    return service, service.close


def _build_server(
    *,
    host: str,
    port: int,
    service,
) -> ThreadingHTTPServer:
    execution_lock = threading.Lock()

    class NativeSpecHandler(BaseHTTPRequestHandler):

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
                if self.path != "/benchmark_complete":
                    self.send_error(404, "unknown path")
                    return
                payload = _load_json(self.rfile.read(_content_length(self)))
                request_kwargs = {
                    "req_id": payload["req_id"],
                    "prompt": payload["prompt"],
                    "sampling_params": dict(payload["sampling_params"]),
                }
                with execution_lock:
                    response = service.complete(**request_kwargs)
                self._send_json(response)
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
            self._send_raw(
                _dump_json(payload),
                status=status,
                content_type="application/json",
            )

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

    return ThreadingHTTPServer((host, port), NativeSpecHandler)


def _load_benchmark_page() -> bytes:
    page_path = Path(__file__).with_name("static") / "native_spec_benchmark.html"
    return page_path.read_bytes()


def _content_length(handler: BaseHTTPRequestHandler) -> int:
    return int(handler.headers.get("Content-Length", "0"))


def _resolve_obj_by_qualname(qualname: str):
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


def _normalize_service_factory_result(result):
    if isinstance(result, tuple):
        service, cleanup = result
        return service, cleanup
    return result, None


def _load_json(payload: bytes) -> dict:
    return json.loads(payload.decode("utf-8"))


def _dump_json(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _metric_delta(before, after, name: str) -> float:
    return _metric_value(after, name) - _metric_value(before, name)


def _metric_value(metrics, name: str) -> float:
    for metric in metrics:
        if metric.name == name:
            return float(metric.value)
    raise KeyError(f"metric {name!r} not found")


def _synchronize_if_needed() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _hard_exit(code: int) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    raise SystemExit(main())
