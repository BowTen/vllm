# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse

from vllm.entrypoints.cli.types import CLISubcommand
from vllm.entrypoints.spec_decode.verifier_server import (
    VerifierServerArgs,
    run_verifier_server,
)
from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG
from vllm.utils.argparse_utils import FlexibleArgumentParser

DESCRIPTION = """Launch the internal verifier service used by the experimental
distributed draft-model speculative decoding flow.
"""


class SpecVerifierSubcommand(CLISubcommand):
    name = "spec-verifier"

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        run_verifier_server(
            VerifierServerArgs(
                model=args.model,
                host=args.host,
                port=args.port,
                device=args.device,
                dtype=args.dtype,
                trust_remote_code=args.trust_remote_code,
                log_level=args.log_level,
                scheduler_max_batch_size=args.scheduler_max_batch_size,
                scheduler_batch_wait_ms=args.scheduler_batch_wait_ms,
                scheduler_max_batch_tokens=args.scheduler_max_batch_tokens,
                scheduler_queue_timeout_ms=args.scheduler_queue_timeout_ms,
                session_idle_timeout_s=args.session_idle_timeout_s,
            )
        )

    def subparser_init(
        self, subparsers: argparse._SubParsersAction
    ) -> FlexibleArgumentParser:
        parser = subparsers.add_parser(
            self.name,
            help="Launch the internal verifier service for distributed draft "
            "speculative decoding.",
            description=DESCRIPTION,
            usage="vllm spec-verifier --model MODEL [options]",
        )
        parser.add_argument("--model", required=True, help="Target model path or ID.")
        parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
        parser.add_argument("--port", type=int, default=9000, help="Bind port.")
        parser.add_argument(
            "--device",
            default=None,
            help="Torch device for the verifier runtime, e.g. cuda:1.",
        )
        parser.add_argument(
            "--dtype",
            default="auto",
            help="Torch dtype passed to transformers, e.g. auto, float16, bfloat16.",
        )
        parser.add_argument(
            "--trust-remote-code",
            action="store_true",
            help="Forward trust_remote_code=True to transformers.",
        )
        parser.add_argument(
            "--log-level",
            default="info",
            choices=["critical", "error", "warning", "info", "debug", "trace"],
            help="uvicorn log level.",
        )
        parser.add_argument(
            "--scheduler-max-batch-size",
            type=int,
            default=8,
            help="Maximum number of proposals to aggregate into one verifier micro-batch.",
        )
        parser.add_argument(
            "--scheduler-batch-wait-ms",
            type=float,
            default=1.0,
            help="Maximum wait time, in milliseconds, to collect additional proposals.",
        )
        parser.add_argument(
            "--scheduler-max-batch-tokens",
            type=int,
            default=None,
            help="Optional total draft-token budget per verifier micro-batch.",
        )
        parser.add_argument(
            "--scheduler-queue-timeout-ms",
            type=float,
            default=None,
            help="Optional timeout, in milliseconds, for proposals waiting in the verifier queue.",
        )
        parser.add_argument(
            "--session-idle-timeout-s",
            type=float,
            default=None,
            help="Optional idle timeout in seconds after which verifier sessions are closed.",
        )
        parser.epilog = VLLM_SUBCMD_PARSER_EPILOG.format(subcmd=self.name)
        return parser


def cmd_init() -> list[CLISubcommand]:
    return [SpecVerifierSubcommand()]
