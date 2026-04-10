# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.metadata as metadata

import vllm
import vllm.version
from vllm import platforms


def test_cuda_platform_plugin_handles_source_checkout_without_dist_metadata(
    monkeypatch,
) -> None:
    class FakePynvml:
        @staticmethod
        def nvmlInit() -> None:
            return None

        @staticmethod
        def nvmlShutdown() -> None:
            return None

        @staticmethod
        def nvmlDeviceGetCount() -> int:
            return 1

    def raise_package_not_found(_dist_name: str) -> str:
        raise metadata.PackageNotFoundError("vllm")

    monkeypatch.setattr(metadata, "version", raise_package_not_found)
    monkeypatch.setattr(vllm, "__version__", "0.1.dev", raising=False)
    monkeypatch.setattr(vllm.version, "__version__", "0.1.dev", raising=False)
    monkeypatch.setattr(
        "vllm.utils.import_utils.import_pynvml",
        lambda: FakePynvml,
    )

    assert platforms.cuda_platform_plugin() == "vllm.platforms.cuda.CudaPlatform"
