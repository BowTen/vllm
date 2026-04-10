import importlib
from dataclasses import is_dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest


def _load_protocol_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    repo_root = Path(__file__).resolve().parents[3]
    vllm_dir = repo_root / "vllm"

    torch_module = ModuleType("torch")
    torch_module.Tensor = object

    lora_request_module = ModuleType("vllm.lora.request")

    class LoRARequest:
        pass

    lora_request_module.LoRARequest = LoRARequest

    sampling_params_module = ModuleType("vllm.sampling_params")

    class SamplingParams:
        pass

    sampling_params_module.SamplingParams = SamplingParams

    vllm_package = ModuleType("vllm")
    vllm_package.__path__ = [str(vllm_dir)]

    monkeypatch.syspath_prepend(str(repo_root))
    with patch.dict(
            "sys.modules", {
                "torch": torch_module,
                "vllm": vllm_package,
                "vllm.lora.request": lora_request_module,
                "vllm.sampling_params": sampling_params_module,
            }):
        for module_name in (
                "vllm.dssd",
                "vllm.dssd.protocol",
                "vllm.dssd.protocol.types",
        ):
            importlib.sys.modules.pop(module_name, None)

        dssd_package = importlib.import_module("vllm.dssd")
        protocol_package = importlib.import_module("vllm.dssd.protocol")
        protocol_types = importlib.import_module("vllm.dssd.protocol.types")

    return SimpleNamespace(
        dssd_package=dssd_package,
        protocol_package=protocol_package,
        CloseSessionAck=protocol_types.CloseSessionAck,
        CloseSessionRequest=protocol_types.CloseSessionRequest,
        OpenSessionRequest=protocol_types.OpenSessionRequest,
        OpenSessionResponse=protocol_types.OpenSessionResponse,
        VerifyRoundRequest=protocol_types.VerifyRoundRequest,
        VerifyRoundResponse=protocol_types.VerifyRoundResponse,
    )


@pytest.fixture
def protocol_modules(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    return _load_protocol_modules(monkeypatch)


def test_protocol_messages_are_dataclasses(
        protocol_modules: SimpleNamespace) -> None:
    assert is_dataclass(protocol_modules.OpenSessionRequest)
    assert is_dataclass(protocol_modules.OpenSessionResponse)
    assert is_dataclass(protocol_modules.VerifyRoundRequest)
    assert is_dataclass(protocol_modules.VerifyRoundResponse)
    assert is_dataclass(protocol_modules.CloseSessionRequest)
    assert is_dataclass(protocol_modules.CloseSessionAck)


def test_verify_round_response_requires_exactly_one_bypass_payload(
        protocol_modules: SimpleNamespace) -> None:
    protocol_modules.VerifyRoundResponse(
        req_id="r",
        accepted_len=0,
        bonus_token_id=7,
    )

    with pytest.raises(ValueError, match="exactly one bypass payload"):
        protocol_modules.VerifyRoundResponse(req_id="r", accepted_len=0)

    with pytest.raises(ValueError, match="exactly one bypass payload"):
        protocol_modules.VerifyRoundResponse(
            req_id="r",
            accepted_len=0,
            bonus_token_id=7,
            rejected_target_logits=object(),
        )


def test_protocol_package_exports_are_stable(
        protocol_modules: SimpleNamespace) -> None:
    assert protocol_modules.protocol_package.OpenSessionRequest is (
        protocol_modules.OpenSessionRequest)
    assert protocol_modules.dssd_package.VerifyRoundResponse is (
        protocol_modules.VerifyRoundResponse)
