import importlib
from dataclasses import is_dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest


def _load_protocol_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_runtime_type_stubs: bool = True,
) -> SimpleNamespace:
    repo_root = Path(__file__).resolve().parents[3]
    vllm_dir = repo_root / "vllm"

    vllm_package = ModuleType("vllm")
    vllm_package.__path__ = [str(vllm_dir)]
    patched_modules: dict[str, ModuleType] = {
        "vllm": vllm_package,
    }

    if include_runtime_type_stubs:
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
        patched_modules.update({
            "torch": torch_module,
            "vllm.lora.request": lora_request_module,
            "vllm.sampling_params": sampling_params_module,
        })
    else:
        SamplingParams = None

    monkeypatch.syspath_prepend(str(repo_root))
    with patch.dict(
            "sys.modules", patched_modules):
        module_names = [
            "vllm.dssd",
            "vllm.dssd.protocol",
            "vllm.dssd.protocol.types",
        ]
        if not include_runtime_type_stubs:
            module_names.extend([
                "torch",
                "vllm.lora.request",
                "vllm.sampling_params",
            ])

        for module_name in module_names:
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
        SamplingParams=SamplingParams,
        VerifyRoundRequest=protocol_types.VerifyRoundRequest,
        VerifyRoundResponse=protocol_types.VerifyRoundResponse,
        protocol_types=protocol_types,
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
    protocol_modules.VerifyRoundResponse(
        req_id="r",
        accepted_len=0,
        rejected_target_logits=object(),
    )
    protocol_modules.VerifyRoundResponse(
        req_id="r",
        accepted_len=0,
        rejected_token_id=8,
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

    with pytest.raises(ValueError, match="exactly one bypass payload"):
        protocol_modules.VerifyRoundResponse(
            req_id="r",
            accepted_len=0,
            rejected_target_logits=object(),
            rejected_token_id=8,
        )


def test_verify_round_request_validate_matches_verifier_contract(
        protocol_modules: SimpleNamespace) -> None:
    request = protocol_modules.VerifyRoundRequest(
        req_id="r",
        committed_token_id=9,
        draft_token_ids=[10, 11],
        draft_q_values=[0.2, 0.3],
    )

    request.validate(gamma=2)

    with pytest.raises(ValueError, match="长度不一致"):
        protocol_modules.VerifyRoundRequest(
            req_id="r",
            committed_token_id=9,
            draft_token_ids=[10, 11],
            draft_q_values=[0.2],
        ).validate(gamma=2)

    with pytest.raises(ValueError, match="超过固定 gamma"):
        protocol_modules.VerifyRoundRequest(
            req_id="r",
            committed_token_id=9,
            draft_token_ids=[10, 11, 12],
            draft_q_values=[0.2, 0.3, 0.4],
        ).validate(gamma=2)


def test_protocol_types_do_not_require_runtime_type_imports(
        monkeypatch: pytest.MonkeyPatch) -> None:
    protocol_modules = _load_protocol_modules(
        monkeypatch,
        include_runtime_type_stubs=False,
    )

    assert "torch" not in protocol_modules.protocol_types.__dict__
    assert "SamplingParams" not in protocol_modules.protocol_types.__dict__
    assert "LoRARequest" not in protocol_modules.protocol_types.__dict__


def test_protocol_package_exports_are_stable(
        protocol_modules: SimpleNamespace) -> None:
    assert protocol_modules.protocol_package.OpenSessionRequest is (
        protocol_modules.OpenSessionRequest)
    assert protocol_modules.dssd_package.VerifyRoundResponse is (
        protocol_modules.VerifyRoundResponse)
