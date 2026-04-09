import importlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest


def _load_verifier_modules(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
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
                "vllm.dssd.verifier",
                "vllm.dssd.verifier.types",
        ):
            importlib.sys.modules.pop(module_name, None)

        verifier_package = importlib.import_module("vllm.dssd.verifier")
        verifier_types = importlib.import_module("vllm.dssd.verifier.types")

    return SimpleNamespace(
        SamplingParams=SamplingParams,
        verifier_package=verifier_package,
        VerifierOpenSessionResult=verifier_types.VerifierOpenSessionResult,
        VerifierRoundRequest=verifier_types.VerifierRoundRequest,
        VerifierRoundResult=verifier_types.VerifierRoundResult,
        VerifierRoundState=verifier_types.VerifierRoundState,
        VerifierSession=verifier_types.VerifierSession,
    )


@pytest.fixture
def verifier_modules(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    return _load_verifier_modules(monkeypatch)


def test_package_exports_verifier_types(
        verifier_modules: SimpleNamespace) -> None:
    verifier_package = verifier_modules.verifier_package

    assert verifier_package.VerifierRoundResult is (
        verifier_modules.VerifierRoundResult)
    assert verifier_package.VerifierRoundState is (
        verifier_modules.VerifierRoundState)
    assert verifier_package.__all__ == [
        "VerifierOpenSessionResult",
        "VerifierRoundRequest",
        "VerifierRoundResult",
        "VerifierRoundState",
        "VerifierSession",
    ]


def test_round_request_validate_checks_lengths_and_gamma(
        verifier_modules: SimpleNamespace) -> None:
    bad_lengths = verifier_modules.VerifierRoundRequest(
        req_id="req-1",
        committed_token_id=9,
        draft_token_ids=[11, 12],
        draft_q_values=[0.2],
    )
    with pytest.raises(ValueError, match="长度不一致"):
        bad_lengths.validate(gamma=2)

    bad_gamma = verifier_modules.VerifierRoundRequest(
        req_id="req-1",
        committed_token_id=9,
        draft_token_ids=[11, 12, 13],
        draft_q_values=[0.2, 0.3, 0.4],
    )
    with pytest.raises(ValueError, match="超过固定 gamma"):
        bad_gamma.validate(gamma=2)


def test_session_output_len_tracks_confirmed_suffix_only(
        verifier_modules: SimpleNamespace) -> None:
    session = verifier_modules.VerifierSession(
        req_id="req-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=verifier_modules.SamplingParams(),
        block_ids=([0],),
        prompt_len=3,
        token_ids=[1, 2, 3, 4, 5],
        total_len=5,
    )
    assert session.output_len == 2


def test_session_output_len_clamps_to_zero(
        verifier_modules: SimpleNamespace) -> None:
    session = verifier_modules.VerifierSession(
        req_id="req-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=verifier_modules.SamplingParams(),
        block_ids=([0],),
        prompt_len=3,
        token_ids=[1, 2],
        total_len=2,
    )
    assert session.output_len == 0


def test_round_result_helpers_match_acceptance_shape(
        verifier_modules: SimpleNamespace) -> None:
    result = verifier_modules.VerifierRoundResult(req_id="req-1",
                                                  accepted_len=2,
                                                  bonus_token_id=17)
    assert result.is_all_accepted()
    assert not result.is_rejected()

    rejected = verifier_modules.VerifierRoundResult(req_id="req-1",
                                                    accepted_len=1,
                                                    rejected_step=1)
    assert not rejected.is_all_accepted()
    assert rejected.is_rejected()

    bootstrap = verifier_modules.VerifierOpenSessionResult(
        req_id="req-1", bootstrap_token_id=7)
    assert bootstrap.bootstrap_token_id == 7


def test_round_state_reset_clears_round_tracking(
        verifier_modules: SimpleNamespace) -> None:
    last_result = verifier_modules.VerifierRoundResult(req_id="req-1",
                                                       accepted_len=1,
                                                       rejected_step=0)
    state = verifier_modules.VerifierRoundState(
        committed_token_id=9,
        draft_token_ids=[11, 12],
        draft_q_values=[0.2, 0.3],
        last_result=last_result,
    )

    state.reset()

    assert state.committed_token_id is None
    assert state.draft_token_ids == []
    assert state.draft_q_values == []
    assert state.last_result is None
