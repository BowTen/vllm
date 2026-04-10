import importlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest


def _make_stub_module(module_name: str, **attrs: object) -> ModuleType:
    module = ModuleType(module_name)
    for attr_name, value in attrs.items():
        setattr(module, attr_name, value)
    return module


def _load_verifier_modules(
    monkeypatch: pytest.MonkeyPatch,
    extra_modules: dict[str, ModuleType] | None = None,
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
    extra_modules = extra_modules or {}

    monkeypatch.syspath_prepend(str(repo_root))
    with patch.dict(
            "sys.modules", {
                "torch": torch_module,
                "vllm": vllm_package,
                "vllm.lora.request": lora_request_module,
                "vllm.sampling_params": sampling_params_module,
                **extra_modules,
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
        "DSSDVerifierSampler",
        "VerifierDecodeEngine",
        "VerifierOpenSessionResult",
        "VerifierRoundRequest",
        "VerifierRoundResult",
        "VerifierRoundState",
        "VerifierSchedulerAdapter",
        "VerifierSession",
        "VerifierStateBridge",
    ]


def test_package_exports_are_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    sampler_class = type("DSSDVerifierSampler", (), {})
    engine_class = type("VerifierDecodeEngine", (), {})
    scheduler_class = type("VerifierSchedulerAdapter", (), {})
    bridge_class = type("VerifierStateBridge", (), {})
    extra_modules = {
        "vllm.dssd.verifier.engine": _make_stub_module(
            "vllm.dssd.verifier.engine",
            VerifierDecodeEngine=engine_class,
        ),
        "vllm.dssd.verifier.sampler": _make_stub_module(
            "vllm.dssd.verifier.sampler",
            DSSDVerifierSampler=sampler_class,
        ),
        "vllm.dssd.verifier.scheduler": _make_stub_module(
            "vllm.dssd.verifier.scheduler",
            VerifierSchedulerAdapter=scheduler_class,
        ),
        "vllm.dssd.verifier.state_bridge": _make_stub_module(
            "vllm.dssd.verifier.state_bridge",
            VerifierStateBridge=bridge_class,
        ),
    }
    verifier_modules = _load_verifier_modules(
        monkeypatch,
        extra_modules=extra_modules,
    )

    with patch.dict("sys.modules", extra_modules):
        (
            DSSDVerifierSampler,
            VerifierDecodeEngine,
            VerifierOpenSessionResult,
            VerifierRoundRequest,
            VerifierRoundResult,
            VerifierRoundState,
            VerifierSchedulerAdapter,
            VerifierSession,
            VerifierStateBridge,
        ) = (
            verifier_modules.verifier_package.DSSDVerifierSampler,
            verifier_modules.verifier_package.VerifierDecodeEngine,
            verifier_modules.verifier_package.VerifierOpenSessionResult,
            verifier_modules.verifier_package.VerifierRoundRequest,
            verifier_modules.verifier_package.VerifierRoundResult,
            verifier_modules.verifier_package.VerifierRoundState,
            verifier_modules.verifier_package.VerifierSchedulerAdapter,
            verifier_modules.verifier_package.VerifierSession,
            verifier_modules.verifier_package.VerifierStateBridge,
        )

    assert DSSDVerifierSampler is sampler_class
    assert VerifierDecodeEngine is engine_class
    assert VerifierOpenSessionResult is verifier_modules.VerifierOpenSessionResult
    assert VerifierRoundRequest is verifier_modules.VerifierRoundRequest
    assert VerifierRoundResult is verifier_modules.VerifierRoundResult
    assert VerifierRoundState is verifier_modules.VerifierRoundState
    assert VerifierSchedulerAdapter is scheduler_class
    assert VerifierSession is verifier_modules.VerifierSession
    assert VerifierStateBridge is bridge_class


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


def test_round_result_helpers_match_bypass_payload_shape(
        verifier_modules: SimpleNamespace) -> None:
    result = verifier_modules.VerifierRoundResult(req_id="req-1",
                                                  accepted_len=2,
                                                  bonus_token_id=17)
    assert result.is_all_accepted()
    assert not result.is_rejected()

    rejected = verifier_modules.VerifierRoundResult(req_id="req-1",
                                                    accepted_len=1,
                                                    rejected_target_logits=object())
    assert not rejected.is_all_accepted()
    assert rejected.is_rejected()

    bootstrap = verifier_modules.VerifierOpenSessionResult(
        req_id="req-1", bootstrap_token_id=7)
    assert bootstrap.bootstrap_token_id == 7


def test_round_result_requires_exactly_one_bypass_payload(
        verifier_modules: SimpleNamespace) -> None:
    with pytest.raises(ValueError, match="exactly one bypass payload"):
        verifier_modules.VerifierRoundResult(req_id="req-1", accepted_len=0)

    with pytest.raises(ValueError, match="exactly one bypass payload"):
        verifier_modules.VerifierRoundResult(
            req_id="req-1",
            accepted_len=1,
            bonus_token_id=17,
            rejected_target_logits=object(),
        )


def test_round_state_reset_clears_round_tracking(
        verifier_modules: SimpleNamespace) -> None:
    state = verifier_modules.VerifierRoundState(
        committed_token_id=9,
        committed_token_committed=True,
        draft_token_ids=[11, 12],
        draft_q_values=[0.2, 0.3],
    )

    state.reset()

    assert state.committed_token_id is None
    assert state.committed_token_committed is False
    assert state.draft_token_ids == []
    assert state.draft_q_values == []
    assert not hasattr(state, "last_result")
