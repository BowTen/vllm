from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types

import pytest


def _load_types_module():
    torch_module = types.ModuleType("torch")
    torch_module.Tensor = object
    sys.modules["torch"] = torch_module

    lora_request_module = types.ModuleType("vllm.lora.request")

    class LoRARequest:
        pass

    lora_request_module.LoRARequest = LoRARequest
    sys.modules["vllm.lora.request"] = lora_request_module

    sampling_params_module = types.ModuleType("vllm.sampling_params")

    class SamplingParams:
        pass

    sampling_params_module.SamplingParams = SamplingParams
    sys.modules["vllm.sampling_params"] = sampling_params_module

    module_name = "test_verifier_types_module"
    module_path = (
        Path(__file__).resolve().parents[3] / "vllm" / "dssd" / "verifier"
        / "types.py")
    spec = spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, SamplingParams


types_module, SamplingParams = _load_types_module()
VerifierOpenSessionResult = types_module.VerifierOpenSessionResult
VerifierRoundRequest = types_module.VerifierRoundRequest
VerifierRoundResult = types_module.VerifierRoundResult
VerifierSession = types_module.VerifierSession


def test_round_request_validate_checks_lengths_and_gamma() -> None:
    bad_lengths = VerifierRoundRequest(
        req_id="req-1",
        committed_token_id=9,
        draft_token_ids=[11, 12],
        draft_q_values=[0.2],
    )
    with pytest.raises(ValueError, match="长度不一致"):
        bad_lengths.validate(gamma=2)

    bad_gamma = VerifierRoundRequest(
        req_id="req-1",
        committed_token_id=9,
        draft_token_ids=[11, 12, 13],
        draft_q_values=[0.2, 0.3, 0.4],
    )
    with pytest.raises(ValueError, match="超过固定 gamma"):
        bad_gamma.validate(gamma=2)


def test_session_output_len_tracks_confirmed_suffix_only() -> None:
    session = VerifierSession(
        req_id="req-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(),
        block_ids=([0],),
        prompt_len=3,
        token_ids=[1, 2, 3, 4, 5],
        total_len=5,
    )
    assert session.output_len == 2


def test_round_result_helpers_match_acceptance_shape() -> None:
    result = VerifierRoundResult(req_id="req-1", accepted_len=2,
                                 bonus_token_id=17)
    assert result.is_all_accepted()
    assert not result.is_rejected()

    rejected = VerifierRoundResult(req_id="req-1", accepted_len=1,
                                   rejected_step=1)
    assert not rejected.is_all_accepted()
    assert rejected.is_rejected()

    bootstrap = VerifierOpenSessionResult(req_id="req-1",
                                          bootstrap_token_id=7)
    assert bootstrap.bootstrap_token_id == 7
