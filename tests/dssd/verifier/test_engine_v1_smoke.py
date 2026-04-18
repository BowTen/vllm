# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.dssd.verifier.engine_v1 import VerifierDecodeEngineV1
from vllm.dssd.verifier.sampler_v1 import DSSDVerifierSamplerV1
from vllm.dssd.verifier.scheduler import VerifierSchedulerAdapter
from vllm.dssd.verifier.state_bridge_v1 import VerifierStateBridgeV1
from vllm.dssd.verifier.types import VerifierRoundRequest
from vllm.sampling_params import SamplingParams


@pytest.mark.skipif(not pytest.importorskip("torch").cuda.is_available(),
                    reason="requires cuda")
def test_open_verify_close_smoke_mrv1(real_worker_v1) -> None:
    worker, vllm_config, kv_cache_manager = real_worker_v1
    engine = VerifierDecodeEngineV1(
        vllm_config=vllm_config,
        worker=worker,
        scheduler=VerifierSchedulerAdapter(kv_cache_manager=kv_cache_manager),
        state_bridge=VerifierStateBridgeV1(),
        verifier_sampler=DSSDVerifierSamplerV1(worker.model_runner.sampler),
    )

    opened = engine.open_session(
        req_id="req-1",
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(temperature=0.0),
    )
    session = engine.sessions["req-1"]

    assert opened.bootstrap_token_id >= 0
    assert session.total_len == session.prompt_len
    assert session.token_ids == [1, 2, 3]

    result = engine.verify_round(
        session,
        VerifierRoundRequest(
            req_id="req-1",
            committed_token_id=opened.bootstrap_token_id,
            draft_token_ids=[],
            draft_q_values=[],
        ),
    )

    assert result.accepted_len == 0
    assert session.token_ids == [1, 2, 3, opened.bootstrap_token_id]

    engine.close_session(session)

    assert "req-1" not in engine.sessions
