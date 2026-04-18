from __future__ import annotations

from .state_bridge import validate_edge_sampling_params


class EdgeStateBridgeV1:
    def bootstrap_first_token(
        self,
        session,
        bootstrap_token_id: int,
        model_runner,
    ) -> int:
        self.inject_external_token(session, bootstrap_token_id, model_runner)
        self.clear_round_state(session)
        return int(bootstrap_token_id)

    def prepare_next_decode(
        self,
        session,
        input_token_id: int,
        model_runner,
    ) -> None:
        self._ensure_supported_sampling_params(session)
        self._ensure_prompt_computed(session)
        session.round_state.committed_token_id = int(input_token_id)
        self._sync_request_state(session, model_runner)

    def commit_token(self, session, token_id: int, model_runner) -> None:
        self._ensure_supported_sampling_params(session)
        self._ensure_prompt_computed(session)
        session.append_token(int(token_id), computed_delta=1)
        self._sync_request_state(session, model_runner)

    def inject_external_token(
        self,
        session,
        token_id: int,
        model_runner,
    ) -> None:
        self._ensure_supported_sampling_params(session)
        self._ensure_prompt_computed(session)
        session.append_token(int(token_id), computed_delta=0)
        self._sync_request_state(session, model_runner)

    def rollback(self, session, rejected_count: int, model_runner) -> None:
        self._ensure_supported_sampling_params(session)
        self._ensure_prompt_computed(session)
        if rejected_count <= 0:
            return
        session.num_computed_tokens = max(
            session.prompt_len,
            session.num_computed_tokens - rejected_count,
        )
        session.rollback(rejected_count)
        self._sync_request_state(session, model_runner)

    def clear_round_state(self, session) -> None:
        session.round_state.reset()

    def _ensure_supported_sampling_params(self, session) -> None:
        validate_edge_sampling_params(session.sampling_params)

    def _ensure_prompt_computed(self, session) -> None:
        if session.num_computed_tokens < session.prompt_len:
            session.num_computed_tokens = session.prompt_len

    def _sync_request_state(self, session, model_runner) -> None:
        req_state = model_runner.requests[session.req_id]
        committed_output_ids = session.committed_output_ids()
        req_state.output_token_ids[:] = committed_output_ids
        req_state.num_computed_tokens = session.num_computed_tokens

        req_idx = model_runner.input_batch.req_id_to_index.get(session.req_id)
        if req_idx is None:
            return

        output_start = session.prompt_len
        output_len = len(committed_output_ids)
        if output_len:
            model_runner.input_batch.token_ids_cpu[
                req_idx, output_start:output_start + output_len
            ] = committed_output_ids
            model_runner.input_batch.is_token_ids[
                req_idx, output_start:output_start + output_len
            ] = True
        model_runner.input_batch.req_output_token_ids[req_idx] = (
            req_state.output_token_ids
        )
        model_runner.input_batch.num_tokens_no_spec[req_idx] = session.total_len
        model_runner.input_batch.num_computed_tokens_cpu[
            req_idx] = session.num_computed_tokens
        model_runner.input_batch.prev_sampled_token_ids = None
        model_runner.input_batch.prev_req_id_to_index = None
