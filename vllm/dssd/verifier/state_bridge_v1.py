from __future__ import annotations

from collections import defaultdict

from .types import (
    VerifierRoundRequest,
    VerifierRoundResult,
    VerifierRoundState,
    VerifierSession,
)


class VerifierStateBridgeV1:
    def __init__(self) -> None:
        self._round_states: dict[str, VerifierRoundState] = defaultdict(
            VerifierRoundState
        )

    def finish_prefill_without_commit(
        self,
        session: VerifierSession,
        model_runner,
    ) -> None:
        req_state = model_runner.requests[session.req_id]
        req_idx = model_runner.input_batch.req_id_to_index[session.req_id]
        session.token_ids[:] = session.prompt_token_ids
        session.num_computed_tokens = session.prompt_len
        session.total_len = session.prompt_len
        req_state.output_token_ids[:] = []
        req_state.num_computed_tokens = session.prompt_len
        if hasattr(model_runner.input_batch, "req_output_token_ids"):
            model_runner.input_batch.req_output_token_ids[req_idx] = (
                req_state.output_token_ids
            )
        model_runner.input_batch.num_tokens_no_spec[req_idx] = session.prompt_len
        model_runner.input_batch.num_computed_tokens_cpu[req_idx] = (
            session.prompt_len
        )
        self.remove_round_state(session)

    def inject_local_token(
        self,
        session: VerifierSession,
        token_id: int,
        model_runner,
        *,
        computed_delta: int,
    ) -> None:
        session.token_ids.append(int(token_id))
        session.total_len += 1
        session.num_computed_tokens += int(computed_delta)
        self._sync_request_state(session, model_runner)
        self.remove_round_state(session)

    def prepare_local_decode(
        self,
        session: VerifierSession,
        input_token_id: int,
        model_runner,
    ) -> None:
        del input_token_id
        self._sync_request_state(session, model_runner)

    def commit_local_token(
        self,
        session: VerifierSession,
        token_id: int,
        model_runner,
    ) -> None:
        session.token_ids.append(int(token_id))
        session.total_len += 1
        session.num_computed_tokens += 1
        self._sync_request_state(session, model_runner)

    def begin_round(
        self,
        session: VerifierSession,
        request: VerifierRoundRequest,
        model_runner,
        *,
        gamma: int,
    ) -> None:
        self._validate_req_id(session.req_id, request.req_id)
        request.validate(gamma=gamma)

        round_state = self._round_states[session.req_id]
        if round_state.committed_token_id is not None:
            raise ValueError("round is already in progress")

        req_state = model_runner.requests[session.req_id]
        req_idx = model_runner.input_batch.req_id_to_index[session.req_id]
        write_pos = int(model_runner.input_batch.num_tokens_no_spec[req_idx])

        session.token_ids.append(request.committed_token_id)
        session.total_len += 1
        req_state.output_token_ids.append(request.committed_token_id)
        model_runner.input_batch.token_ids_cpu[req_idx, write_pos] = (
            request.committed_token_id
        )
        if hasattr(model_runner.input_batch, "is_token_ids"):
            model_runner.input_batch.is_token_ids[req_idx, write_pos] = True
        model_runner.input_batch.num_tokens_no_spec[req_idx] = write_pos + 1
        if hasattr(model_runner.input_batch, "prev_sampled_token_ids"):
            model_runner.input_batch.prev_sampled_token_ids = None
        if hasattr(model_runner.input_batch, "prev_req_id_to_index"):
            model_runner.input_batch.prev_req_id_to_index = None

        round_state.committed_token_id = request.committed_token_id
        round_state.committed_token_committed = True
        round_state.draft_token_ids = list(request.draft_token_ids)
        round_state.draft_q_values = list(request.draft_q_values)

    def finish_round(
        self,
        session: VerifierSession,
        request: VerifierRoundRequest,
        result: VerifierRoundResult,
        model_runner,
    ) -> None:
        self._validate_req_id(session.req_id, request.req_id)
        self._validate_req_id(session.req_id, result.req_id)

        round_state = self._round_states[session.req_id]
        if not round_state.committed_token_committed:
            raise ValueError("committed token must be written before finish_round")

        draft_len = len(round_state.draft_token_ids)
        if not 0 <= result.accepted_len <= draft_len:
            raise ValueError("accepted_len must be within the current draft range")
        if result.accepted_len == draft_len and not result.is_all_accepted():
            raise ValueError(
                "accepted_len equal to draft_len requires the bonus payload"
            )
        if result.accepted_len < draft_len and not result.is_rejected():
            raise ValueError(
                "accepted_len below draft_len requires the rejected payload"
            )

        accepted_prefix = round_state.draft_token_ids[: result.accepted_len]
        req_state = model_runner.requests[session.req_id]
        req_idx = model_runner.input_batch.req_id_to_index[session.req_id]
        start = int(model_runner.input_batch.num_tokens_no_spec[req_idx])
        end = start + len(accepted_prefix)

        if accepted_prefix:
            model_runner.input_batch.token_ids_cpu[req_idx, start:end] = (
                accepted_prefix
            )
            if hasattr(model_runner.input_batch, "is_token_ids"):
                model_runner.input_batch.is_token_ids[req_idx, start:end] = True
            req_state.output_token_ids.extend(accepted_prefix)
            session.token_ids.extend(accepted_prefix)
            model_runner.input_batch.num_tokens_no_spec[req_idx] = end
            session.total_len += len(accepted_prefix)

        session.num_computed_tokens += 1 + result.accepted_len
        req_state.num_computed_tokens = session.num_computed_tokens
        model_runner.input_batch.num_computed_tokens_cpu[req_idx] = (
            session.num_computed_tokens
        )
        round_state.reset()

    def remove_round_state(self, session: VerifierSession) -> None:
        self._round_states.pop(session.req_id, None)

    @staticmethod
    def _validate_req_id(session_req_id: str, round_req_id: str) -> None:
        if session_req_id != round_req_id:
            raise ValueError("session and round req_id must match")

    def _sync_request_state(
        self,
        session: VerifierSession,
        model_runner,
    ) -> None:
        req_state = model_runner.requests[session.req_id]
        output_token_ids = session.token_ids[session.prompt_len:]
        req_state.output_token_ids[:] = output_token_ids
        req_state.num_computed_tokens = session.num_computed_tokens

        input_batch = model_runner.input_batch
        if hasattr(input_batch, "prev_sampled_token_ids"):
            input_batch.prev_sampled_token_ids = None
        if hasattr(input_batch, "prev_req_id_to_index"):
            input_batch.prev_req_id_to_index = None

        req_idx = input_batch.req_id_to_index.get(session.req_id)
        if req_idx is None:
            return

        output_start = session.prompt_len
        output_len = len(output_token_ids)
        if output_len:
            output_end = output_start + output_len
            input_batch.token_ids_cpu[
                req_idx,
                output_start:output_end,
            ] = output_token_ids
            if hasattr(input_batch, "is_token_ids"):
                input_batch.is_token_ids[
                    req_idx,
                    output_start:output_end,
                ] = True
        if hasattr(input_batch, "req_output_token_ids"):
            input_batch.req_output_token_ids[req_idx] = req_state.output_token_ids
        input_batch.num_tokens_no_spec[req_idx] = session.total_len
        input_batch.num_computed_tokens_cpu[req_idx] = session.num_computed_tokens
