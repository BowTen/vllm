from __future__ import annotations

from collections import defaultdict

import torch

from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from .types import (
    VerifierRoundRequest,
    VerifierRoundResult,
    VerifierRoundState,
    VerifierSession,
)


class VerifierStateBridge:
    """负责把 DSSD 的协议语义写进 GPUModelRunner.req_states。"""

    def __init__(self) -> None:
        self._round_states: dict[str, VerifierRoundState] = defaultdict(
            VerifierRoundState
        )

    def finish_prefill_without_commit(self, session: VerifierSession) -> None:
        # bootstrap 阶段只允许把 prompt 标记为“已计算”，
        # 不能把 bootstrap token 提前推进到协议层前缀。
        session.num_computed_tokens = session.prompt_len
        session.total_len = session.prompt_len
        self.clear_round_state(session)

    def prepare_round(
        self,
        session: VerifierSession,
        request: VerifierRoundRequest,
        model_runner: GPUModelRunner,
        gamma: int,
    ) -> None:
        request.validate(gamma=gamma)
        self.set_last_committed_token(
            session, request.committed_token_id, model_runner
        )
        self.set_draft_tokens(session, request.draft_token_ids, model_runner)
        self.set_round_q_values(session, request.draft_q_values)

    def set_last_committed_token(
        self,
        session: VerifierSession,
        committed_token_id: int,
        model_runner: GPUModelRunner,
    ) -> None:
        req_idx = self._req_idx(session, model_runner)
        # 这是 verifier 设计里最关键的一点：
        # 当前轮开始前，只把 committed_token 写到 last_sampled_tokens，
        # 不提前把它 append 到 prefix。
        model_runner.req_states.last_sampled_tokens[req_idx, 0] = committed_token_id
        self._round_state(session).committed_token_id = committed_token_id

    def set_draft_tokens(
        self,
        session: VerifierSession,
        draft_token_ids: list[int],
        model_runner: GPUModelRunner,
    ) -> None:
        req_idx = self._req_idx(session, model_runner)
        draft_tokens = model_runner.req_states.draft_tokens
        draft_tokens[req_idx].zero_()
        if draft_token_ids:
            draft_tokens[req_idx, : len(draft_token_ids)] = torch.tensor(
                draft_token_ids,
                dtype=torch.int64,
                device=draft_tokens.device,
            )
        self._round_state(session).draft_token_ids = list(draft_token_ids)

    def set_round_q_values(
        self,
        session: VerifierSession,
        q_values: list[float],
    ) -> None:
        self._round_state(session).draft_q_values = list(q_values)

    def commit_committed_token_before_postprocess(
        self,
        session: VerifierSession,
        committed_token_id: int,
        model_runner: GPUModelRunner,
    ) -> None:
        req_idx = self._req_idx(session, model_runner)
        total_len = session.total_len

        # 这里显式提交当前轮输入 token。它已经作为 last_sampled_tokens
        # 参与了本轮 forward，但不属于 postprocess() 视角下的“新输出 token”。
        model_runner.req_states.last_sampled_tokens[req_idx, 0] = committed_token_id
        model_runner.req_states.all_token_ids.gpu[req_idx, total_len] = (
            committed_token_id
        )
        model_runner.req_states.total_len.gpu[req_idx] = total_len + 1

        session.token_ids.extend([committed_token_id])
        session.total_len += 1

    def set_round_result(
        self,
        session: VerifierSession,
        result: VerifierRoundResult,
    ) -> None:
        round_state = self._round_state(session)
        round_state.last_result = result

        # postprocess() 现在只提交 accepted draft prefix。
        committed_token_ids: list[int] = []
        if result.accepted_len > 0:
            committed_token_ids.extend(
                round_state.draft_token_ids[: result.accepted_len]
            )
        session.token_ids.extend(committed_token_ids)
        session.num_computed_tokens += 1 + result.accepted_len
        session.total_len += result.accepted_len

    def take_round_result(self, session: VerifierSession) -> VerifierRoundResult:
        result = self._round_state(session).last_result
        if result is None:
            raise RuntimeError("当前会话还没有 round result")
        return result

    def clear_round_state(self, session: VerifierSession) -> None:
        self._round_state(session).reset()

    def remove_round_state(self, session: VerifierSession) -> None:
        self._round_states.pop(session.req_id, None)

    def _round_state(self, session: VerifierSession) -> VerifierRoundState:
        return self._round_states[session.req_id]

    def _req_idx(self, session: VerifierSession, model_runner: GPUModelRunner) -> int:
        try:
            return model_runner.req_states.req_id_to_index[session.req_id]
        except KeyError as exc:
            raise RuntimeError(f"找不到 verifier request state: {session.req_id}") from exc
