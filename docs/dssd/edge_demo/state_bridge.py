from __future__ import annotations

from typing import cast

from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from .types import EdgeSession


class EdgeStateBridge:
    """负责把 DSSD edge 的协议语义写进 GPUModelRunner.req_states。"""

    def bootstrap_first_token(
        self,
        session: EdgeSession,
        bootstrap_token_id: int,
        model_runner: GPUModelRunner,
    ) -> int:
        # 这里的 bootstrap token 来自 verifier，不是 edge 本地 sample。
        self.inject_external_token(session, bootstrap_token_id, model_runner)
        self.clear_round_state(session)
        return int(bootstrap_token_id)

    def prepare_next_decode(
        self,
        session: EdgeSession,
        input_token_id: int,
        model_runner: GPUModelRunner,
    ) -> None:
        req_idx = self._req_idx(session, model_runner)
        req_states = model_runner.req_states
        req_states.last_sampled_tokens[req_idx, 0] = int(input_token_id)
        session.round_state.committed_token_id = int(input_token_id)

    def commit_token(
        self,
        session: EdgeSession,
        token_id: int,
    ) -> None:
        # 本地 decode 后，真正的 req_states 更新已经由 vLLM postprocess 完成；
        # 这里仅同步 edge 协议层镜像。
        session.append_token(int(token_id), computed_delta=1)

    def inject_external_token(
        self,
        session: EdgeSession,
        token_id: int,
        model_runner: GPUModelRunner,
    ) -> None:
        req_idx = self._req_idx(session, model_runner)
        old_total_len = session.total_len
        session.append_token(int(token_id), computed_delta=0)

        req_states = model_runner.req_states
        req_states.last_sampled_tokens[req_idx, 0] = int(token_id)
        req_states.all_token_ids.stage_write(req_idx, old_total_len, [int(token_id)])
        req_states.total_len.stage_write_elem(req_idx, session.total_len)
        req_states.num_computed_tokens.stage_write_elem(
            req_idx,
            session.num_computed_tokens,
        )
        req_states.apply_staged_writes()

    def rollback(
        self,
        session: EdgeSession,
        rejected_count: int,
        model_runner: GPUModelRunner,
    ) -> None:
        if rejected_count <= 0:
            return

        session.rollback(rejected_count)

        req_idx = self._req_idx(session, model_runner)
        req_states = model_runner.req_states
        req_states.total_len.stage_write_elem(req_idx, session.total_len)
        req_states.num_computed_tokens.stage_write_elem(
            req_idx,
            session.num_computed_tokens,
        )
        if session.token_ids:
            req_states.last_sampled_tokens[req_idx, 0] = int(session.token_ids[-1])
        req_states.apply_staged_writes()

    def clear_round_state(self, session: EdgeSession) -> None:
        session.round_state.reset()

    def _req_idx(self, session: EdgeSession, model_runner: GPUModelRunner) -> int:
        req_idx = model_runner.req_states.req_id_to_index.get(session.req_id)
        if req_idx is None:
            raise RuntimeError(f"找不到 edge request state: {session.req_id}")
        return cast(int, req_idx)
