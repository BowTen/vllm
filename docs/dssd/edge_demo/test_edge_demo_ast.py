from __future__ import annotations

import ast
import unittest
from pathlib import Path


EDGE_DEMO_DIR = Path(__file__).resolve().parent


def _parse_module(filename: str) -> ast.Module:
    path = EDGE_DEMO_DIR / filename
    if not path.exists():
        raise AssertionError(f"缺少文件: {path}")
    return ast.parse(path.read_text(encoding="utf-8"))


def _find_class(module: ast.Module, name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"缺少类: {name}")


def _method_names(class_node: ast.ClassDef) -> set[str]:
    return {
        node.name
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _contains_call(
    method_node: ast.FunctionDef | ast.AsyncFunctionDef,
    dotted_name: str,
) -> bool:
    parts = dotted_name.split(".")
    for node in ast.walk(method_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        seen: list[str] = []
        while isinstance(func, ast.Attribute):
            seen.append(func.attr)
            func = func.value
        if isinstance(func, ast.Name):
            seen.append(func.id)
            if list(reversed(seen)) == parts:
                return True
    return False


def _contains_attribute_chain(
    node: ast.AST,
    dotted_name: str,
) -> bool:
    parts = dotted_name.split(".")
    for child in ast.walk(node):
        if not isinstance(child, ast.Attribute):
            continue
        seen: list[str] = [child.attr]
        value = child.value
        while isinstance(value, ast.Attribute):
            seen.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            seen.append(value.id)
            if list(reversed(seen)) == parts:
                return True
    return False


def _contains_binop(
    method_node: ast.FunctionDef | ast.AsyncFunctionDef,
    left_name: str,
    op_type: type[ast.operator],
    right_name: str,
) -> bool:
    for node in ast.walk(method_node):
        if not isinstance(node, ast.BinOp):
            continue
        if not isinstance(node.op, op_type):
            continue
        if not isinstance(node.left, ast.Attribute):
            continue
        if not isinstance(node.right, ast.Name):
            continue
        if (
            isinstance(node.left.value, ast.Name)
            and node.left.value.id == "self"
            and node.left.attr == left_name
            and node.right.id == right_name
        ):
            return True
    return False


def _contains_min_with_kept_len(
    method_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    for node in ast.walk(method_node):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "min":
            continue
        if len(node.args) != 2:
            continue
        left, right = node.args
        if not isinstance(left, ast.Attribute):
            continue
        if not (
            isinstance(left.value, ast.Name)
            and left.value.id == "self"
            and left.attr == "num_computed_tokens"
        ):
            continue
        if not isinstance(right, ast.Name) or right.id != "kept_len":
            continue
        return True
    return False


class EdgeDemoAstTest(unittest.TestCase):
    def test_required_files_and_classes_exist(self) -> None:
        expected = {
            "types.py": [
                "EdgeVerifyRequest",
                "EdgeVerifyResponse",
                "EdgeOpenSessionResult",
                "EdgeRoundState",
                "EdgeSession",
            ],
            "scheduler.py": ["EdgeSchedulerAdapter"],
            "state_bridge.py": ["EdgeStateBridge"],
            "sampler.py": ["DSSDEdgeDraftSampler"],
            "engine.py": ["EdgeDecodeEngine"],
            "service.py": ["DSSDEdgeService"],
        }

        for filename, classes in expected.items():
            module = _parse_module(filename)
            for class_name in classes:
                _find_class(module, class_name)

    def test_core_methods_exist(self) -> None:
        scheduler_cls = _find_class(
            _parse_module("scheduler.py"),
            "EdgeSchedulerAdapter",
        )
        self.assertTrue(
            {
                "allocate_blocks",
                "build_prefill_step",
                "build_decode_step",
                "build_close_step",
                "free_blocks",
            }.issubset(_method_names(scheduler_cls))
        )

        bridge_cls = _find_class(
            _parse_module("state_bridge.py"),
            "EdgeStateBridge",
        )
        self.assertTrue(
            {
                "bootstrap_first_token",
                "prepare_next_decode",
                "commit_token",
                "rollback",
                "clear_round_state",
            }.issubset(_method_names(bridge_cls))
        )

        engine_cls = _find_class(_parse_module("engine.py"), "EdgeDecodeEngine")
        self.assertTrue(
            {
                "open_session",
                "prefill",
                "decode_one",
                "draft",
                "rollback",
                "close_session",
            }.issubset(_method_names(engine_cls))
        )

        service_cls = _find_class(_parse_module("service.py"), "DSSDEdgeService")
        self.assertTrue(
            {
                "open_session",
                "generate",
                "close_session",
            }.issubset(_method_names(service_cls))
        )

    def test_close_session_frees_blocks(self) -> None:
        engine_cls = _find_class(_parse_module("engine.py"), "EdgeDecodeEngine")
        close_session = next(
            node
            for node in engine_cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "close_session"
        )
        self.assertTrue(
            _contains_call(close_session, "self.scheduler.free_blocks"),
            "close_session 应先调用 self.scheduler.free_blocks(session)",
        )

    def test_prefill_allocate_blocks_uses_kv_cache_manager(self) -> None:
        scheduler_cls = _find_class(
            _parse_module("scheduler.py"),
            "EdgeSchedulerAdapter",
        )
        allocate_blocks = next(
            node
            for node in scheduler_cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "allocate_blocks"
        )
        self.assertTrue(
            _contains_call(allocate_blocks, "self._build_prefill_request"),
            "allocate_blocks 应先构造 prefill request",
        )
        self.assertTrue(
            _contains_call(allocate_blocks, "self.kv_cache_manager.allocate_slots"),
            "allocate_blocks 应调用 kv_cache_manager.allocate_slots",
        )

    def test_service_generate_uses_engine_and_verifier(self) -> None:
        service_cls = _find_class(_parse_module("service.py"), "DSSDEdgeService")
        generate = next(
            node
            for node in service_cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "generate"
        )
        self.assertTrue(
            _contains_call(generate, "self.decode_engine.open_session"),
            "generate 应打开本地 edge session",
        )
        self.assertTrue(
            _contains_call(generate, "self.verifier.open_session"),
            "generate 应打开 verifier session",
        )
        self.assertTrue(
            _contains_call(generate, "self.decode_engine.draft"),
            "generate 应调用本地 draft 逻辑",
        )
        self.assertTrue(
            _contains_call(generate, "self._has_eos_in_recent_committed_tokens"),
            "generate 应检查上一轮刚 commit 的整段 token 是否含 eos",
        )
        self.assertFalse(
            _contains_call(generate, "self._is_eos"),
            "generate 不应只用最后一个 token 判断 eos",
        )

    def test_edge_session_rollback_clamps_computed_prefix_by_kept_len(self) -> None:
        session_cls = _find_class(_parse_module("types.py"), "EdgeSession")
        rollback = next(
            node
            for node in session_cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "rollback"
        )
        self.assertFalse(
            _contains_binop(rollback, "num_computed_tokens", ast.Sub, "count"),
            "rollback 不应再按 num_computed_tokens - count 回退",
        )
        self.assertTrue(
            _contains_min_with_kept_len(rollback),
            "rollback 应按 min(self.num_computed_tokens, kept_len) 收缩 computed 前缀",
        )

    def test_round_state_preallocates_logits_buffer(self) -> None:
        round_state_cls = _find_class(_parse_module("types.py"), "EdgeRoundState")
        method_names = _method_names(round_state_cls)
        self.assertIn(
            "prepare_logits_buffer",
            method_names,
            "EdgeRoundState 应提供预分配 logits buffer 的方法",
        )
        self.assertIn(
            "logits_row_view",
            method_names,
            "EdgeRoundState 应提供按 step 取 row view 的方法",
        )
        append_step = next(
            node
            for node in round_state_cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "append_step"
        )
        self.assertFalse(
            _contains_call(append_step, "processed_logits.detach"),
            "append_step 不应再对 processed_logits 做 detach",
        )
        self.assertFalse(
            _contains_call(append_step, "processed_logits.detach.clone"),
            "append_step 不应再 clone processed_logits",
        )

    def test_draft_sampler_writes_processed_logits_into_buffer(self) -> None:
        sampler_cls = _find_class(_parse_module("sampler.py"), "DSSDEdgeDraftSampler")
        method_names = _method_names(sampler_cls)
        self.assertIn(
            "apply_sampling_params_into",
            method_names,
            "DSSDEdgeDraftSampler 应提供直写 buffer 的 apply_sampling_params_into",
        )
        sample_step = next(
            node
            for node in sampler_cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "sample_step"
        )
        self.assertTrue(
            _contains_call(sample_step, "self.apply_sampling_params_into"),
            "sample_step 应使用直写 buffer 的 apply_sampling_params_into",
        )
        self.assertFalse(
            _contains_attribute_chain(sample_step, "detach"),
            "sample_step 不应再通过 detach/clone 复制 logits",
        )
        self.assertFalse(
            _contains_attribute_chain(sample_step, "clone"),
            "sample_step 不应再通过 detach/clone 复制 logits",
        )

    def test_draft_prepares_round_logits_buffer_before_loop(self) -> None:
        engine_cls = _find_class(_parse_module("engine.py"), "EdgeDecodeEngine")
        draft = next(
            node
            for node in engine_cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "draft"
        )
        self.assertTrue(
            _contains_call(draft, "session.round_state.prepare_logits_buffer"),
            "draft 应在循环前准备 round logits buffer",
        )


if __name__ == "__main__":
    unittest.main()
