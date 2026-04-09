from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENGINE_FILE = ROOT / "engine.py"
STATE_BRIDGE_FILE = ROOT / "state_bridge.py"


def _get_class(tree: ast.AST, class_name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"找不到类: {class_name}")


def _get_method(class_node: ast.ClassDef, method_name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"找不到方法: {class_node.name}.{method_name}")


class BootstrapDemoTest(unittest.TestCase):
    def test_state_bridge_has_finish_prefill_without_commit(self) -> None:
        tree = ast.parse(
            STATE_BRIDGE_FILE.read_text(encoding="utf-8"),
            filename=str(STATE_BRIDGE_FILE),
        )
        bridge_cls = _get_class(tree, "VerifierStateBridge")
        method = _get_method(bridge_cls, "finish_prefill_without_commit")

        called_methods: list[str] = []
        has_token_ids_append = False
        for node in ast.walk(method):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            called_methods.append(node.func.attr)
            if node.func.attr != "append":
                continue
            value = node.func.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "token_ids"
                and isinstance(value.value, ast.Name)
                and value.value.id == "session"
            ):
                has_token_ids_append = True

        self.assertFalse(
            has_token_ids_append,
            "finish_prefill_without_commit() 不能把 bootstrap token 写进 "
            "session.token_ids",
        )
        self.assertNotIn(
            "sample_tokens",
            called_methods,
            "finish_prefill_without_commit() 不能调用 sample_tokens()",
        )

    def test_run_prefill_uses_sample_without_commit_then_prefill_postprocess(self) -> None:
        tree = ast.parse(ENGINE_FILE.read_text(encoding="utf-8"), filename=str(ENGINE_FILE))
        engine_cls = _get_class(tree, "VerifierDecodeEngine")
        method = _get_method(engine_cls, "run_prefill")

        called_methods: list[str] = []
        for node in ast.walk(method):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            called_methods.append(node.func.attr)

        self.assertIn(
            "_sample_bootstrap_token_without_commit",
            called_methods,
            "run_prefill() 必须先走只采样 bootstrap token 的路径",
        )
        self.assertIn(
            "_postprocess_prefill_only",
            called_methods,
            "run_prefill() 必须单独做 prefill-only postprocess",
        )


if __name__ == "__main__":
    unittest.main()
