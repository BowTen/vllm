from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENGINE_FILE = ROOT / "engine.py"
STATE_BRIDGE_FILE = ROOT / "state_bridge.py"
TYPES_FILE = ROOT / "types.py"


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


class InterfaceCleanupDemoTest(unittest.TestCase):
    def test_open_session_result_no_longer_exposes_session(self) -> None:
        tree = ast.parse(TYPES_FILE.read_text(encoding="utf-8"), filename=str(TYPES_FILE))
        cls = _get_class(tree, "VerifierOpenSessionResult")

        field_names = []
        for node in cls.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                field_names.append(node.target.id)

        self.assertEqual(
            field_names,
            ["req_id", "bootstrap_token_id"],
            "VerifierOpenSessionResult 不应再暴露内部 session 字段",
        )

    def test_engine_open_session_does_not_return_session_field(self) -> None:
        tree = ast.parse(ENGINE_FILE.read_text(encoding="utf-8"), filename=str(ENGINE_FILE))
        cls = _get_class(tree, "VerifierDecodeEngine")
        method = _get_method(cls, "open_session")

        has_session_kw = False
        for node in ast.walk(method):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "VerifierOpenSessionResult":
                continue
            for kw in node.keywords:
                if kw.arg == "session":
                    has_session_kw = True
                    break

        self.assertFalse(
            has_session_kw,
            "VerifierDecodeEngine.open_session() 不应把 session 放进 "
            "VerifierOpenSessionResult",
        )

    def test_state_bridge_has_remove_round_state(self) -> None:
        tree = ast.parse(
            STATE_BRIDGE_FILE.read_text(encoding="utf-8"),
            filename=str(STATE_BRIDGE_FILE),
        )
        cls = _get_class(tree, "VerifierStateBridge")
        method = _get_method(cls, "remove_round_state")

        has_pop_call = False
        for node in ast.walk(method):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "pop":
                continue
            value = node.func.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "_round_states"
                and isinstance(value.value, ast.Name)
                and value.value.id == "self"
            ):
                has_pop_call = True
                break

        self.assertTrue(
            has_pop_call,
            "remove_round_state() 必须对 self._round_states 执行 pop",
        )

    def test_engine_close_session_removes_round_state(self) -> None:
        tree = ast.parse(ENGINE_FILE.read_text(encoding="utf-8"), filename=str(ENGINE_FILE))
        cls = _get_class(tree, "VerifierDecodeEngine")
        method = _get_method(cls, "close_session")

        has_remove_round_state_call = False
        for node in ast.walk(method):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "remove_round_state":
                continue
            value = node.func.value
            if (
                isinstance(value, ast.Attribute)
                and value.attr == "state_bridge"
                and isinstance(value.value, ast.Name)
                and value.value.id == "self"
            ):
                has_remove_round_state_call = True
                break

        self.assertTrue(
            has_remove_round_state_call,
            "VerifierDecodeEngine.close_session() 必须删除 round state 条目",
        )


if __name__ == "__main__":
    unittest.main()
