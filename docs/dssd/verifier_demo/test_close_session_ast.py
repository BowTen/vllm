from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENGINE_FILE = ROOT / "engine.py"
SCHEDULER_FILE = ROOT / "scheduler.py"


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


def _load_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


class CloseSessionAstTest(unittest.TestCase):
    def test_scheduler_has_free_blocks_method(self) -> None:
        tree = _load_tree(SCHEDULER_FILE)
        scheduler_cls = _get_class(tree, "VerifierSchedulerAdapter")
        method = _get_method(scheduler_cls, "free_blocks")

        has_kv_free_call = False
        for node in ast.walk(method):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "free":
                continue
            value = node.func.value
            if not isinstance(value, ast.Attribute):
                continue
            if value.attr != "kv_cache_manager":
                continue
            if isinstance(value.value, ast.Name) and value.value.id == "self":
                has_kv_free_call = True
                break

        self.assertTrue(
            has_kv_free_call,
            "VerifierSchedulerAdapter.free_blocks() 必须调用 "
            "self.kv_cache_manager.free(...)",
        )

    def test_engine_close_session_calls_scheduler_free_blocks(self) -> None:
        tree = _load_tree(ENGINE_FILE)
        engine_cls = _get_class(tree, "VerifierDecodeEngine")
        method = _get_method(engine_cls, "close_session")

        has_scheduler_free_blocks_call = False
        for node in ast.walk(method):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "free_blocks":
                continue
            value = node.func.value
            if not isinstance(value, ast.Attribute):
                continue
            if value.attr != "scheduler":
                continue
            if isinstance(value.value, ast.Name) and value.value.id == "self":
                has_scheduler_free_blocks_call = True
                break

        self.assertTrue(
            has_scheduler_free_blocks_call,
            "VerifierDecodeEngine.close_session() 必须调用 "
            "self.scheduler.free_blocks(session)",
        )


if __name__ == "__main__":
    unittest.main()
