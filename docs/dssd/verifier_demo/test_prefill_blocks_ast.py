from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCHEDULER_FILE = ROOT / "scheduler.py"
ENGINE_FILE = ROOT / "engine.py"


def _load_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


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


def _contains_call(method: ast.FunctionDef, dotted_name: str) -> bool:
    parts = dotted_name.split(".")
    for node in ast.walk(method):
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


class PrefillBlocksAstTest(unittest.TestCase):
    def test_scheduler_allocate_blocks_uses_prefill_request_and_kv_manager(self) -> None:
        tree = _load_tree(SCHEDULER_FILE)
        scheduler_cls = _get_class(tree, "VerifierSchedulerAdapter")
        method = _get_method(scheduler_cls, "allocate_blocks")

        self.assertTrue(
            _contains_call(method, "self._build_prefill_request"),
            "allocate_blocks() 应先构造 verifier prefill request",
        )
        self.assertTrue(
            _contains_call(method, "self.kv_cache_manager.allocate_slots"),
            "allocate_blocks() 应调用 kv_cache_manager.allocate_slots()",
        )

    def test_engine_open_session_passes_full_context_to_allocate_blocks(self) -> None:
        tree = _load_tree(ENGINE_FILE)
        engine_cls = _get_class(tree, "VerifierDecodeEngine")
        method = _get_method(engine_cls, "open_session")

        allocate_call: ast.Call | None = None
        for node in ast.walk(method):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr == "allocate_blocks":
                allocate_call = node
                break

        self.assertIsNotNone(
            allocate_call,
            "open_session() 必须调用 scheduler.allocate_blocks(...)",
        )
        assert allocate_call is not None
        kw_names = {kw.arg for kw in allocate_call.keywords}
        self.assertEqual(
            kw_names,
            {"req_id", "prompt_token_ids", "sampling_params", "lora_request"},
            "open_session() 应把 prefill block 申请所需上下文完整传给 scheduler.allocate_blocks()",
        )


if __name__ == "__main__":
    unittest.main()
