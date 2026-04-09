from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENGINE_FILE = ROOT / "engine.py"
SAMPLER_FILE = ROOT / "sampler.py"
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


class PostprocessSemanticsDemoTest(unittest.TestCase):
    def test_sampler_postprocess_tokens_do_not_include_committed_token(self) -> None:
        tree = ast.parse(SAMPLER_FILE.read_text(encoding="utf-8"), filename=str(SAMPLER_FILE))
        sampler_cls = _get_class(tree, "DSSDVerifierSampler")
        method = _get_method(sampler_cls, "build_postprocess_tokens")

        stores_committed_token = False
        for node in ast.walk(method):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Subscript):
                    continue
                value = node.value
                if (
                    isinstance(value, ast.Attribute)
                    and value.attr == "committed_token_id"
                    and isinstance(value.value, ast.Name)
                    and value.value.id == "request"
                ):
                    stores_committed_token = True
                    break

        self.assertFalse(
            stores_committed_token,
            "build_postprocess_tokens() 不能再把 request.committed_token_id "
            "写进 sampled_token_ids",
        )

    def test_build_num_sampled_matches_only_accepted_draft_len(self) -> None:
        tree = ast.parse(SAMPLER_FILE.read_text(encoding="utf-8"), filename=str(SAMPLER_FILE))
        sampler_cls = _get_class(tree, "DSSDVerifierSampler")
        method = _get_method(sampler_cls, "build_num_sampled")

        adds_one_to_accepted_len = False
        for node in ast.walk(method):
            if not isinstance(node, ast.BinOp):
                continue
            if not isinstance(node.op, ast.Add):
                continue
            left, right = node.left, node.right
            names = {n.id for n in (left, right) if isinstance(n, ast.Name)}
            constants = {
                n.value for n in (left, right) if isinstance(n, ast.Constant)
            }
            if "accepted_len" in names and 1 in constants:
                adds_one_to_accepted_len = True
                break

        self.assertFalse(
            adds_one_to_accepted_len,
            "build_num_sampled() 不应再返回 1 + accepted_len",
        )

    def test_engine_commits_current_input_before_postprocess(self) -> None:
        tree = ast.parse(ENGINE_FILE.read_text(encoding="utf-8"), filename=str(ENGINE_FILE))
        engine_cls = _get_class(tree, "VerifierDecodeEngine")
        method = _get_method(engine_cls, "_sample_with_dssd")

        called_methods: list[str] = []
        for node in ast.walk(method):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            called_methods.append(node.func.attr)

        self.assertIn(
            "commit_committed_token_before_postprocess",
            called_methods,
            "_sample_with_dssd() 调 postprocess() 前必须先手动提交 "
            "committed_token",
        )

    def test_state_bridge_exposes_commit_helper(self) -> None:
        tree = ast.parse(
            STATE_BRIDGE_FILE.read_text(encoding="utf-8"),
            filename=str(STATE_BRIDGE_FILE),
        )
        bridge_cls = _get_class(tree, "VerifierStateBridge")
        method = _get_method(bridge_cls, "commit_committed_token_before_postprocess")

        has_session_extend = False
        has_total_len_update = False
        for node in ast.walk(method):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if (
                    node.func.attr == "extend"
                    and isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "token_ids"
                ):
                    has_session_extend = True
            if isinstance(node, ast.AugAssign):
                if (
                    isinstance(node.target, ast.Attribute)
                    and node.target.attr == "total_len"
                    and isinstance(node.target.value, ast.Name)
                    and node.target.value.id == "session"
                ):
                    has_total_len_update = True

        self.assertTrue(
            has_session_extend,
            "commit_committed_token_before_postprocess() 必须推进 session.token_ids",
        )
        self.assertTrue(
            has_total_len_update,
            "commit_committed_token_before_postprocess() 必须推进 session.total_len",
        )


if __name__ == "__main__":
    unittest.main()
