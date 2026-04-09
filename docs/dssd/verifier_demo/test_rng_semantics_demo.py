from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SAMPLER_FILE = ROOT / "sampler.py"


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


class RngSemanticsDemoTest(unittest.TestCase):
    def test_compute_accept_length_uses_seeded_uniform_helper(self) -> None:
        tree = ast.parse(
            SAMPLER_FILE.read_text(encoding="utf-8"),
            filename=str(SAMPLER_FILE),
        )
        sampler_cls = _get_class(tree, "DSSDVerifierSampler")
        method = _get_method(sampler_cls, "compute_accept_length")

        called_methods: list[str] = []
        uses_torch_rand = False
        for node in ast.walk(method):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                called_methods.append(node.func.attr)
                if (
                    node.func.attr == "rand"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "torch"
                ):
                    uses_torch_rand = True

        self.assertIn(
            "_sample_accept_uniform",
            called_methods,
            "compute_accept_length() 必须通过 seed+position helper 取随机数",
        )
        self.assertFalse(
            uses_torch_rand,
            "compute_accept_length() 不能再直接调用 torch.rand()",
        )

    def test_seeded_uniform_helper_reads_seed_and_position(self) -> None:
        tree = ast.parse(
            SAMPLER_FILE.read_text(encoding="utf-8"),
            filename=str(SAMPLER_FILE),
        )
        sampler_cls = _get_class(tree, "DSSDVerifierSampler")
        method = _get_method(sampler_cls, "_sample_accept_uniform")

        attr_names: list[str] = []
        for node in ast.walk(method):
            if isinstance(node, ast.Attribute):
                attr_names.append(node.attr)

        self.assertIn(
            "seeds",
            attr_names,
            "_sample_accept_uniform() 必须读取 sampler.sampling_states.seeds",
        )
        self.assertIn(
            "positions",
            attr_names,
            "_sample_accept_uniform() 必须读取 input_batch.positions",
        )
        self.assertIn(
            "logits_indices",
            attr_names,
            "_sample_accept_uniform() 必须按当前 logit row 的 position 取随机数",
        )


if __name__ == "__main__":
    unittest.main()
