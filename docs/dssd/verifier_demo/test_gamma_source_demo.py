from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENGINE_FILE = ROOT / "engine.py"
SERVICE_FILE = ROOT / "service.py"
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


class GammaSourceDemoTest(unittest.TestCase):
    def test_engine_initializes_fixed_gamma(self) -> None:
        tree = ast.parse(ENGINE_FILE.read_text(encoding="utf-8"), filename=str(ENGINE_FILE))
        engine_cls = _get_class(tree, "VerifierDecodeEngine")
        init_method = _get_method(engine_cls, "__init__")

        called_methods: list[str] = []
        for node in ast.walk(init_method):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                called_methods.append(node.func.attr)

        self.assertIn(
            "_resolve_fixed_gamma",
            called_methods,
            "VerifierDecodeEngine.__init__() 必须在初始化时解析固定 gamma",
        )

    def test_state_bridge_prepare_round_no_longer_reads_runner_gamma(self) -> None:
        tree = ast.parse(
            STATE_BRIDGE_FILE.read_text(encoding="utf-8"),
            filename=str(STATE_BRIDGE_FILE),
        )
        bridge_cls = _get_class(tree, "VerifierStateBridge")
        method = _get_method(bridge_cls, "prepare_round")

        attr_names: list[str] = []
        for node in ast.walk(method):
            if isinstance(node, ast.Attribute):
                attr_names.append(node.attr)

        self.assertNotIn(
            "num_speculative_steps",
            attr_names,
            "prepare_round() 不应再直接读取 model_runner.num_speculative_steps",
        )

    def test_service_gamma_is_bound_to_decode_engine_fixed_gamma(self) -> None:
        tree = ast.parse(
            SERVICE_FILE.read_text(encoding="utf-8"),
            filename=str(SERVICE_FILE),
        )
        service_cls = _get_class(tree, "DSSDVerifierService")
        init_method = _get_method(service_cls, "__init__")

        assigns_decode_engine_gamma = False
        for node in ast.walk(init_method):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "gamma"
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "gamma"
                    and isinstance(node.value.value, ast.Attribute)
                    and node.value.value.attr == "decode_engine"
                ):
                    assigns_decode_engine_gamma = True

        self.assertTrue(
            assigns_decode_engine_gamma,
            "DSSDVerifierService.__init__() 必须把 gamma 绑定到 decode_engine.gamma",
        )


if __name__ == "__main__":
    unittest.main()
