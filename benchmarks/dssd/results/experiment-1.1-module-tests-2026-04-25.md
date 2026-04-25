# Experiment 1.1 Module Test Verification

## Basic Information

| Field | Value |
|---|---|
| Date | 2026-04-25 |
| Git commit | `6647651e9` |
| Experiment | 1.1 模块化测试验证 |
| Purpose | 验证 DSSD 系统协议结构、采样逻辑、edge/verifier 引擎、服务层、通信层、网络模拟和启动入口等核心模块的功能正确性。 |

## Commands And Results

| Command | Result |
|---|---|
| `timeout 600s pytest tests/dssd --ignore=tests/dssd/service/test_real_smoke.py -q --import-mode=importlib` | `221 passed, 16 warnings in 108.95s` |
| `timeout 360s pytest tests/dssd/service/test_real_smoke.py -q --import-mode=importlib` | `9 passed, 16 warnings in 32.43s` |
| `timeout 300s pytest tests/benchmarks/test_dssd_system_benchmark.py -q` | `19 passed, 16 warnings in 4.28s` |
| `timeout 300s pytest tests/benchmarks/test_dssd_local_engine_benchmark.py -q` | `13 passed, 16 warnings in 3.05s` |

说明：`tests/dssd` 下多个子目录存在相同 basename 的测试文件，例如 `test_types.py`、`test_engine.py`。因此实际执行时使用 `--import-mode=importlib` 避免 pytest 默认 import 模式下的同名模块 collection 冲突。该参数不改变测试覆盖范围。`test_real_smoke.py` 单独执行，以避免真实 GPU runtime 初始化和清理对同一进程内长测试序列造成干扰。

## Thesis Table

| 测试类别 | 覆盖内容 | 测试结果 |
|---|---|---|
| 协议结构测试 | open session、verify round、close session 请求与响应结构 | 通过 |
| 采样逻辑测试 | draft 采样、贪心验证、拒绝分支、residual resample | 通过 |
| Edge 引擎测试 | prefill、draft、commit、rollback、session close | 通过 |
| Verifier 引擎测试 | open session、verify round、target-only generate、状态恢复 | 通过 |
| 服务层测试 | DSSD 主循环、target-only 路径、acceptance 统计 | 通过 |
| 通信层测试 | HTTP transport、in-process transport、网络模拟 | 通过 |
| 启动入口测试 | edge/verifier server 参数解析、接口返回、异常处理 | 通过 |

## Notes

本次实验中，初始执行 `pytest tests/dssd -v` 时由于 pytest collection 阶段同名测试模块冲突而中断；改用 `--import-mode=importlib` 后进入完整测试执行。随后发现并修复了一个 real smoke 测试夹具问题：该测试在同一 GPU runtime 中模拟 edge 和 verifier 时复用了同一个 `req_id`，导致 edge 侧状态写入 verifier 侧请求状态。修复后，本地 edge 请求和远端 verifier 请求在共享 runtime 测试环境中使用不同 `req_id`，更符合真实边云隔离部署语义。

## Conclusion

DSSD 系统核心模块及 benchmark 辅助模块测试均通过，可以作为论文中“系统正确性与可行性验证”的模块化测试结果。
