# DSSD Thesis Experiment Plan

本文档是本科毕业论文 DSSD 实验的执行计划和结果记录模板。实验目标集中在两个结论：

1. DSSD 的输出分布等价于 target 模型。
2. 在一定 edge-verifier 网络条件下，DSSD 的吞吐率可以超过 target-only 推理。

后续每次正式实验完成后，将原始 JSON、日志和汇总表补充到本文档对应实验项中，并把原始文件保存到 `benchmarks/dssd/results/`。

## 0. 统一实验配置

除非单个实验另有说明，所有实验使用下表配置。

| 项目 | 固定配置 |
|---|---|
| Draft / edge 模型 | OPT-125M，例如 `/root/autodl-tmp/opt-125m` 或 `facebook/opt-125m` |
| Target / verifier 模型 | OPT-6.7B，例如 `/root/autodl-tmp/opt-6.7b` 或 `facebook/opt-6.7b` |
| Tokenizer | 使用 OPT-125M tokenizer；OPT-125M 与 OPT-6.7B 词表一致 |
| Model runner | edge `v1`，verifier `v1` |
| GPU 分配 | edge: GPU 0，verifier: GPU 1 |
| prompt 文件 | `benchmarks/dssd/prompts/high_acceptance_opt.jsonl` |
| prompt 选择 | 默认使用前 5 条做正确性实验；性能实验使用第一条并截断 |
| prompt_len | 128 |
| 输出长度 | 正确性实验 64 tokens；性能实验 256 tokens |
| EOS 处理 | `ignore_eos=True`，强制生成固定长度，便于对比 |
| 执行模式 | `--no-enforce-eager` |
| repeats | 性能实验 `20`，warmup repeats `2` |
| max_model_len | `512` |
| max_num_batched_tokens | `128` |
| max_num_seqs | `2` |

正式运行前先设置本机模型路径：

```bash
export EDGE_MODEL=/root/autodl-tmp/opt-125m
export TARGET_MODEL=/root/autodl-tmp/opt-6.7b
export TOKENIZER=/root/autodl-tmp/opt-125m
export PROMPT_JSONL=benchmarks/dssd/prompts/high_acceptance_opt.jsonl
export RESULT_DIR=benchmarks/dssd/results
mkdir -p "$RESULT_DIR"
```

建议每次记录当前代码版本：

```bash
git rev-parse HEAD
git status --short
```

如果 `git status --short` 中有本次实验相关的未提交改动，需要在结果记录中写清楚。

## 1. 实验总表

| 编号 | 实验名称 | 证明目标 | 关键变量 | 主要指标 | 判定标准 |
|---|---|---|---|---|---|
| E1 | 贪心路径输出一致性 | DSSD 在 greedy decoding 下不改变 target 输出 | `gamma = 1, 2, 4, 6, 8` | 输出一致率、首个 mismatch 位置、接受率 | 所有 prompt 的 DSSD 输出 token 序列与 target-only 完全一致 |
| E2 | 非贪心路径参考一致性 | DSSD 非贪心采样路径与 DSSD reference 分布实现一致 | `temperature/top_p/top_k/seed/gamma` | 输出一致率、逐 case token 序列、拒绝恢复路径 | 相同 case、相同采样参数、相同 seed 下，DSSD 输出与 reference 完全一致 |
| E3 | gamma 与网络延迟性能分析 | DSSD 在部分网络条件下吞吐超过 target-only | `gamma` 与 DSSD 单向延迟 | DSSD token/s、0ms target-only token/s、speedup、平均接受长度、接受率 | 至少存在若干 latency/gamma 组合使 `speedup > 1.0` |

## 2. E1 贪心路径输出一致性

### 2.1 实验目的

验证 DSSD 的 greedy path 是否保持 target 模型语义。贪心采样下，verifier 对每个 draft token 做 target argmax 比较；若拒绝，则 edge 使用 verifier 返回的 target token。因此 DSSD 最终输出应与 target-only 贪心输出完全一致。

### 2.2 实验配置

| 项目 | 设置 |
|---|---|
| 模型组合 | OPT-125M / OPT-6.7B |
| 对比对象 | DSSD vs target-only |
| temperature | `0.0` |
| top_p | `1.0` |
| top_k | 默认值 |
| prompt_count | `5` |
| prompt_len | `128` |
| output_len | `64` |
| gamma | `1, 2, 4, 6, 8` |
| 网络模拟 | 关闭 |

### 2.3 执行命令

每个 `gamma` 单独运行一次，避免长生命周期 vLLM runtime teardown 影响后续实验。

```bash
for GAMMA in 1 2 4 6 8; do
  PYTHONPATH=$PWD .venv/bin/python benchmarks/dssd/benchmark_output_consistency.py \
    --edge-model "$EDGE_MODEL" \
    --verifier-model "$TARGET_MODEL" \
    --tokenizer "$TOKENIZER" \
    --prompt-jsonl "$PROMPT_JSONL" \
    --prompt-count 5 \
    --prompt-len 128 \
    --output-len 64 \
    --gamma "$GAMMA" \
    --edge-model-runner v1 \
    --verifier-model-runner v1 \
    --edge-cuda-visible-devices 0 \
    --verifier-cuda-visible-devices 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 512 \
    --max-num-batched-tokens 128 \
    --max-num-seqs 2 \
    --no-enforce-eager \
    --output-json "$RESULT_DIR/e1-greedy-consistency-opt125m-opt67b-p128-o64-g${GAMMA}.json"
done
```

### 2.4 结果记录表

| gamma | prompt_count | matched_prompts | 输出一致率 | mean draft acceptance | mean avg accepted len | 原始 JSON |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 5 | 5 | 100% | 0.7253 | 0.7253 | `benchmarks/dssd/results/e1-greedy-consistency-opt125m-opt67b-p128-o64-g1.json` |
| 2 | 5 | 5 | 100% | 0.6218 | 1.2435 | `benchmarks/dssd/results/e1-greedy-consistency-opt125m-opt67b-p128-o64-g2.json` |
| 4 | 5 | 5 | 100% | 0.4796 | 1.9183 | `benchmarks/dssd/results/e1-greedy-consistency-opt125m-opt67b-p128-o64-g4.json` |
| 6 | 5 | 5 | 100% | 0.3731 | 2.2387 | `benchmarks/dssd/results/e1-greedy-consistency-opt125m-opt67b-p128-o64-g6.json` |
| 8 | 5 | 5 | 100% | 0.3222 | 2.5779 | `benchmarks/dssd/results/e1-greedy-consistency-opt125m-opt67b-p128-o64-g8.json` |

### 2.5 论文中可用结论

若所有 `gamma` 的输出一致率均为 100%，可说明在贪心采样条件下，本文 DSSD 实现不会改变 target 模型的输出序列，验证了 open session、verify、commit、rollback 和 greedy rejection recovery 的端到端正确性。

## 3. E2 非贪心路径参考一致性

### 3.1 实验目的

验证 DSSD 的非贪心采样路径是否与 DSSD reference 完全一致。该实验不直接比较 target-only，因为非贪心采样的“单次输出完全一致”依赖随机数实现；这里使用项目中的简化 DSSD reference 作为标准实现，固定 seed、采样参数和 prompt token ids，要求 DSSD 与 reference 输出 token 序列完全一致。

### 3.2 实验配置

| 项目 | 设置 |
|---|---|
| 模型组合 | OPT-125M / OPT-6.7B |
| Reference 实现 | `experiments/dssd_correctness/generate_vllm_reference.py` |
| DSSD 实现 | edge server `/generate`，配合 `experiments/dssd_correctness/run_dssd_cases.py` |
| 比较器 | `experiments/dssd_correctness/compare_outputs.py` |
| case 数量 | 建议 5 条 prompt，每条 2 个 seed，每个 gamma 共 10 cases；两个 gamma 共 20 cases |
| prompt_len | `128` |
| max_tokens | `32` |
| gamma | `4` 和 `8` |
| sampling params | `temperature=0.8`、`top_p=0.95`、`top_k=50`、`ignore_eos=True` |
| seed | `0` 和 `1234` |
| 网络模拟 | 关闭 |

### 3.3 准备 cases 文件

DSSD edge `/generate` 接口使用 token ids。先把固定 prompt 转成 `prompt_token_ids`，生成 official cases 文件：

```bash
PYTHONPATH=$PWD .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(os.environ["TOKENIZER"], local_files_only=Path(os.environ["TOKENIZER"]).exists())
prompt_path = Path(os.environ["PROMPT_JSONL"])
out_path = Path(os.environ["RESULT_DIR"]) / "e2-nongreedy-cases-opt125m-opt67b-p128-o32.jsonl"

prompts = []
with prompt_path.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            prompts.append(json.loads(line)["prompt"])
        if len(prompts) == 5:
            break

with out_path.open("w", encoding="utf-8") as out:
    for prompt_idx, prompt in enumerate(prompts):
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)[:128]
        if len(token_ids) < 128:
            raise RuntimeError(f"prompt {prompt_idx} has only {len(token_ids)} tokens")
        for gamma in (4, 8):
            for seed in (0, 1234):
                out.write(json.dumps({
                    "case_id": f"opt-p{prompt_idx}-g{gamma}-seed{seed}",
                    "prompt_token_ids": token_ids,
                    "max_tokens": 32,
                    "gamma": gamma,
                    "seed": seed,
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "top_k": 50,
                    "ignore_eos": True,
                }) + "\n")
print(out_path)
PY
```

### 3.4 生成 reference 输出

```bash
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  -m experiments.dssd_correctness.generate_vllm_reference \
    --draft-model "$EDGE_MODEL" \
    --target-model "$TARGET_MODEL" \
    --tokenizer "$TOKENIZER" \
    --cases "$RESULT_DIR/e2-nongreedy-cases-opt125m-opt67b-p128-o32.jsonl" \
    --output "$RESULT_DIR/e2-reference-opt125m-opt67b-p128-o32.jsonl" \
    --device cuda \
    --dtype auto \
    --max-model-len 256 \
    --max-num-batched-tokens 128 \
    --max-num-seqs 2 \
    --gpu-memory-utilization 0.05 \
    --kv-cache-memory-bytes 536870912 \
    --logprobs-mode raw_logits \
    --no-enforce-eager
```

### 3.5 启动 verifier 和 edge server

终端 1 启动 verifier。下面以 `gamma=4` 为例；跑 `gamma=8` 时需要停止当前 verifier 和 edge，再把两个命令中的 `--gamma 4` 都改成 `--gamma 8`。

```bash
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m vllm.dssd.entrypoints.verifier_server \
  --host 127.0.0.1 \
  --port 18021 \
  --model "$TARGET_MODEL" \
  --model-runner-version v1 \
  --gamma 4 \
  --max-model-len 256 \
  --gpu-memory-utilization 0.9 \
  --kv-cache-memory-bytes 536870912 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 2 \
  --no-enforce-eager
```

终端 2 启动 edge。注意 verifier、edge 和当前 cases 的 `gamma` 必须一致。

```bash
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m vllm.dssd.entrypoints.edge_server \
  --host 127.0.0.1 \
  --port 6006 \
  --verifier-url http://127.0.0.1:18021 \
  --model "$EDGE_MODEL" \
  --model-runner-version v1 \
  --gamma 4 \
  --max-model-len 256 \
  --gpu-memory-utilization 0.9 \
  --kv-cache-memory-bytes 536870912 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 2 \
  --no-enforce-eager
```

### 3.6 运行 DSSD cases 并比较

如果当前 edge server 用 `gamma=4` 启动，就只运行 `case_id` 中含 `g4` 的 cases；`gamma=8` 同理。可用下面命令生成分组文件：

```bash
grep 'g4-seed' "$RESULT_DIR/e2-nongreedy-cases-opt125m-opt67b-p128-o32.jsonl" > "$RESULT_DIR/e2-cases-g4-o32.jsonl"
grep 'g8-seed' "$RESULT_DIR/e2-nongreedy-cases-opt125m-opt67b-p128-o32.jsonl" > "$RESULT_DIR/e2-cases-g8-o32.jsonl"
```

运行 DSSD：

```bash
PYTHONPATH=$PWD .venv/bin/python experiments/dssd_correctness/run_dssd_cases.py \
  --endpoint http://127.0.0.1:6006/generate \
  --cases "$RESULT_DIR/e2-cases-g4-o32.jsonl" \
  --output "$RESULT_DIR/e2-dssd-opt125m-opt67b-p128-o32-g4.jsonl"
```

比较输出：

```bash
grep 'g4-seed' "$RESULT_DIR/e2-reference-opt125m-opt67b-p128-o32.jsonl" > "$RESULT_DIR/e2-reference-g4-o32.jsonl"

PYTHONPATH=$PWD .venv/bin/python experiments/dssd_correctness/compare_outputs.py \
  --reference "$RESULT_DIR/e2-reference-g4-o32.jsonl" \
  --actual "$RESULT_DIR/e2-dssd-opt125m-opt67b-p128-o32-g4.jsonl"
```

对 `gamma=8` 重复启动、运行和比较步骤。

### 3.7 结果记录表

| gamma | cases | matched_cases | 输出一致率 | sampling params | reference JSONL | DSSD JSONL | 比较结果 |
|---:|---:|---:|---:|---|---|---|---|
| 4 | 10 | 10 | 100% | `temp=0.8, top_p=0.95, top_k=50` | `benchmarks/dssd/results/e2-reference-g4-o32.jsonl` | `benchmarks/dssd/results/e2-dssd-opt125m-opt67b-p128-o32-g4.jsonl` | passed: `DSSD outputs match exactly.`, see `benchmarks/dssd/results/e2-compare-g4-o32.txt` |
| 8 | 10 | 10 | 100% | `temp=0.8, top_p=0.95, top_k=50` | `benchmarks/dssd/results/e2-reference-g8-o32.jsonl` | `benchmarks/dssd/results/e2-dssd-opt125m-opt67b-p128-o32-g8.jsonl` | passed: `DSSD outputs match exactly.`, see `benchmarks/dssd/results/e2-compare-g8-o32.txt` |

### 3.8 论文中可用结论

若所有 cases 完全一致，可说明本文 DSSD 实现中的非贪心 verify、accept/reject 判断、residual resample 与 reference 实现一致，从实验上支持 DSSD 输出分布等价于 target 模型分布。

## 4. E3 gamma 与网络延迟性能分析

### 4.1 实验目的

评估 DSSD 在非贪心采样条件下，不同 draft 长度和 edge-verifier 网络延迟对吞吐的影响，找出 DSSD 超过 target-only 的网络条件和最佳 `gamma` 区间。这里的网络延迟只作用于 DSSD 的 edge-verifier 通信；target-only 表示 target 模型单独推理，不存在 DSSD 的多轮 edge-verifier 交互，因此所有 speedup 都使用 0ms target-only 基线计算。

### 4.2 实验配置

| 项目 | 设置 |
|---|---|
| 模型组合 | OPT-125M / OPT-6.7B |
| prompt | `high_acceptance_opt.jsonl` 第一条；通过 `/benchmark_complete` 执行时，OPT tokenizer 会自动加入 BOS，本实验使用前 127 个内容 token decode 成文本，使服务端 `prompt_token_count=128` |
| output_tokens | `256` |
| temperature | `0.1` |
| seed | `0` |
| gamma | `1, 2, 4, 6, 8, 10, 12` |
| DSSD 单向延迟 | `0, 10, 20, 30, 40 ms` |
| DSSD 带宽 | 固定 100 Mbps |
| target-only 网络设置 | 固定使用 0ms baseline，不参与 latency sweep |
| repeats | `20` |
| warmup_repeats | `2` |
| warmup_tokens | `0` |
| 主要吞吐指标 | measured repeats 的平均 `output_token_count / server_inference_seconds` |

带宽换算：

```text
100 Mbps = 12,500,000 bytes/s
```

### 4.3 启动服务

E3 使用 edge server 的 `/benchmark_complete` 接口测量 DSSD 和 target-only。DSSD 请求使用待评估的网络模拟参数；target-only 请求只用于获得 0ms baseline，后续所有 latency 下的 speedup 都复用该 baseline。

每个 DSSD `gamma` 单独启动一组 verifier/edge server，完成全部 latency 扫描后再切换下一个 `gamma`。target-only baseline 不依赖 draft token，但为了减少环境差异，可在同一服务启动后先测一次。

终端 1 启动 verifier。下面以 `GAMMA=4` 为例：

```bash
export GAMMA=4

PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m vllm.dssd.entrypoints.verifier_server \
  --host 127.0.0.1 \
  --port 18021 \
  --model "$TARGET_MODEL" \
  --model-runner-version v1 \
  --gamma "$GAMMA" \
  --max-model-len 512 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 2 \
  --no-enforce-eager
```

终端 2 启动 edge：

```bash
export GAMMA=4

PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m vllm.dssd.entrypoints.edge_server \
  --host 127.0.0.1 \
  --port 6006 \
  --verifier-url http://127.0.0.1:18021 \
  --model "$EDGE_MODEL" \
  --model-runner-version v1 \
  --gamma "$GAMMA" \
  --max-model-len 512 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 2 \
  --no-enforce-eager
```

### 4.4 执行 target-only baseline 与当前 gamma 的 latency sweep

终端 3 先执行 target-only 0ms baseline，然后执行当前 `GAMMA` 下的 DSSD latency sweep。完成后停止终端 1 和终端 2，修改 `GAMMA` 为 `1, 2, 4, 6, 8, 10, 12` 中的下一个值并重复。整理结果时，每个 latency 的 `target-only token/s` 都填该 0ms baseline，不使用带延迟的 target-only 结果。

```bash
export GAMMA=4

PYTHONPATH=$PWD .venv/bin/python - <<'PY'
import json
import os
import time
import urllib.request
from pathlib import Path
from transformers import AutoTokenizer

endpoint = "http://127.0.0.1:6006/benchmark_complete"
result_dir = Path(os.environ["RESULT_DIR"])
tokenizer = AutoTokenizer.from_pretrained(os.environ["TOKENIZER"], local_files_only=Path(os.environ["TOKENIZER"]).exists())
source_prompt = json.loads(Path(os.environ["PROMPT_JSONL"]).read_text(encoding="utf-8").splitlines()[0])["prompt"]
# OPT tokenizer 会在 edge server 的 tokenizer(prompt).input_ids 中加入 BOS。
# 因此这里使用 127 个内容 token，使服务端 prompt_token_count 等于 128。
prompt_token_ids = tokenizer.encode(source_prompt, add_special_tokens=False)[:127]
prompt = tokenizer.decode(prompt_token_ids)
gamma = int(os.environ["GAMMA"])

def run_mode(mode, *, latency_ms, bandwidth_mbps, output_name):
    records = []
    for repeat in range(22):
        payload = {
            "req_id": f"e3-temp01-{mode}-g{gamma}-lat{latency_ms}-r{repeat}-{time.time_ns()}",
            "prompt": prompt,
            "mode": mode,
            "sampling_params": {
                "max_tokens": 256,
                "temperature": 0.1,
                "seed": 0,
                "ignore_eos": True,
            },
            "network_simulation": {
                "latency_ms": latency_ms,
                "bandwidth_mbps": bandwidth_mbps,
            },
        }
        req = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=900) as resp:
            record = json.loads(resp.read().decode())
        if record["prompt_token_count"] != 128:
            raise RuntimeError(f"expected prompt_token_count=128, got {record['prompt_token_count']}")
        if record["output_token_count"] != 256:
            raise RuntimeError(f"expected output_token_count=256, got {record['output_token_count']}")
        record["warmup"] = repeat < 2
        record["repeat"] = repeat
        record["gamma"] = gamma
        record["temperature"] = 0.1
        record["seed"] = 0
        record["dssd_latency_ms"] = latency_ms
        record["bandwidth_mbps"] = bandwidth_mbps
        records.append(record)

    out = result_dir / output_name
    out.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    measured = [r for r in records if not r["warmup"]]
    tps = [r["output_token_count"] / r["server_inference_seconds"] for r in measured]
    accept = [r["draft_acceptance_rate"] for r in measured if r["draft_acceptance_rate"] is not None]
    avg_accept_len = [r["avg_accepted_len_per_round"] for r in measured if r["avg_accepted_len_per_round"] is not None]
    summary = {
        "mode": mode,
        "gamma": gamma,
        "latency_ms": latency_ms,
        "bandwidth_mbps": bandwidth_mbps,
        "mean_tokens_per_s": sum(tps) / len(tps),
        "mean_draft_acceptance_rate": None if not accept else sum(accept) / len(accept),
        "mean_avg_accepted_len_per_round": None if not avg_accept_len else sum(avg_accept_len) / len(avg_accept_len),
        "output": str(out),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return summary

target_baseline = run_mode(
    "target_only",
    latency_ms=0,
    bandwidth_mbps=0,
    output_name=f"e3-temp01-target-only-opt125m-opt67b-p128-o256-g{gamma}-0ms-baseline.jsonl",
)

for lat in (0, 10, 20, 30, 40):
    dssd_summary = run_mode(
        "dssd",
        latency_ms=lat,
        bandwidth_mbps=100,
        output_name=f"e3-temp01-dssd-opt125m-opt67b-p128-o256-g{gamma}-lat{lat}ms-100mbps.jsonl",
    )
    print(json.dumps({
        "gamma": gamma,
        "dssd_latency_ms": lat,
        "target_only_baseline_tokens_per_s": target_baseline["mean_tokens_per_s"],
        "dssd_tokens_per_s": dssd_summary["mean_tokens_per_s"],
        "speedup_vs_0ms_target_only": (
            dssd_summary["mean_tokens_per_s"]
            / target_baseline["mean_tokens_per_s"]
        ),
    }, ensure_ascii=False))
PY
```

### 4.5 需要记录的原始指标

| 指标 | 来源 |
|---|---|
| DSSD tokens/s | DSSD JSONL measured repeats 的平均 `output_token_count / server_inference_seconds` |
| target-only tokens/s | 0ms target-only baseline JSONL 中 measured repeats 的平均 `output_token_count / server_inference_seconds` |
| speedup | `DSSD tokens/s / 0ms target-only tokens/s` |
| 平均接受长度 | DSSD JSONL measured repeats 的平均 `avg_accepted_len_per_round` |
| 接受率 | DSSD JSONL measured repeats 的平均 `draft_acceptance_rate` |
| all-accept round rate | DSSD JSONL measured repeats 的平均 `all_accept_round_rate` |
| draft round 数 | DSSD JSONL measured repeats 的平均 `total_rounds` |

### 4.6 结果记录表：100 Mbps

执行日期：2026-05-04 至 2026-05-05。该组实验保持 E3 的 prompt、模型、带宽、输出长度和 repeats 配置不变，将采样温度改为 `temperature=0.1`，固定 `seed=0`、`ignore_eos=True`。每个 JSONL 文件包含 22 条记录，其中前 2 条为 warmup，后 20 条用于计算均值。第一轮覆盖 `gamma=1,2,4,6,8`；补充实验继续增加 `gamma=10,12`。

0ms target-only baseline 原始结果：`benchmarks/dssd/results/e3-temp01-target-only-opt125m-opt67b-p128-o256-g1-0ms-baseline.jsonl`、`benchmarks/dssd/results/e3-temp01-target-only-opt125m-opt67b-p128-o256-g2-0ms-baseline.jsonl`、`benchmarks/dssd/results/e3-temp01-target-only-opt125m-opt67b-p128-o256-g4-0ms-baseline.jsonl`、`benchmarks/dssd/results/e3-temp01-target-only-opt125m-opt67b-p128-o256-g6-0ms-baseline.jsonl`、`benchmarks/dssd/results/e3-temp01-target-only-opt125m-opt67b-p128-o256-g8-0ms-baseline.jsonl`、`benchmarks/dssd/results/e3-temp01-target-only-opt125m-opt67b-p128-o256-g10-0ms-baseline.jsonl`、`benchmarks/dssd/results/e3-temp01-target-only-opt125m-opt67b-p128-o256-g12-0ms-baseline.jsonl`。

| DSSD latency(ms) | gamma | 0ms target-only token/s | DSSD token/s | speedup vs 0ms target-only | draft acceptance | avg accepted len | all-accept round | rounds | 原始结果 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 1 | 102.09 | 119.19 | 1.17 | 0.947 | 0.95 | 0.947 | 131.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g1-lat0ms-100mbps.jsonl` |
| 10 | 1 | 102.09 | 51.89 | 0.51 | 0.947 | 0.95 | 0.947 | 131.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g1-lat10ms-100mbps.jsonl` |
| 20 | 1 | 102.09 | 33.46 | 0.33 | 0.947 | 0.95 | 0.947 | 131.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g1-lat20ms-100mbps.jsonl` |
| 30 | 1 | 102.09 | 24.65 | 0.24 | 0.947 | 0.95 | 0.947 | 131.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g1-lat30ms-100mbps.jsonl` |
| 40 | 1 | 102.09 | 19.61 | 0.19 | 0.947 | 0.95 | 0.947 | 131.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g1-lat40ms-100mbps.jsonl` |
| 0 | 2 | 101.95 | 170.88 | 1.68 | 0.966 | 1.93 | 0.954 | 87.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g2-lat0ms-100mbps.jsonl` |
| 10 | 2 | 101.95 | 74.30 | 0.73 | 0.966 | 1.93 | 0.954 | 87.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g2-lat10ms-100mbps.jsonl` |
| 20 | 2 | 101.95 | 48.83 | 0.48 | 0.966 | 1.93 | 0.954 | 87.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g2-lat20ms-100mbps.jsonl` |
| 30 | 2 | 101.95 | 36.24 | 0.36 | 0.966 | 1.93 | 0.954 | 87.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g2-lat30ms-100mbps.jsonl` |
| 40 | 2 | 101.95 | 28.84 | 0.28 | 0.966 | 1.93 | 0.954 | 87.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g2-lat40ms-100mbps.jsonl` |
| 0 | 4 | 101.63 | 177.30 | 1.74 | 0.843 | 3.37 | 0.797 | 59.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g4-lat0ms-100mbps.jsonl` |
| 10 | 4 | 101.63 | 92.06 | 0.91 | 0.843 | 3.37 | 0.797 | 59.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g4-lat10ms-100mbps.jsonl` |
| 20 | 4 | 101.63 | 63.80 | 0.63 | 0.843 | 3.37 | 0.797 | 59.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g4-lat20ms-100mbps.jsonl` |
| 30 | 4 | 101.63 | 48.70 | 0.48 | 0.843 | 3.37 | 0.797 | 59.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g4-lat30ms-100mbps.jsonl` |
| 40 | 4 | 101.63 | 39.50 | 0.39 | 0.843 | 3.37 | 0.797 | 59.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g4-lat40ms-100mbps.jsonl` |
| 0 | 6 | 100.30 | 182.93 | 1.82 | 0.748 | 4.49 | 0.660 | 47.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g6-lat0ms-100mbps.jsonl` |
| 10 | 6 | 100.30 | 102.05 | 1.02 | 0.748 | 4.49 | 0.660 | 47.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g6-lat10ms-100mbps.jsonl` |
| 20 | 6 | 100.30 | 72.67 | 0.72 | 0.748 | 4.49 | 0.660 | 47.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g6-lat20ms-100mbps.jsonl` |
| 30 | 6 | 100.30 | 56.69 | 0.57 | 0.748 | 4.49 | 0.660 | 47.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g6-lat30ms-100mbps.jsonl` |
| 40 | 6 | 100.30 | 46.54 | 0.46 | 0.748 | 4.49 | 0.660 | 47.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g6-lat40ms-100mbps.jsonl` |
| 0 | 8 | 99.92 | 225.86 | 2.26 | 0.796 | 6.37 | 0.714 | 35.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g8-lat0ms-100mbps.jsonl` |
| 10 | 8 | 99.92 | 129.83 | 1.30 | 0.796 | 6.37 | 0.714 | 35.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g8-lat10ms-100mbps.jsonl` |
| 20 | 8 | 99.92 | 93.70 | 0.94 | 0.796 | 6.37 | 0.714 | 35.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g8-lat20ms-100mbps.jsonl` |
| 30 | 8 | 99.92 | 73.34 | 0.73 | 0.796 | 6.37 | 0.714 | 35.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g8-lat30ms-100mbps.jsonl` |
| 40 | 8 | 99.92 | 60.31 | 0.60 | 0.796 | 6.37 | 0.714 | 35.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g8-lat40ms-100mbps.jsonl` |
| 0 | 10 | 99.66 | 235.02 | 2.36 | 0.755 | 7.55 | 0.677 | 31.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g10-lat0ms-100mbps.jsonl` |
| 10 | 10 | 99.66 | 135.52 | 1.36 | 0.755 | 7.55 | 0.677 | 31.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g10-lat10ms-100mbps.jsonl` |
| 20 | 10 | 99.66 | 98.91 | 0.99 | 0.755 | 7.55 | 0.677 | 31.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g10-lat20ms-100mbps.jsonl` |
| 30 | 10 | 99.66 | 79.03 | 0.79 | 0.755 | 7.55 | 0.677 | 31.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g10-lat30ms-100mbps.jsonl` |
| 40 | 10 | 99.66 | 65.47 | 0.66 | 0.755 | 7.55 | 0.677 | 31.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g10-lat40ms-100mbps.jsonl` |
| 0 | 12 | 99.54 | 187.91 | 1.89 | 0.573 | 6.88 | 0.515 | 33.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g12-lat0ms-100mbps.jsonl` |
| 10 | 12 | 99.54 | 115.11 | 1.16 | 0.573 | 6.88 | 0.515 | 33.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g12-lat10ms-100mbps.jsonl` |
| 20 | 12 | 99.54 | 86.80 | 0.87 | 0.573 | 6.88 | 0.515 | 33.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g12-lat20ms-100mbps.jsonl` |
| 30 | 12 | 99.54 | 70.22 | 0.71 | 0.573 | 6.88 | 0.515 | 33.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g12-lat30ms-100mbps.jsonl` |
| 40 | 12 | 99.54 | 58.63 | 0.59 | 0.573 | 6.88 | 0.515 | 33.0 | `benchmarks/dssd/results/e3-temp01-dssd-opt125m-opt67b-p128-o256-g12-lat40ms-100mbps.jsonl` |

#### 4.6.1 gamma x latency 吞吐率矩阵

单元格为 DSSD token/s，带宽固定为 100 Mbps。

| gamma \ DSSD latency(ms) | 0 | 10 | 20 | 30 | 40 |
|---:|---:|---:|---:|---:|---:|
| 1 | 119.19 | 51.89 | 33.46 | 24.65 | 19.61 |
| 2 | 170.88 | 74.30 | 48.83 | 36.24 | 28.84 |
| 4 | 177.30 | 92.06 | 63.80 | 48.70 | 39.50 |
| 6 | 182.93 | 102.05 | 72.67 | 56.69 | 46.54 |
| 8 | 225.86 | 129.83 | 93.70 | 73.34 | 60.31 |
| 10 | 235.02 | 135.52 | 98.91 | 79.03 | 65.47 |
| 12 | 187.91 | 115.11 | 86.80 | 70.22 | 58.63 |

### 4.7 结果记录表：每个 latency 的最优 gamma

| DSSD latency(ms) | best gamma | 0ms target-only token/s | best DSSD token/s | best speedup vs 0ms target-only | draft acceptance | avg accepted len |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 10 | 99.66 | 235.02 | 2.36 | 0.755 | 7.55 |
| 10 | 10 | 99.66 | 135.52 | 1.36 | 0.755 | 7.55 |
| 20 | 10 | 99.66 | 98.91 | 0.99 | 0.755 | 7.55 |
| 30 | 10 | 99.66 | 79.03 | 0.79 | 0.755 | 7.55 |
| 40 | 10 | 99.66 | 65.47 | 0.66 | 0.755 | 7.55 |

### 4.8 论文图表规划

| 图表 | 数据来源 | 展示内容 | 论文表达重点 |
|---|---|---|---|
| 图 1：gamma x latency 吞吐率热力图 | 4.6 表 | 每个 `gamma` 与 latency 下的 DSSD token/s 或 speedup | 说明 DSSD 性能随网络延迟上升而下降，且最佳 `gamma` 与网络条件相关 |
| 图 2：每个 latency 下最优 gamma 的 speedup 曲线 | 4.7 表 | 横轴 DSSD latency，纵轴 best speedup vs 0ms target-only | 说明 DSSD 在低延迟或合适 `gamma` 下可超过 target-only，并展示收益边界 |

### 4.9 论文中可用结论

在 `temperature=0.1` 下，`gamma=10` 是本组所有 latency 中的最优配置。`gamma=10` 在 0ms、10ms 下的 speedup 分别为 2.36、1.36，20ms 时为 0.99，接近 target-only 但略低；30ms 和 40ms 时分别下降到 0.79、0.66。因此该采样配置下，收益边界位于 10ms 到 20ms 单向延迟之间。`gamma=12` 的接受率最低，为 0.573，但吞吐低于 `gamma=8` 和 `gamma=10`，说明较低接受率并不必然带来更高吞吐，仍需结合平均轮数和单轮 draft 长度选择 `gamma`。

## 5. 每次实验后的记录要求

每个结果文件或结果小节必须记录：

| 字段 | 说明 |
|---|---|
| 实验日期 | 例如 `2026-05-04` |
| Git commit | `git rev-parse HEAD` 输出 |
| 工作区状态 | `git status --short` 输出；如有本地改动需说明 |
| 硬件 | GPU 型号、GPU 数量、CUDA 版本 |
| 模型路径 | edge、verifier、tokenizer 实际路径 |
| 命令 | 完整命令，不省略参数 |
| 采样参数 | temperature、top_p、top_k、seed、ignore_eos、max_tokens |
| 网络参数 | latency、bandwidth、是否对称 |
| 原始文件 | JSON、JSONL 或 log 的相对路径 |
| 摘要结果 | 一致率、token/s、speedup、接受率、平均接受长度 |
| 异常说明 | OOM、server 重启、首轮异常、复跑原因 |

建议原始文件命名格式：

```text
<实验编号>-<模式>-opt125m-opt67b-p<prompt_len>-o<output_len>-g<gamma>-lat<latency>ms-<bandwidth>-<日期>.json
```

## 6. 最终论文结果汇总占位

### 6.1 正确性结论

| 实验 | 输出一致率 | 覆盖路径 | 结论 |
|---|---:|---|---|
| E1 贪心路径 |  | greedy accept/reject、commit/rollback |  |
| E2 非贪心路径 |  | random sampling、residual resample |  |

### 6.2 性能结论

| DSSD 网络条件 | 最优 gamma | 0ms target-only token/s | DSSD token/s | speedup vs 0ms target-only | 结论 |
|---|---:|---:|---:|---:|---|
| 0ms, 100 Mbps |  |  |  |  |  |
| 10ms, 100 Mbps |  |  |  |  |  |
| 20ms, 100 Mbps |  |  |  |  |  |
| 30ms, 100 Mbps |  |  |  |  |  |
| 40ms, 100 Mbps |  |  |  |  |  |
