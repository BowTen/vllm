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
| max_tokens | `64` |
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
out_path = Path(os.environ["RESULT_DIR"]) / "e2-nongreedy-cases-opt125m-opt67b-p128-o64.jsonl"

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
                    "max_tokens": 64,
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
  experiments/dssd_correctness/generate_vllm_reference.py \
    --draft-model "$EDGE_MODEL" \
    --target-model "$TARGET_MODEL" \
    --tokenizer "$TOKENIZER" \
    --cases "$RESULT_DIR/e2-nongreedy-cases-opt125m-opt67b-p128-o64.jsonl" \
    --output "$RESULT_DIR/e2-reference-opt125m-opt67b-p128-o64.jsonl" \
    --device cuda \
    --dtype auto \
    --max-model-len 512 \
    --max-num-batched-tokens 128 \
    --max-num-seqs 2 \
    --gpu-memory-utilization 0.9 \
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
  --max-model-len 512 \
  --gpu-memory-utilization 0.9 \
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
  --max-model-len 512 \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 128 \
  --max-num-seqs 2 \
  --no-enforce-eager
```

### 3.6 运行 DSSD cases 并比较

如果当前 edge server 用 `gamma=4` 启动，就只运行 `case_id` 中含 `g4` 的 cases；`gamma=8` 同理。可用下面命令生成分组文件：

```bash
grep 'g4-seed' "$RESULT_DIR/e2-nongreedy-cases-opt125m-opt67b-p128-o64.jsonl" > "$RESULT_DIR/e2-cases-g4.jsonl"
grep 'g8-seed' "$RESULT_DIR/e2-nongreedy-cases-opt125m-opt67b-p128-o64.jsonl" > "$RESULT_DIR/e2-cases-g8.jsonl"
```

运行 DSSD：

```bash
PYTHONPATH=$PWD .venv/bin/python experiments/dssd_correctness/run_dssd_cases.py \
  --endpoint http://127.0.0.1:6006/generate \
  --cases "$RESULT_DIR/e2-cases-g4.jsonl" \
  --output "$RESULT_DIR/e2-dssd-opt125m-opt67b-p128-o64-g4.jsonl"
```

比较输出：

```bash
grep 'g4-seed' "$RESULT_DIR/e2-reference-opt125m-opt67b-p128-o64.jsonl" > "$RESULT_DIR/e2-reference-g4.jsonl"

PYTHONPATH=$PWD .venv/bin/python experiments/dssd_correctness/compare_outputs.py \
  --reference "$RESULT_DIR/e2-reference-g4.jsonl" \
  --actual "$RESULT_DIR/e2-dssd-opt125m-opt67b-p128-o64-g4.jsonl"
```

对 `gamma=8` 重复启动、运行和比较步骤。

### 3.7 结果记录表

| gamma | cases | matched_cases | 输出一致率 | sampling params | reference JSONL | DSSD JSONL | 比较结果 |
|---:|---:|---:|---:|---|---|---|---|
| 4 | 10 | 8 | 80% | `temp=0.8, top_p=0.95, top_k=50` | `benchmarks/dssd/results/e2-reference-g4.jsonl` | `benchmarks/dssd/results/e2-dssd-opt125m-opt67b-p128-o64-g4-rerun.jsonl` | failed: 2 mismatches, see `benchmarks/dssd/results/e2-compare-g4.txt` |
| 8 | 10 | 0 | 0% | `temp=0.8, top_p=0.95, top_k=50` | `benchmarks/dssd/results/e2-reference-g8.jsonl` | `benchmarks/dssd/results/e2-dssd-opt125m-opt67b-p128-o64-g8.jsonl` | failed: 10 mismatches, see `benchmarks/dssd/results/e2-compare-g8.txt` |

### 3.8 论文中可用结论

若所有 cases 完全一致，可说明本文 DSSD 实现中的非贪心 verify、accept/reject 判断、residual resample 与 reference 实现一致，从实验上支持 DSSD 输出分布等价于 target 模型分布。

## 4. E3 gamma 与网络延迟性能分析

### 4.1 实验目的

评估 DSSD 在不同 draft 长度和 edge-verifier 网络延迟下的吞吐表现，找出 DSSD 超过 target-only 的网络条件和最佳 `gamma` 区间。这里的网络延迟只应作用于 DSSD 的 edge-verifier 通信；target-only 表示 target 模型单独推理，不存在 DSSD 的多轮 edge-verifier 交互，因此所有 speedup 都使用 0ms target-only 基线计算。

### 4.2 实验配置

| 项目 | 设置 |
|---|---|
| 模型组合 | OPT-125M / OPT-6.7B |
| prompt | `high_acceptance_opt.jsonl` 第一条，截断到 128 tokens |
| output_tokens | `256` |
| temperature | `0.0` |
| gamma | `1, 2, 4, 6, 8` |
| DSSD 单向延迟 | `0, 10, 20, 30, 40, 50 ms` |
| DSSD 带宽 | 固定 100 Mbps；同时建议保留 unlimited 作为对照 |
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

终端 3 先执行 target-only 0ms baseline，然后执行当前 `GAMMA` 下的 DSSD latency sweep。完成后停止终端 1 和终端 2，修改 `GAMMA` 为 `1, 2, 4, 6, 8` 中的下一个值并重复。整理结果时，每个 latency 的 `target-only token/s` 都填该 0ms baseline，不使用带延迟的 target-only 结果。

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
prompt_token_ids = tokenizer.encode(source_prompt, add_special_tokens=False)[:128]
prompt = tokenizer.decode(prompt_token_ids)
gamma = int(os.environ["GAMMA"])

def run_mode(mode, *, latency_ms, bandwidth_mbps, output_name):
    records = []
    for repeat in range(22):
        payload = {
            "req_id": f"e3-{mode}-g{gamma}-lat{latency_ms}-r{repeat}-{time.time_ns()}",
            "prompt": prompt,
            "mode": mode,
            "sampling_params": {
                "max_tokens": 256,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            "network_simulation": {
                "latency_ms": latency_ms,
                "bandwidth_mbps": bandwidth_mbps,
            },
        }
        req = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=600) as resp:
            record = json.loads(resp.read().decode())
        if record["prompt_token_count"] != 128:
            raise RuntimeError(f"expected prompt_token_count=128, got {record['prompt_token_count']}")
        record["warmup"] = repeat < 2
        record["gamma"] = gamma
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
    output_name=f"e3-target-only-opt125m-opt67b-p128-o256-0ms-baseline.jsonl",
)

for lat in (0, 10, 20, 30, 40, 50):
    dssd_summary = run_mode(
        "dssd",
        latency_ms=lat,
        bandwidth_mbps=100,
        output_name=f"e3-dssd-opt125m-opt67b-p128-o256-g{gamma}-lat{lat}ms-100mbps.jsonl",
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

### 4.6 结果记录表：100 Mbps

| DSSD latency(ms) | gamma | 0ms target-only token/s | DSSD token/s | speedup vs 0ms target-only | draft acceptance | avg accepted len | 原始结果 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 1 |  |  |  |  |  |  |
| 0 | 2 |  |  |  |  |  |  |
| 0 | 4 |  |  |  |  |  |  |
| 0 | 6 |  |  |  |  |  |  |
| 0 | 8 |  |  |  |  |  |  |
| 10 | 1 |  |  |  |  |  |  |
| 10 | 2 |  |  |  |  |  |  |
| 10 | 4 |  |  |  |  |  |  |
| 10 | 6 |  |  |  |  |  |  |
| 10 | 8 |  |  |  |  |  |  |
| 20 | 1 |  |  |  |  |  |  |
| 20 | 2 |  |  |  |  |  |  |
| 20 | 4 |  |  |  |  |  |  |
| 20 | 6 |  |  |  |  |  |  |
| 20 | 8 |  |  |  |  |  |  |
| 30 | 1 |  |  |  |  |  |  |
| 30 | 2 |  |  |  |  |  |  |
| 30 | 4 |  |  |  |  |  |  |
| 30 | 6 |  |  |  |  |  |  |
| 30 | 8 |  |  |  |  |  |  |
| 40 | 1 |  |  |  |  |  |  |
| 40 | 2 |  |  |  |  |  |  |
| 40 | 4 |  |  |  |  |  |  |
| 40 | 6 |  |  |  |  |  |  |
| 40 | 8 |  |  |  |  |  |  |
| 50 | 1 |  |  |  |  |  |  |
| 50 | 2 |  |  |  |  |  |  |
| 50 | 4 |  |  |  |  |  |  |
| 50 | 6 |  |  |  |  |  |  |
| 50 | 8 |  |  |  |  |  |  |

### 4.7 结果记录表：每个 latency 的最优 gamma

| DSSD latency(ms) | best gamma | 0ms target-only token/s | best DSSD token/s | best speedup vs 0ms target-only | draft acceptance | avg accepted len |
|---:|---:|---:|---:|---:|---:|---:|
| 0 |  |  |  |  |  |  |
| 10 |  |  |  |  |  |  |
| 20 |  |  |  |  |  |  |
| 30 |  |  |  |  |  |  |
| 40 |  |  |  |  |  |  |
| 50 |  |  |  |  |  |  |

### 4.8 论文图表规划

| 图表 | 数据来源 | 展示内容 | 论文表达重点 |
|---|---|---|---|
| 图 1：gamma x latency 吞吐率热力图 | 4.6 表 | 每个 `gamma` 与 latency 下的 DSSD token/s 或 speedup | 说明 DSSD 性能随网络延迟上升而下降，且最佳 `gamma` 与网络条件相关 |
| 图 2：每个 latency 下最优 gamma 的 speedup 曲线 | 4.7 表 | 横轴 DSSD latency，纵轴 best speedup vs 0ms target-only | 说明 DSSD 在低延迟或合适 `gamma` 下可超过 target-only，并展示收益边界 |

### 4.9 论文中可用结论

若低延迟条件下存在 `speedup > 1.0`，可说明 DSSD 在边云链路足够好、draft 接受率较高时能提升系统吞吐。若高延迟下 speedup 下降到 1 以下，可进一步说明 DSSD 收益受网络往返次数影响，需要根据链路条件选择合适 `gamma`。

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
| 50ms, 100 Mbps |  |  |  |  |  |
