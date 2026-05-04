# Experiment 1.2 Greedy Output Consistency

## Basic Information

| Field | Value |
|---|---|
| Date | 2026-04-25 |
| Git commit | `6647651e9` |
| Experiment | 1.2 贪心解码输出一致性验证 |
| Purpose | 验证 DSSD 在贪心解码条件下是否保持 target-only 的输出 token 序列。 |

## Settings

| Field | Value |
|---|---|
| Official model pair | OPT-125M / OPT-6.7B |
| Edge model | `/root/autodl-tmp/opt-125m` |
| Verifier model | `/root/autodl-tmp/opt-6.7b` |
| Tokenizer | `/root/autodl-tmp/opt-125m` |
| Prompt file | `benchmarks/dssd/prompts/high_acceptance_opt.jsonl` |
| Prompt count | 5 |
| Prompt length | 128 tokens |
| Output length | 64 tokens |
| Decoding | greedy decoding |
| temperature | 0.0 |
| Edge model runner | v1 |
| Verifier model runner | v1 |

## Official Results

| Model Pair | Gamma | Prompt Count | Matched Prompts | Output Consistency Rate | Result JSON |
|---|---:|---:|---:|---:|---|
| OPT-125M / OPT-6.7B | 1 | 5 | 5 | 100% | `consistency-opt-125m-opt-6.7b-p128-o64-g1-after-cache-fix-20260425.json` |
| OPT-125M / OPT-6.7B | 2 | 5 | 5 | 100% | `consistency-opt-125m-opt-6.7b-p128-o64-g2-after-cache-fix-20260425.json` |
| OPT-125M / OPT-6.7B | 4 | 5 | 5 | 100% | `consistency-opt-125m-opt-6.7b-p128-o64-g4-after-cache-fix-20260425.json` |
| OPT-125M / OPT-6.7B | 8 | 5 | 5 | 100% | `consistency-opt-125m-opt-6.7b-p128-o64-g8-after-cache-fix-20260425.json` |

## Acceptance Metrics

| Gamma | Mean Draft Acceptance Rate | Mean Avg Accepted Len / Round |
|---:|---:|---:|
| 1 | 72.53% | 0.73 |
| 2 | 62.18% | 1.24 |
| 4 | 47.96% | 1.92 |
| 8 | 32.22% | 2.58 |

The packed verifier path was rerun after restoring the original packed verification implementation:

| Model Pair | Gamma | Prompt Count | Matched Prompts | Output Consistency Rate | Mean Draft Acceptance Rate | Result JSON |
|---|---:|---:|---:|---:|---:|---|
| OPT-125M / OPT-6.7B | 1 | 5 | 5 | 100% | 72.53% | `consistency-opt-125m-opt-6.7b-p128-o64-g1-packed-restored-20260425.json` |
| OPT-125M / OPT-6.7B | 2 | 5 | 5 | 100% | 62.18% | `consistency-opt-125m-opt-6.7b-p128-o64-g2-packed-restored-20260425.json` |
| OPT-125M / OPT-6.7B | 4 | 5 | 5 | 100% | 47.96% | `consistency-opt-125m-opt-6.7b-p128-o64-g4-packed-restored-20260425.json` |
| OPT-125M / OPT-6.7B | 8 | 5 | 5 | 100% | 32.22% | `consistency-opt-125m-opt-6.7b-p128-o64-g8-packed-restored-20260425.json` |

## Conclusion

For the official OPT-125M / OPT-6.7B model pair, DSSD output token sequences exactly matched target-only output token sequences for all tested gamma values. Therefore, the official correctness validation uses OPT-125M / OPT-6.7B as the controlled model pair.
