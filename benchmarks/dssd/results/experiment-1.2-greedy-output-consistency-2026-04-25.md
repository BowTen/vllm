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

## Supplementary Qwen3 Validation

Qwen3-0.6B / Qwen3-8B was also checked as a supplementary model pair. During the first diagnostic run, the empty-draft control (`gamma=0`) reached 5/5 consistency, while the packed external-draft verifier path did not reach full token-level consistency:

| Model Pair | Gamma | Verifier Runner | Matched Prompts | Output Consistency Rate | Result JSON |
|---|---:|---|---:|---:|---|
| Qwen3-0.6B / Qwen3-8B | 0 | v1 | 5/5 | 100% | `consistency-qwen3-0.6b-qwen3-8b-p128-o64-g0-debug-20260425.json` |
| Qwen3-0.6B / Qwen3-8B | 1 | v1 | 3/5 | 60% | `consistency-qwen3-0.6b-qwen3-8b-p128-o64-g1-debug-20260425.json` |
| Qwen3-0.6B / Qwen3-8B | 4 | v1 | 2/5 | 40% | `consistency-qwen3-0.6b-qwen3-8b-p128-o64-g4-after-cache-fix-20260425.json` |
| Qwen3-0.6B / Qwen3-8B | 4 | v2 | 0/5 | 0% | `consistency-qwen3-0.6b-qwen3-8b-p128-o64-g4-verifier-v2-debug-20260425.json` |

The issue was traced to the V1 verifier using a packed multi-token verify path for externally supplied draft tokens. The implementation was changed to use a target-only sequential verification step for each draft token in the V1 verifier. After the fix, Qwen3 reached full output consistency:

| Model Pair | Gamma | Verifier Runner | Matched Prompts | Output Consistency Rate | Result JSON |
|---|---:|---|---:|---:|---|
| Qwen3-0.6B / Qwen3-8B | 1 | v1 | 5/5 | 100% | `consistency-qwen3-0.6b-qwen3-8b-p128-o64-g1-after-sequential-verify-fix-20260425.json` |
| Qwen3-0.6B / Qwen3-8B | 4 | v1 | 5/5 | 100% | `consistency-qwen3-0.6b-qwen3-8b-p128-o64-g4-after-sequential-verify-fix-20260425.json` |

Final verification was rerun after the code cleanup:

| Model Pair | Gamma | Verifier Runner | Matched Prompts | Output Consistency Rate | Mean Draft Acceptance Rate | Result JSON |
|---|---:|---|---:|---:|---:|---|
| Qwen3-0.6B / Qwen3-8B | 1 | v1 | 5/5 | 100% | 57.35% | `consistency-qwen3-0.6b-qwen3-8b-p128-o64-g1-final-verify-20260425.json` |
| Qwen3-0.6B / Qwen3-8B | 4 | v1 | 5/5 | 100% | 31.63% | `consistency-qwen3-0.6b-qwen3-8b-p128-o64-g4-final-verify-20260425.json` |

## Conclusion

For the official OPT-125M / OPT-6.7B model pair, DSSD output token sequences exactly matched target-only output token sequences for all tested gamma values. The supplementary Qwen3-0.6B / Qwen3-8B validation also reached 100% output consistency after the V1 verifier fix. These results support the claim that the implemented DSSD path preserves target-model greedy decoding semantics under the tested configurations.
