# Distributed Draft-Model Speculative Decoding

This experimental flow splits speculative decoding into two services:

- edge node: OpenAI-compatible `vllm serve` process with the local draft model
- cloud node: internal `vllm spec-verifier` process with the target model

Both models must share the same tokenizer / vocabulary.

## Quick Start

On the verifier node:

```bash
CUDA_VISIBLE_DEVICES=1 vllm spec-verifier \
  --model Qwen/Qwen3-0.6B \
  --device cuda:0 \
  --dtype bfloat16 \
  --host 0.0.0.0 \
  --port 9000 \
  --scheduler-max-batch-size 8 \
  --scheduler-max-batch-tokens 32 \
  --scheduler-queue-timeout-ms 50 \
  --session-idle-timeout-s 30
```

On the edge node:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu-memory-utilization 0.7 \
  --speculative-config '{
    "method": "draft_model",
    "model": "Qwen/Qwen3-0.6B",
    "num_speculative_tokens": 4,
    "verifier_url": "http://127.0.0.1:9000",
    "draft_device": "cuda:0"
  }'
```

The OpenAI-compatible request path is unchanged; only the backend changes.

Optional verifier tuning flags:

- `--scheduler-max-batch-size`: cap the number of proposals per micro-batch
- `--scheduler-max-batch-tokens`: cap the summed draft-token count per micro-batch
- `--scheduler-queue-timeout-ms`: reject proposals that wait too long in the verifier queue
- `--session-idle-timeout-s`: reclaim cloud-side verifier sessions after inactivity
