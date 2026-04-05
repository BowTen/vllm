# DSSD Edge-Verifier Serving

This page covers the minimal operator setup for DSSD in vLLM.

The first deployment step is to run edge and verifier on the same host so you
can validate the control plane before splitting them across a network boundary.

## Edge Mode

Run the user-facing edge instance with the draft model and a bound verifier URL.

```bash
vllm serve Qwen/Qwen3.5-0.8B \
  --dssd-config '{"enabled": true, "role": "edge", "gamma": 4, "verifier_url": "http://127.0.0.1:9001"}'
```

## Verifier Mode

Run the verifier instance with the target model.

```bash
vllm serve Qwen/Qwen3.5-7B-Instruct \
  --dssd-config '{"enabled": true, "role": "verifier", "gamma": 4}'
```

## Notes

- Keep the tokenizer and vocabulary aligned between edge and verifier.
- Use the same `gamma` on both sides.
- For lab experiments, you can keep both processes on one machine and let the
  transport layer simulate network behavior.
