# DSSD Edge-Verifier Serving

This page covers the current control-plane bring-up for DSSD in vLLM.

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
  --port 9001 \
  --dssd-config '{"enabled": true, "role": "verifier", "gamma": 4}'
```

## Notes

- Keep the tokenizer and vocabulary aligned between edge and verifier.
- Use the same `gamma` on both sides.
- The current branch wires `bind/create_session/verify_round/close_session`
  over HTTP and enters the DSSD edge control path from `/v1/chat/completions`.
- The edge path now completes a single synchronous DSSD round:
  `draft_round -> verify_round -> optional residual resample -> response`.
- The edge path closes the verifier session at request end and enforces
  `max_tokens` / `stop_token_ids` on the returned completion.
- The current draft/verifier worker hooks are still placeholder
  implementations; this is a serving-path bring-up, not the final algorithmic
  implementation.
- Streaming chat completions are still not implemented in DSSD edge mode.
- Network simulation is currently limited to latency and bandwidth delays in the
  HTTP transport; it is not yet a full network emulator.
