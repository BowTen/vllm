# DSSD Experiment Harness

This page documents the config-driven experiment harness for local DSSD
evaluation.

The harness launches DSSD or target-only baseline services as managed
subprocesses, waits for the needed endpoints to become healthy, sends the
configured chat requests, and writes machine-readable result artifacts for
offline analysis.

## Entry Point

Run the harness with:

```bash
vllm bench dssd --experiment-config /path/to/experiment.json
```

The config file is JSON. The harness currently focuses on local edge/verifier
experiments, not remote orchestration.

## What The Harness Controls

One experiment config can define:

- draft model and target model selections
- DSSD versus target-only baseline experiment mode
- gamma sweep
- transport network simulation
  - `latency_ms`
  - `bandwidth_mbps`
  - `jitter_ms`
- edge, verifier, and target-only baseline ports
- `CUDA_VISIBLE_DEVICES` assignments
- extra server args such as `--max-model-len` and `--max-num-seqs`
- one or more chat requests

For DSSD cases the harness automatically adds `--no-async-scheduling`, because
current DSSD replay does not support async scheduling yet.

## Output Artifacts

The harness writes artifacts under `output_dir`:

- `summary.csv`: one summary row per resolved run case
- `summary.json`: JSON version of the summary rows
- `runs.json`: detailed per-run records, including commands and request results
- `<case-id>/dssd-traces.jsonl`: per-request DSSD experiment traces when the run
  is in DSSD mode and the edge emits detailed experiment results

Each summary row includes:

- `case_id`
- `status`
- `experiment_mode`
- `draft_model`
- `target_model`
- `gamma`
- `network_profile`
- average request latency
- `speedup_vs_target_baseline`
- trace record count and trace path

## Qwen Example

This example compares Qwen `0.6B` as drafter against Qwen `8B` as target model
on one machine, sweeping gamma and latency.

```json
{
  "output_dir": "/tmp/dssd-qwen-results",
  "draft_models": [
    "/data/zz/hf/Qwen/Qwen3-0.6B-msfull"
  ],
  "target_models": [
    "/data/zz/hf/Qwen/Qwen3-8B-msfull"
  ],
  "experiment_modes": ["dssd", "target-baseline"],
  "gammas": [4, 6],
  "network_profiles": [
    {"name": "lan", "latency_ms": 0.0},
    {"name": "wan-20ms", "latency_ms": 20.0, "bandwidth_mbps": 100.0}
  ],
  "edge": {
    "host": "127.0.0.1",
    "port": 8000,
    "cuda_visible_devices": "1",
    "extra_args": ["--max-num-seqs", "64"]
  },
  "verifier": {
    "host": "127.0.0.1",
    "port": 9001,
    "cuda_visible_devices": "0",
    "extra_args": ["--max-model-len", "32768", "--max-num-seqs", "64"]
  },
  "requests": [
    {
      "request_id": "counting",
      "messages": [
        {"role": "user", "content": "从1数到5。"}
      ],
      "max_tokens": 64,
      "temperature": 0.7
    }
  ]
}
```

Run it with:

```bash
vllm bench dssd --experiment-config /tmp/dssd-qwen.json
```

## OPT Example

This example mirrors a paper-style OPT setup with a latency and bandwidth sweep.

```json
{
  "output_dir": "/tmp/dssd-opt-results",
  "draft_models": [
    "/data/zz/hf/facebook-opt-125m"
  ],
  "target_models": [
    "/data/zz/hf/facebook-opt-6.7b"
  ],
  "experiment_modes": ["dssd", "target-baseline"],
  "gammas": [4, 8],
  "network_profiles": [
    {"name": "rtt-0ms", "latency_ms": 0.0, "bandwidth_mbps": 1000.0},
    {"name": "rtt-20ms", "latency_ms": 20.0, "bandwidth_mbps": 100.0},
    {"name": "rtt-50ms", "latency_ms": 50.0, "bandwidth_mbps": 50.0}
  ],
  "edge": {
    "host": "127.0.0.1",
    "port": 8000,
    "cuda_visible_devices": "1",
    "extra_args": ["--max-num-seqs", "64"]
  },
  "verifier": {
    "host": "127.0.0.1",
    "port": 9001,
    "cuda_visible_devices": "0",
    "extra_args": ["--max-model-len", "32768", "--max-num-seqs", "64"]
  },
  "requests": [
    {
      "request_id": "edge-cloud",
      "messages": [
        {
          "role": "user",
          "content": "Explain distributed speculative decoding in one paragraph."
        }
      ],
      "max_tokens": 96,
      "temperature": 0.7,
      "top_k": 20
    }
  ]
}
```

## Practical Notes

- Keep draft and target tokenizers aligned for DSSD runs.
- `target-baseline` launches a single target-model server and does not enable
  DSSD mode on that server.
- If you want to source requests from a file instead of inline JSON, use
  `requests_path` and provide either a JSON list or a JSONL file of request
  objects with `messages` and `max_tokens`.
