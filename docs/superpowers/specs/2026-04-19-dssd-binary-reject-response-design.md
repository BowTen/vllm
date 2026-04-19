# DSSD Binary Reject Response Design

**Goal**

Reduce non-network communication overhead in DSSD by removing JSON list serialization for `rejected_target_logits` while keeping the current HTTP-based system architecture.

**Scope**

- Change only the `/verify_round` reject response path.
- Keep `/open_session`, `/close_session`, and verify-round accepted responses unchanged.
- Keep the existing single `/verify_round` endpoint.
- Keep calibrated HTTP network measurement working from real response body bytes.

**Current Problem**

Today the verifier reject path converts `rejected_target_logits` to a Python list and then to JSON text. The edge side parses that JSON back into Python objects and rebuilds a tensor. This adds large CPU overhead and inflates payload size far beyond the underlying float tensor bytes.

**Design**

`/verify_round` will support two response encodings:

1. Accepted response: unchanged
   - `Content-Type: application/json`
   - Existing JSON response body

2. Reject response: new binary format
   - `Content-Type: application/octet-stream`
   - Response body layout:

```text
[4-byte little-endian metadata_len][metadata_json_bytes][logits_raw_bytes]
```

`metadata_json` contains only small fields:

- `req_id`
- `accepted_len`
- `bonus_token_id` set to `null`
- `dtype` set to `"float32"`
- `shape` set to the logits tensor shape, initially `[vocab_size]`

`logits_raw_bytes` is the contiguous float32 tensor payload in row-major order.

**Server Behavior**

- In verifier server handling for `/verify_round`:
  - If the response is all-accepted, continue returning JSON.
  - If the response contains `rejected_target_logits`, serialize metadata JSON and append raw tensor bytes.
- Reject responses must ensure tensor data is contiguous and on CPU before writing bytes.

**Client Behavior**

- In HTTP verifier transport:
  - Inspect `Content-Type`.
  - For `application/json`, keep the current decode logic.
  - For `application/octet-stream`, parse:
    - first 4 bytes as little-endian unsigned metadata length
    - next `metadata_len` bytes as metadata JSON
    - remaining bytes as raw logits payload
  - Rebuild `VerifyRoundResponse` with `torch.frombuffer` or an equivalent low-copy decode path, then reshape from `shape`.

**Compatibility**

- The endpoint path stays the same.
- Accepted-path callers remain unchanged.
- Existing benchmarks and calibrated network logic continue to work because they already measure real HTTP body sizes.

**Error Handling**

- Reject binary responses must validate:
  - body length is at least 4 bytes
  - metadata length is within body bounds
  - metadata contains supported `dtype` and `shape`
  - remaining bytes exactly match `prod(shape) * sizeof(dtype)`
- Malformed binary responses should raise a transport-level `RuntimeError` with enough detail to identify decode failures.

**Testing**

- Keep existing accepted-path HTTP transport tests green.
- Add reject-path roundtrip tests covering:
  - verifier returns binary reject response
  - edge transport reconstructs `VerifyRoundResponse`
  - payload bytes reflect compact binary size rather than JSON-expanded list size
- Add malformed binary response tests for bounds and shape/byte mismatch.
- Add a focused microbenchmark or benchmark smoke comparison showing lower reject-path overhead than the JSON-list path.

**Tradeoffs**

- This is not a general protocol rewrite; it only removes the dominant hot-path overhead.
- JSON remains for control metadata to preserve debuggability and limit code churn.
- The format is intentionally simple and custom rather than multipart or gRPC to keep the implementation small and benchmark-friendly.

**Out of Scope**

- Replacing all HTTP payloads with a binary protocol
- Changing request encoding
- Adding a new endpoint
- Cross-language protocol standardization
