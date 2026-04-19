# DSSD Binary Reject Response Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace JSON-list serialization for `/verify_round` reject responses with a compact binary payload while keeping the current HTTP endpoint and accepted-path behavior unchanged.

**Architecture:** Keep `/verify_round` as a single endpoint. Accepted responses stay `application/json`. Reject responses switch to `application/octet-stream` with a fixed body layout: 4-byte little-endian metadata length, metadata JSON bytes, then contiguous float32 logits bytes. The HTTP client inspects `Content-Type` and decodes either JSON or the binary reject format.

**Tech Stack:** Python stdlib HTTP server/client, PyTorch tensor serialization via CPU contiguous bytes, pytest.

---

### Task 1: Add failing transport tests for binary reject responses

**Files:**
- Modify: `tests/dssd/transport/test_http_verifier_transport.py`
- Modify: `tests/dssd/transport/test_fake_network.py`
- Test: `tests/dssd/transport/test_http_verifier_transport.py`

- [ ] **Step 1: Write failing tests for reject binary decode and compact byte accounting**

```python
def test_http_verifier_transport_decodes_binary_reject_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.dssd.transport.http_verifier_transport import HTTPVerifierTransport

    logits = torch.tensor([0.5, -0.25, 1.25], dtype=torch.float32)
    metadata = {
        "req_id": "req-1",
        "accepted_len": 0,
        "bonus_token_id": None,
        "dtype": "float32",
        "shape": [3],
    }
    metadata_bytes = json.dumps(metadata).encode("utf-8")
    response_body = (
        len(metadata_bytes).to_bytes(4, "little")
        + metadata_bytes
        + logits.numpy().tobytes()
    )

    class FakeHTTPResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def __init__(self, body: bytes) -> None:
            self._body = body
            self.headers = {"Content-Type": "application/octet-stream"}

        def read(self) -> bytes:
            return self._body

    class FakeOpener:
        def open(self, request, timeout):
            return FakeHTTPResponse(response_body)

    transport = HTTPVerifierTransport(server_url="http://127.0.0.1:18021")
    monkeypatch.setattr(transport, "_opener", FakeOpener())

    response = transport.verify_round(
        VerifyRoundRequest(
            req_id="req-1",
            committed_token_id=17,
            draft_token_ids=[19],
            draft_q_values=[0.6],
        )
    )

    assert response.accepted_len == 0
    assert response.bonus_token_id is None
    assert torch.equal(response.rejected_target_logits, logits)
```

- [ ] **Step 2: Run the new transport tests and verify they fail**

Run:

```bash
./.venv/bin/python -m pytest -q \
  tests/dssd/transport/test_http_verifier_transport.py
```

Expected: failure because the current transport only treats `/verify_round` responses as JSON.

- [ ] **Step 3: Add malformed binary response tests**

```python
def test_http_verifier_transport_rejects_malformed_binary_reject_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.dssd.transport.http_verifier_transport import HTTPVerifierTransport

    class FakeHTTPResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def __init__(self, body: bytes) -> None:
            self._body = body
            self.headers = {"Content-Type": "application/octet-stream"}

        def read(self) -> bytes:
            return self._body

    class FakeOpener:
        def open(self, request, timeout):
            return FakeHTTPResponse(b"\\x08\\x00\\x00")

    transport = HTTPVerifierTransport(server_url="http://127.0.0.1:18021")
    monkeypatch.setattr(transport, "_opener", FakeOpener())

    with pytest.raises(RuntimeError, match="malformed binary verify_round response"):
        transport.verify_round(
            VerifyRoundRequest(
                req_id="req-1",
                committed_token_id=17,
                draft_token_ids=[19],
                draft_q_values=[0.6],
            )
        )
```

- [ ] **Step 4: Run the new malformed-response test and verify it fails**

Run:

```bash
./.venv/bin/python -m pytest -q \
  tests/dssd/transport/test_http_verifier_transport.py -k malformed
```

Expected: failure because the current transport has no binary error handling path.

- [ ] **Step 5: Commit the red tests**

```bash
git -C /root/workspace/vllm/.worktrees/dssd-v1 add \
  tests/dssd/transport/test_http_verifier_transport.py
git -C /root/workspace/vllm/.worktrees/dssd-v1 commit -m "test: cover binary reject transport"
```

### Task 2: Add server-side binary reject serialization

**Files:**
- Modify: `vllm/dssd/transport/http_utils.py`
- Modify: `vllm/dssd/entrypoints/verifier_server.py`
- Test: `tests/dssd/transport/test_http_verifier_transport.py`

- [ ] **Step 1: Add serialization helpers for binary reject payloads**

```python
def verify_round_response_to_http_payload(
    response: VerifyRoundResponse,
) -> tuple[str, bytes]:
    if response.rejected_target_logits is None:
        return "application/json", dump_json(
            {
                "req_id": response.req_id,
                "accepted_len": response.accepted_len,
                "bonus_token_id": response.bonus_token_id,
                "rejected_target_logits": None,
            }
        )

    logits = response.rejected_target_logits.detach().to(
        device="cpu",
        dtype=torch.float32,
    ).contiguous()
    metadata = {
        "req_id": response.req_id,
        "accepted_len": response.accepted_len,
        "bonus_token_id": None,
        "dtype": "float32",
        "shape": list(logits.shape),
    }
    metadata_bytes = dump_json(metadata)
    return (
        "application/octet-stream",
        len(metadata_bytes).to_bytes(4, "little")
        + metadata_bytes
        + logits.numpy().tobytes(),
    )
```

- [ ] **Step 2: Update the verifier server to use content-type-aware verify-round responses**

```python
if self.path == "/verify_round":
    request = verify_round_request_from_payload(payload)
    response = verifier_service.verify_round(request)
    content_type, body = verify_round_response_to_http_payload(response)
    self._send_raw(body, content_type=content_type)
    return
```

- [ ] **Step 3: Run the transport tests and verify only client-side decode still fails**

Run:

```bash
./.venv/bin/python -m pytest -q \
  tests/dssd/transport/test_http_verifier_transport.py
```

Expected: server-side response selection works, but client decode for binary reject still fails.

- [ ] **Step 4: Commit the server-side serialization support**

```bash
git -C /root/workspace/vllm/.worktrees/dssd-v1 add \
  vllm/dssd/transport/http_utils.py \
  vllm/dssd/entrypoints/verifier_server.py
git -C /root/workspace/vllm/.worktrees/dssd-v1 commit -m "feat: emit binary reject responses"
```

### Task 3: Add client-side binary reject decode

**Files:**
- Modify: `vllm/dssd/transport/http_utils.py`
- Modify: `vllm/dssd/transport/http_verifier_transport.py`
- Test: `tests/dssd/transport/test_http_verifier_transport.py`

- [ ] **Step 1: Add binary reject decode helper**

```python
def verify_round_response_from_http_payload(
    *,
    content_type: str,
    payload: bytes,
) -> VerifyRoundResponse:
    if content_type.startswith("application/json"):
        return verify_round_response_from_payload(load_json(payload))

    if not content_type.startswith("application/octet-stream"):
        raise RuntimeError(f"unsupported verify_round content type: {content_type}")

    if len(payload) < 4:
        raise RuntimeError("malformed binary verify_round response: missing header")

    metadata_len = int.from_bytes(payload[:4], "little")
    if len(payload) < 4 + metadata_len:
        raise RuntimeError("malformed binary verify_round response: truncated metadata")

    metadata = load_json(payload[4:4 + metadata_len])
    logits_bytes = payload[4 + metadata_len:]
    shape = metadata["shape"]
    expected_bytes = math.prod(shape) * 4
    if len(logits_bytes) != expected_bytes:
        raise RuntimeError("malformed binary verify_round response: logits size mismatch")

    logits = torch.frombuffer(memoryview(logits_bytes), dtype=torch.float32).clone()
    return VerifyRoundResponse(
        req_id=metadata["req_id"],
        accepted_len=metadata["accepted_len"],
        bonus_token_id=metadata["bonus_token_id"],
        rejected_target_logits=logits.reshape(shape),
    )
```

- [ ] **Step 2: Update HTTP transport to branch on response content type**

```python
with self._opener.open(http_request, timeout=self.timeout_s) as response:
    response_body = response.read()
    self._simulate_network(
        network=self.response_network,
        payload=response_body,
        payload_bytes=len(response_body),
    )
    if path == "/verify_round":
        return verify_round_response_from_http_payload(
            content_type=response.headers.get("Content-Type", "application/json"),
            payload=response_body,
        )
    return load_json(response_body)
```

- [ ] **Step 3: Run transport tests and verify they pass**

Run:

```bash
./.venv/bin/python -m pytest -q \
  tests/dssd/transport/test_http_verifier_transport.py \
  tests/dssd/transport/test_fake_network.py
```

Expected: PASS.

- [ ] **Step 4: Commit the client-side binary decode**

```bash
git -C /root/workspace/vllm/.worktrees/dssd-v1 add \
  vllm/dssd/transport/http_utils.py \
  vllm/dssd/transport/http_verifier_transport.py \
  tests/dssd/transport/test_http_verifier_transport.py
git -C /root/workspace/vllm/.worktrees/dssd-v1 commit -m "feat: decode binary reject responses"
```

### Task 4: Verify end-to-end compatibility

**Files:**
- Modify: none unless regressions appear
- Test: `tests/dssd/entrypoints/test_runtime_factory.py`
- Test: `tests/benchmarks/test_dssd_system_benchmark.py`

- [ ] **Step 1: Run broader regression coverage**

Run:

```bash
./.venv/bin/python -m pytest -q \
  tests/dssd/transport/test_http_verifier_transport.py \
  tests/dssd/entrypoints/test_runtime_factory.py \
  tests/benchmarks/test_dssd_system_benchmark.py
```

Expected: PASS.

- [ ] **Step 2: Run one DSSD smoke benchmark to confirm accepted-path compatibility**

Run:

```bash
timeout 300 ./.venv/bin/python benchmarks/dssd/benchmark_edge_verifier_decode.py \
  --model ~/autodl-tmp/Qwen3-0.6B \
  --edge-cuda-visible-devices 0 \
  --verifier-cuda-visible-devices 1 \
  --gpu-memory-utilization 0.8 \
  --prompt-len 18 \
  --decode-tokens 8 \
  --warmup-tokens 0 \
  --repeats 1 \
  --gamma 1 \
  --no-enforce-eager \
  --json
```

Expected: benchmark completes and JSON output still prints normally.

- [ ] **Step 3: Add one focused reject-path microbenchmark or measurement script if needed**

```bash
./.venv/bin/python -m pytest -q \
  tests/dssd/transport/test_http_verifier_transport.py -k binary
```

Expected: PASS, with binary reject path covered by direct transport tests even if the smoke benchmark does not force rejection.

- [ ] **Step 4: Commit the verification sweep if code changed during integration**

```bash
git -C /root/workspace/vllm/.worktrees/dssd-v1 status --short
```

Expected: no unexpected modified files before preparing the final integration commit.
