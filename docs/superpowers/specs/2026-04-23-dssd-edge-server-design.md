# DSSD Persistent Edge Server Design

**Goal**

Add the smallest possible persistent edge-side service for DSSD so one edge runtime can stay resident in memory and accept repeated generation requests over HTTP.

**Scope**

- Add a new persistent `edge_server` entrypoint.
- Support one synchronous `POST /generate` API.
- Reuse the existing `DSSDEdgeService.generate(...)` flow unchanged.
- Process requests serially in a single server process.
- Return a clear busy error when another request is already running.

**Out of Scope**

- No multi-request concurrency.
- No streaming responses.
- No edge-side `/open_session`, `/verify_round`, or `/close_session` APIs.
- No long-lived client-visible session state.
- No protocol changes to verifier HTTP APIs.
- No scheduler or decode-engine semantic changes.

**Current Problem**

Today the real edge runtime only ships as the one-shot CLI entrypoint `vllm.dssd.entrypoints.edge_runner`. Each invocation creates the runtime, runs exactly one `generate(...)`, prints JSON, and exits. That is sufficient for subprocess smoke tests, but it defeats the main reason to keep edge local: avoiding repeated model startup and runtime initialization across requests.

The verifier side already has a persistent HTTP server entrypoint. The missing piece is a matching persistent edge entrypoint that keeps one `DSSDEdgeService` alive and exposes a minimal request API.

**Design Summary**

Introduce `vllm.dssd.entrypoints.edge_server` as the edge-side analogue of `verifier_server`.

On startup it will:

1. Parse runtime arguments.
2. Build the real edge service through `runtime_factory.build_real_edge_service`.
3. Construct an HTTP server.
4. Optionally write a `ready-file` with the bound server URL.
5. Call `serve_forever()`.

At request time it will:

1. Accept `POST /generate`.
2. Decode the JSON request body into the arguments needed by `DSSDEdgeService.generate(...)`.
3. Reject the request with `409` if another request is already executing.
4. Otherwise call `edge_service.generate(...)`.
5. Return a JSON body containing `req_id` and `output_ids`.

This keeps all DSSD coordination logic inside `DSSDEdgeService` and limits the new server layer to transport, lifecycle, and error mapping.

**API**

The persistent edge server needs only one endpoint in the first version:

### `POST /generate`

Request body:

- `req_id`
- `prompt_token_ids`
- `sampling_params`
- optional `lora_request`

Response body:

- `req_id`
- `output_ids`

This API intentionally mirrors the existing `edge_runner` call boundary instead of exposing lower-level session operations. The current `generate(...)` implementation already performs `open_session -> draft/verify loop -> close_session`, so adding a thinner transport wrapper is the lowest-risk option.

**Busy Semantics**

The server must remain single-request-at-a-time.

The simplest behavior is:

- keep one in-process lock guarding request execution
- if a request arrives while the lock is held, return HTTP `409`
- response body:
  - `error`
  - `error_type` set to `"EdgeBusyError"`

Returning `409` is better than blocking the second request indefinitely because it makes the serialization constraint explicit and keeps failure behavior easy to test.

**Code Structure**

Add a new entrypoint file:

- `vllm/dssd/entrypoints/edge_server.py`

Reuse existing components:

- `vllm/dssd/entrypoints/runtime_factory.py`
- `vllm/dssd/service/edge_service.py`
- `vllm/dssd/transport/http_utils.py`

Add small HTTP payload helpers for edge generate requests and responses in `http_utils.py` rather than hand-rolling JSON parsing inside the server. That keeps encoding rules in one place, matches the existing verifier HTTP utilities, and simplifies tests.

The new server should follow the verifier server structure closely:

- `build_parser()`
- `main()`
- `_build_server(...)`
- `_content_length(...)`
- `_normalize_service_factory_result(...)`
- `_hard_exit(...)`

Parser additions should mirror verifier conventions:

- `--service-factory`
- `--host`
- `--port`
- `--ready-file`
- existing runtime args from `add_runtime_args(...)`
- existing edge-specific args already required to build the real service:
  - `--verifier-url`
  - `--eos-token-id`
  - `--gamma`
  - `--model-runner-version`

Unlike `edge_runner`, request-specific fields such as `--req-id`, `--prompt-token-ids`, `--max-tokens`, and `--temperature` must move into the HTTP body.

**Lifecycle**

Startup and shutdown should match `verifier_server` behavior:

- build the service once at process start
- write `ready-file` after the server socket is bound
- shut down the HTTP server cleanly on `KeyboardInterrupt`
- always run service cleanup in `finally`
- hard-exit with `os._exit(...)` after flushing streams, consistent with existing standalone entrypoints

This preserves the current subprocess-based integration testing pattern.

**Error Handling**

The server should map failures into three buckets:

1. Unknown route
   - HTTP `404`
2. Busy server
   - HTTP `409`
   - JSON body with `error` and `error_type="EdgeBusyError"`
3. Request decode or generation failure
   - HTTP `500`
   - JSON body with `error` and `error_type`

The first version does not need a separate `400` validation layer. Matching the verifier server's simple “catch exception and return structured error JSON” behavior is enough.

**Testing**

Add focused tests in the existing entrypoint and transport suites.

1. Entrypoint unit tests
   - parser defaults
   - `main()` writes `ready-file`
   - `main()` closes the server and cleanup on exit

2. Process integration tests
   - start persistent edge server in a subprocess
   - call `/generate` and assert JSON output
   - verify repeated sequential requests succeed against the same process

3. Busy-path tests
   - use a blocking fake service to hold the lock
   - send a second request while the first is active
   - assert HTTP `409` with structured error JSON

4. Failure-path tests
   - service raises during `generate`
   - client sees a surfaced `500` error payload

This is enough to prove that the server is persistent, serial, and reuses the existing edge generation flow without changing DSSD semantics.

**Tradeoffs**

- This is intentionally not a full edge session protocol. It is a wrapper around the existing synchronous `generate(...)` API.
- `409 busy` keeps the implementation simple, but clients must retry themselves.
- `ThreadingHTTPServer` is acceptable even for serial mode because the lock enforces one active request; reusing the verifier server shape keeps code churn low.

**Future Extensions**

If later needed, this design can expand in-place to:

- add edge-side streaming endpoints
- add explicit `/open_session` and `/close_session`
- replace `409 busy` with a queue
- attach request metrics or tracing

None of those are required for the first persistent edge server.
