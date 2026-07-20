# Results

The persistent-thread route experiment is unsupported by the pinned JAX/Warp
runtime and was removed.

- Server: `192.168.9.220`
- Device: physical GPU3, RTX 4090
- W&B run: `z4qzkqn6`
- Failure point: first concurrent 77-pair evaluation step
- Process result: fatal XLA abort before any training update
- CUDA error: `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`
- XLA assertion: `stream->BlockHostUntilDone() is OK`

The independent route calls entered concurrently in the local barrier test, but
Warp FFI execution uses CUDA stream capture that cannot be synchronized from
the competing host workers. Multi-threading the current route implementation is
therefore not a valid production optimization.
