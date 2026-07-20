# Results

The staged route submission was correct but did not improve production
performance and was removed.

- Server: `192.168.9.220`
- Device: physical GPU3, RTX 4090
- Budget: all 77 resolved pairs, 2,048 environments, 1 PPO update
- Controls: device-resident, transition diagnostics disabled
- Same-host baseline: 3,737.5 transitions/s
- Staged result: 3,740.5 transitions/s
- Difference: +0.08%, within normal run-to-run variation
- W&B staged run: `cg3pouee`

Continuous GPU sampling also remained in the same low-utilization range during
the 77-pair evaluation. The JAX/Warp backend did not execute the 13 static MJX
models concurrently merely because their work was submitted before host
materialization. A meaningful speedup requires a different multi-object model
or training organization, not this internal call-order refactor.
