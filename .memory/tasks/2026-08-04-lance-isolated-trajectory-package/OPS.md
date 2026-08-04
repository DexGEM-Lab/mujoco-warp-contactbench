# OPS

## 2026-08-04 20:35 CST — task initialization

- User approved the Source → Compile → Run paradigm and requested Lance isolation followed by a training test.
- Created `feat/lance-isolated-trajectory-package` from `dev@9360be8d650bcdf478d740a6c0a3698ee71e8b95` using `scripts/start_pi_task.sh --no-launch`.
- Direct evidence motivating the change: server2 N8192 scratch run `/home/ubuntu/dexrobot/FromSSH/manoRL_mujoco/outputs/manorl/production_server2_allpairs_f120_u8000_n8192_g0_scratch/run-20260804T120821Z` exited139 during CPU-side decode; kernel recorded a random invalid address in `libpython`; GPU0 peak was only 36 MiB.
- Prediction: if the trainer consumes a precompiled pure-NumPy complete catalog, N8192 startup will cross the previous failure boundary without importing Lance/PyArrow, and GPU memory should approach the previously measured ~20.5 GiB rather than fail before context creation.

Cross-reference: `EPISTEMIC.md`.

## 2026-08-04 21:42 CST — package boundary implementation and real-source anomalies

- Added `sim/manorl/trajectory_package.py`: atomic `manorl.trajectory_package.v1` writer, canonical manifest/package/catalog digests, per-NPY SHA256, `READY`, hash-verifying mmap loader, and deterministic env assignment. The module imports neither Lance nor PyArrow.
- Added isolated discovery/decode worker and coordinator CLIs. The coordinator retries native SIGSEGV/SIGABRT exits at most three times and recursively bisects a repeatedly crashing shard. Non-native worker contract failures are not retried.
- Added trainer `--trajectory-package`, package-aware evaluation assignment, checkpoint package/catalog ABI fields, and `MANORL_TRAJECTORY_PACKAGE` support in `train.sh`. Importing the trainer with the package implementation loaded leaves both `lance` and `pyarrow` absent from `sys.modules`.
- Synthetic package round-trip/hash/assignment/checkpoint tests: `7 passed`; focused trajectory/checkpoint tests: `31 passed, 7 deselected`; asset-focused tests after pinned submodule materialization: `15 passed`.
- Real v295 isolated discovery observation: dataset version 295, schema digest `86691c1375496d65cff684943ea038ce9b97b9b6d27abad41574b09a7cefdcff`, 75 pairs, 3434 candidates.
- First NAS-local compile failed after successful decoding because CIFS denied directory `os.replace`; the causal error was `PermissionError [Errno 13]`, not a Lance/native or row-semantic failure. Decision: compile atomically on local storage, then publish to NAS without `READY`, verify there, and write `READY` last.
- Second local compile exposed a source anomaly at `cube1_01_1628`, row 1627: `ValueError: source timestamps are not strictly increasing`. Existing direct selection silently excludes such invalid candidates. The package contract now records every candidate as decoded or rejected in a hash-bound source ledger and requires the partition to equal discovery exactly. Raw candidate ordering remains in the manifest so pair-assignment cycles retain direct-loader semantics.
- Real 128-row rejection smoke: 127 decoded + 1 rejected = 128, with the rejected row and error message present in both worker ledger and shard manifest.
- Full local compile restarted in tmux `manorl-mtp-compile-v295-final`; expected output is `/media/jay/02A61421A614182D/manorl-mtp-build/mtp-v1-v295-all75-right-f120-pre180-post250.pending`.

Cross-reference: `EPISTEMIC.md`.

## 2026-08-04 21:54 CST — full v295 package and source equivalence

- Full compile completed with 3434 discovered candidates, 3429 decoded trajectories, 5 explicit timestamp-order rejections, 75 valid pairs, 2,427,490 total frames, and no native worker retries.
- Rejected identities: `cube1_01_1628`, `cube2_01_2809`, `cylinder3_09_362`, `cylinder4_03_731`, `mayonnaisebottle_01_2589`; every rejection is `source timestamps are not strictly increasing`.
- Local package: `/media/jay/02A61421A614182D/manorl-mtp-build/mtp-v1-v295-all75-right-f120-pre180-post250.pending`, size 1.3 GiB.
- Package digest: `994ff82737dbf839165e4e22830efa107e227a5d5b613446c15a959586cbfe5c`.
- Catalog digest: `0ab4e64badd921890ead91e729f49c59b7234e0010db97fee7178917db296223`.
- Lance-free local verification: 3/3 full hash + mmap + N8192 assignment passes; assignment digest `444de860a5e22415f2de3b66899558a94cee821ad0c0e623c6e28cc8042c4063`; validator confirmed Lance/PyArrow absent.
- Healthy-host direct-source comparison: all 8192 env assignments matched; those assignments cover all 3429 valid trajectories; identity, source indices, timestamps, bilateral q_ref, object poses, z shift, clock, and movement bounds are array-exact.
- Published canonical NAS package with `READY` withheld until destination-side SHA verification:
  `/mnt/nas-222-projects/mocap_v2/manorl_trajectory_packages/mtp-v1-v295-all75-right-f120-pre180-post250-994ff82737dbf839165e4e22830efa107e227a5d5b613446c15a959586cbfe5c`.
- Focused validation after implementation: `119 passed, 8 deselected`; package/trajectory/checkpoint subset: `33 passed, 7 deselected`; compileall, shell syntax, and diff checks pass. One unrelated evaluator fake-runtime test remains a known baseline failure and reproduces on primary `dev`.

Cross-reference: `EPISTEMIC.md`.
