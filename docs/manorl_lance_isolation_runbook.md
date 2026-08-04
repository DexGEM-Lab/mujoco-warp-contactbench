# ManoRL Lance Isolation Runbook

## Operational decision

Production ManoRL training uses **Source → Compile → Run**.

- Lance is the immutable archival source.
- Short-lived compiler workers are the only processes allowed to import Lance or PyArrow.
- Long-lived MuJoCo/MJX/PPO trainers consume a verified ManoRL Trajectory Package (MTP) through JSON and memory-mapped NPY arrays.
- Supplying `--trajectory-package` is fail-closed. Missing files, `READY`, hash drift, or ABI drift aborts startup; the trainer never falls back to direct Lance.

This boundary is required on server2. Direct Lance remains a bounded source-validation tool on a healthy host, not a production trainer input.

## Incident signature

The server2 failure had a distinct signature:

1. A direct-Lance N8192 trainer exited with SIGSEGV/139 during CPU-side trajectory decoding.
2. GPU0 peaked at only 36 MiB, so MuJoCo model construction, JIT, rollout, PPO, and production VRAM allocation had not started.
3. Kernel faults appeared at varying invalid addresses in `libpython`, the Lance extension, or nearby native code.
4. The same pinned Lance version and bytes were stable on a healthy machine.
5. On server2, repeated fresh-process decoding failed intermittently across Python environments. Arrow thread limiting reduced the observed rate but did not eliminate it.

These observations support a host/VM-specific Lance/PyArrow native decode boundary. Increasing `num_envs` increased the number of source candidates decoded in the long-lived process and therefore increased exposure. Retrying the whole trainer was rejected as a production strategy: a decoder that returns successfully may still have damaged the trainer heap, and a launch retry cannot certify a multi-day process address space.

A prior unexplained server2 VM reset is a separate host-reliability issue. MTP isolation removes Lance from the trainer but does not prove that the hypervisor/VM cannot reset again.

## Solution architecture

### 1. Source

Current Guangguan source:

```text
/mnt/nas-222-projects/mocap_v2/lance_datasets/human_p1_guangguan/human_p1_guangguan_clean.lance
```

Pinned Lance version: `295`.

The source is never rewritten by MTP compilation.

### 2. Compile

Run the coordinator on a healthy Lance host. The coordinator itself imports neither Lance nor PyArrow. It launches isolated discovery and decode workers.

```bash
PYTHONPATH=$PWD /home/jay/anaconda3/envs/manorl_mujoco/bin/python \
  -m tools.compile_manorl_trajectory_package \
  --output /local/filesystem/mtp-v1-v295-all75-right-f120-pre180-post250.pending \
  --dataset-path /mnt/nas-222-projects/mocap_v2/lance_datasets/human_p1_guangguan/human_p1_guangguan_clean.lance \
  --dataset-version 295 \
  --reference-fps 120 \
  --hand-side right \
  --pre-padding 180 \
  --post-padding 250 \
  --shard-size 128 \
  --max-attempts 3
```

Worker rules:

- Native SIGSEGV/SIGABRT receives at most three fresh-process attempts.
- A repeatedly crashing shard is recursively bisected.
- A persistently crashing single candidate fails the build.
- Deterministically invalid source candidates enter the manifest rejection ledger.
- `decoded + rejected` must equal discovery exactly. Unaccounted rows, identity mismatches, and package contract errors fail the build.

Version 295 currently resolves:

- 75 object/action pairs.
- 3,434 discovered candidates.
- 3,429 valid trajectories.
- 5 explicit rejections, all caused by non-increasing source timestamps.
- 2,427,490 compiled frames.

Rejected identities are `cube1_01_1628`, `cube2_01_2809`, `cylinder3_09_362`, `cylinder4_03_731`, and `mayonnaisebottle_01_2589`.

### 3. Package format

`manorl.trajectory_package.v1` contains:

```text
manifest.json
READY
offsets.npy
source_indices.npy
timestamps.npy
q_ref_by_side.npy
object_pos_raw.npy
object_pos.npy
object_quat_xyzw.npy
```

Properties:

- NPY files are loaded with `allow_pickle=False`.
- Arrays are mmapable; side-major `q_ref_by_side` preserves contiguous trajectory slices.
- Every array has a manifest SHA256, dtype, and shape.
- The canonical manifest binds the source catalog, decoded/rejected partition, pair counts, clocks, padding, and compiler contract.
- `READY` contains the package digest and is the publication boundary.
- The package is a complete catalog independent of `num_envs`; N32, N4096, and N8192 share it.

Current package identity:

```text
package_digest = 994ff82737dbf839165e4e22830efa107e227a5d5b613446c15a959586cbfe5c
catalog_digest = 0ab4e64badd921890ead91e729f49c59b7234e0010db97fee7178917db296223
manifest_sha256 = 55d386b41d9637bb6e6f9d8538869637d815896ff8dfdcd77d4d22814afcf547
N8192_assignment_digest = 444de860a5e22415f2de3b66899558a94cee821ad0c0e623c6e28cc8042c4063
```

### 4. Publish

The local builder publishes atomically with a directory rename. CIFS denied that directory rename during the incident. Publish to NAS with `READY` withheld until destination-side hashes pass:

```bash
PYTHONPATH=$PWD /home/jay/anaconda3/envs/manorl_mujoco/bin/python \
  -m tools.publish_manorl_trajectory_package \
  --source /local/filesystem/mtp-v1-v295-all75-right-f120-pre180-post250.pending \
  --destination-parent /mnt/nas-222-projects/mocap_v2/manorl_trajectory_packages
```

Canonical NAS package:

```text
/mnt/nas-222-projects/mocap_v2/manorl_trajectory_packages/mtp-v1-v295-all75-right-f120-pre180-post250-994ff82737dbf839165e4e22830efa107e227a5d5b613446c15a959586cbfe5c
```

Do not train across CIFS. Publish once, then stage the immutable package to the training host's local storage with the same publisher or an equivalent `READY`-last procedure.

Server2 local package:

```text
/home/ubuntu/data/manorl_trajectory_packages/mtp-v1-v295-all75-right-f120-pre180-post250-994ff82737dbf839165e4e22830efa107e227a5d5b613446c15a959586cbfe5c
```

## Validation

### Package-only host validation

This command performs full hash verification, NPY mmap, catalog reconstruction, and deterministic assignment without Lance/PyArrow:

```bash
PYTHONPATH=$PWD /home/ubuntu/miniconda3/envs/manorl_mujoco/bin/python \
  -m tools.validate_manorl_trajectory_package \
  --trajectory-package /home/ubuntu/data/manorl_trajectory_packages/mtp-v1-v295-all75-right-f120-pre180-post250-994ff82737dbf839165e4e22830efa107e227a5d5b613446c15a959586cbfe5c \
  --num-envs 8192 \
  --iterations 30
```

Acceptance requires:

- 30/30 identical package, catalog, and assignment digests.
- 75 pairs and 3,429 trajectories.
- `lance_imported=false` and `pyarrow_imported=false`.

Server2 passed this acceptance.

### Source equivalence

Run only on a healthy Lance host:

```bash
PYTHONPATH=$PWD /home/jay/anaconda3/envs/manorl_mujoco/bin/python \
  -m tools.validate_manorl_trajectory_package_source \
  --trajectory-package /path/to/verified/package \
  --num-envs 8192
```

The accepted package matched all 8,192 assignments and covered all 3,429 valid trajectories. Identity, source indices, timestamps, bilateral references, object poses, z shift, clocks, and movement bounds were array-exact.

## Production launch

For server2, pass the local MTP explicitly:

```bash
export MANORL_PYTHON=/home/ubuntu/miniconda3/envs/manorl_mujoco/bin/python
export MANORL_TRAJECTORY_PACKAGE=/home/ubuntu/data/manorl_trajectory_packages/mtp-v1-v295-all75-right-f120-pre180-post250-994ff82737dbf839165e4e22830efa107e227a5d5b613446c15a959586cbfe5c
export MANORL_DATASET_PATH=/mnt/nas-222-projects/mocap_v2/lance_datasets/human_p1_guangguan/human_p1_guangguan_clean.lance
export MANORL_DATASET_VERSION=295
export MANORL_REFERENCE_FPS=120
export MANORL_UPDATES=8000
export MANORL_CHECKPOINT_INTERVAL=25
export MANORL_WANDB=false
export MANORL_TIMEOUT=30d

./train.sh all 8192 0
```

`MANORL_DATASET_PATH` remains source provenance. `MANORL_TRAJECTORY_PACKAGE` is the runtime data path. Supplying the package prevents Lance fallback.

The current production contract is all 75 pairs, right-hand policy, 120 Hz source/reference/policy, 480 Hz physics with four equal substeps, pre-padding 180, post-padding 250, FiLM, terminal handling, unified object batch, persistent CCD, and 16 contacts/world.

Check a live trainer boundary with:

```bash
PID=$(pgrep -f '/home/ubuntu/miniconda3/envs/manorl_mujoco/bin/python -m tools.train_manorl_cube1' | tail -1)
grep -Eia 'lance|pyarrow' /proc/$PID/maps
ls -l /proc/$PID/fd | grep trajectory_packages
```

The first command must produce no mappings; the second should show local NPY descriptors.

## Checkpoint ABI

Package-backed checkpoints record:

- `trajectory_package_schema`
- `trajectory_package_digest`
- `trajectory_package_manifest_sha256`
- `trajectory_catalog_digest`

Strict resume requires an exact match. A package checkpoint cannot strict-resume into a direct-Lance runtime, and a direct-Lance checkpoint cannot strict-resume into a package runtime. Cross-package transfer is an explicit warm-start that resets optimizer, scheduler, memory, and progress.

## Capacity and remaining risks

Accepted server2 observations:

- N4096 used about 11.78 GiB and sustained roughly 11k transitions/s.
- N8192 used about 20.59 GiB, leaving about 3.5 GiB, and sustained roughly 11.2–11.5k transitions/s over updates 22–26.
- No Lance/PyArrow mappings, kernel segfault, NVIDIA Xid, or OOM appeared in package-backed runs.

N12288 is not a valid current-contract capacity test. Linear extrapolation from N4096/N8192 predicts about 29.4 GiB, exceeding the 24 GiB GPU before transient margin. Use N8192 unless a separate memory-reduction contract is implemented and validated.

MTP isolation addresses the observed Lance/PyArrow decoder failure. It does not explain the earlier VM reset. Long production jobs still require checkpoint cadence and host/hypervisor monitoring.

## Failure response

- **Missing `READY`, manifest mismatch, or array hash mismatch:** do not bypass validation or use direct Lance. Re-stage from the canonical NAS package; if NAS also fails, rebuild on the healthy source host.
- **Compiler worker native exit 139:** inspect worker logs, allow the bounded retry/bisection policy, and preserve the failed shard. Do not move Lance back into the trainer.
- **Candidate-level source rejection:** confirm it is present in the manifest ledger and that decoded plus rejected equals discovery. Never silently drop an unaccounted row.
- **Trainer shows Lance/PyArrow mappings:** stop before production updates and inspect the launcher. Confirm `--trajectory-package` and the direct Conda Python path are present.
- **CUDA OOM:** treat it as a capacity issue, not a data-decoder issue. Reduce env count or redesign memory use; do not weaken package validation.
- **VM reset:** collect hypervisor evidence. Guest logs alone were insufficient in the original incident.

Preserve the source Lance, canonical NAS MTP, server-local MTP, package digests, and setup evidence. Test checkpoints and observation-run outputs may be deleted after their conclusions are recorded.
