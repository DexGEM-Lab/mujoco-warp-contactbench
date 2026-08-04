# EPISTEMIC

## Phenomenon

Long-lived ManoRL trainers directly decode nested Lance/PyArrow rows during startup. On server2 this path non-deterministically corrupts native process state: N4096 can succeed, while N8192 exited139 before GPU initialization with a random-address `libpython` segfault. The same pinned data is reliable on a healthy host, and independent fresh-process retries can succeed.

## Supported mechanism

The failure trigger is the nested Lance/PyArrow decode boundary, not MuJoCo, CUDA, or N8192 VRAM. More candidate decoding increases exposure. A successful direct decode does not prove the long-lived trainer address space is clean. Process isolation changes the causal graph: native corruption can kill only a short-lived compiler worker; the trainer receives a validated, non-executable array artifact.

The real v295 discovery catalog contains 3434 syntactic candidates across 75 pairs, but at least one candidate (`cube1_01_1628`, row 1627) is semantically invalid because its source timestamps are not strictly increasing. The existing direct loader defines the training catalog as the ordered valid subset by skipping these rows. Exact MTP equivalence therefore requires a visible, hash-bound decoded/rejected partition rather than the earlier assumption that every discovered row must decode. Raw candidate order must remain available because pair-assignment cycles rotate candidates before filtering invalid rows.

Evidence: `OPS.md` task initialization and package-boundary entry; `.memory/local/server2-reinstall-setup.md` in the primary worktree.

## Intervention

Compile the complete pinned valid trajectory catalog into `manorl.trajectory_package.v1` using isolated Lance workers. The manifest contains the complete source-candidate ledger, explicit rejection reasons, canonical array metadata, per-file hashes, and package/catalog identities. The long-lived trainer verifies and mmaps JSON+NPY only, assigns N32/N4096/N8192 from the same catalog, and binds package identity into strict checkpoint compatibility.

Compilation is atomic on a local filesystem. CIFS does not permit the required directory rename, so NAS publication uses a fail-closed two-phase protocol: upload all content except `READY` to a temporary destination, verify every hash on NAS, then write `READY` last. Consumers reject incomplete uploads.

## Ruled out

- GPU capacity as the N8192 startup cause: failure occurred at 36 MiB before GPU context.
- Deterministic corruption of every v295 row: 3433+ rows decode normally; the observed timestamp defect is a deterministic candidate-level rejection, distinct from random native crashes.
- Generic guest RAM failure: post-reboot 8 GiB multi-pattern memory test had zero errors.
- Arrow thread count as a complete fix: it reduced but did not eliminate failures.
- NAS decode failure during the first package compile: decoding succeeded; CIFS directory rename produced the observed failure.

## Current justified claim

The completed v295 package represents the direct-Lance training catalog exactly: all 3429 valid trajectories and all 8192 deterministic assignments match source identities and arrays, while five invalid candidates are explicit and hash-bound. Server2 reproduced the same package/catalog/assignment identities in 30/30 package-only loads without Lance/PyArrow, then completed N32 and N8192 one-update PPO runs. Live process mappings during N8192 contained the package NPY files and no Lance/PyArrow libraries. The Source → Compile → Run boundary therefore removes the observed direct-Lance N8192 startup failure while preserving training semantics and checkpoint ABI.

## Unresolved anomalies

The deepest random native cause—Lance, PyArrow allocator behavior, or host/VM interaction—is not isolated. The package boundary removes it from the trainer's correctness path. Server2's earlier VM crash also remains unexplained by guest logs, so a successful one-update test establishes the training data boundary, not unattended multi-day host reliability.

## Highest-value next question

Can server2 sustain the same package-only N8192 path over a multi-update soak and then an 8000-update production run without a host/VM reset? The remaining uncertainty is host endurance, not trajectory decoding, assignment, GPU capacity, or PPO closure.
