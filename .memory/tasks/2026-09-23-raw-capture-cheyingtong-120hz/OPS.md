# Evidence trail

## 2026-09-23T16:25:51+08:00 — canonical raw-source inventory
- Read-only inspection of Remake v978 and Guangguan v530.
- Remake: 10,584 rows; all right-hand; all strictly increasing timestamps; elapsed rate approximately 100 Hz; 103 pairs.
- Guangguan: 5,164 rows; elapsed aggregate rate 120.00015 Hz; 35 no-right rows; 9 non-increasing timestamp rows; 55 compound bottle/cap rows; 5,120 eligible with explicit bottle target; 109 pairs.
- Combined eligible: 15,704 rows and 139 unique pairs. Source UUID sets are internally unique and have zero overlap.
- Inspected current compiler and found uniform-grid resampling based on , not raw elapsed timestamps.
- Inspected fixed-manifest decoder and confirmed it rejects source operator/beta mismatch, so an explicit transfer mode is required.
- Inspected reward and contact mechanics: trainer defaults to five fingertips; residual action penalty scale is zero.

## 2026-09-23T16:26:58+08:00 — correction to compiler observation
- The omitted expression in the preceding compiler note is `source_times = np.arange(source_count) / reference_fps`.
- This means the current generic path does not use raw elapsed timestamps when binding a modern capture to the requested control clock.

## 2026-09-23T17:11:53+08:00 — raw-transfer package and composite-scene validation
- Added explicit fixed-hand manifest generation and process-local asset-root selection.
- Generated Cheyingtong/f98 manifest:   `outputs/raw_capture_transfer/cheyingtong_asset_manifest.json`; SHA256   `d9fa818613cfc899041d4d195d14a8d5cfd8d45af8a00234b55e75c0b59aeb42`.
- Compiled Remake row 3282 (cylinder5:03): source 1263 frames/12.621 s -> 1516 frames/12.625 s; terminal hold 4.0 ms.
- Compiled Guangguan row 1057 (cube2:11): source 578 frames/4.809 s -> 579 frames/4.816667 s; terminal hold 7.667 ms.
- Both packages retain source operator/betas and bind physical hand operator `cheyingtong`.
- Compiled all 55 Guangguan bottle+cap action18 rows with explicit `bottle:18` target; no rejections.
- One-world CPU MJX-Warp scene contained free bottle and free cap, object-object collisions enabled. Both initial XYZ states matched package scene states within 5e-9 m and four zero-residual controls remained finite.
- Focused validation with explicit f98 asset root: 57 passed, 7 skipped.

## 2026-09-23T17:37:23+08:00 — multi-source training contract
- Added deterministic composition of independently verified MTP catalogs. A real two-package load assigned one Guangguan and one Remake trajectory without Lance/PyArrow and produced aggregate checkpoint digests under `manorl.trajectory_package_bundle.v1`.
- Raw training profile requires right-only120Hz full captures, fixed package backing, action-specific `raw_gesture` contact intent, all22 finger residual axes available, and positive cumulative-residual regularization.
- Initial conservative profile is1mm/control XYZ,10mm XYZ cap,1x joint scale/cap, penalty scale1.0.
- Checkpoint strict signature now binds contact mode, residual-joint mask, complete reward config, and the package/bundle digest.
- Focused suites:57 passed/7 skipped for asset+trajectory+package+unified raw boundary;98 passed with one known pre-existing test failure excluded and19 environment setup errors caused by the obsolete singular NAS fixture path.

## 2026-09-23T18:01:18+08:00 — zero-residual physical baseline
- All screens used the fixed Cheyingtong/f98 hand, 120/480Hz, zero residual, Warp CCD16 with32 contacts/world. The earlier16/world bottle screen overflowed at901>880 total slots and is invalid; it was replaced by the clean32/world run.
- Remake cylinder5:03 row3282: no hand-object contact; actual bottom clearance max14.43mm vs reference109.17mm; reason2 at frame198 with102.92mm object error.
- Guangguan cube2:11 row1057:76 contact frames, max3 keypoints, but zero loaded-airborne frames; actual bottom clearance max1.88mm vs reference99.67mm; reason2 at frame214 with101.34mm error.
- Guangguan bottle+cap action18:55/55 reason2;52/55 ever contacted; every contacting row reached exactly one keypoint and none reached opposing two-keypoint contact; zero loaded-airborne frames. Median contact duration98 frames, median actual bottom clearance0.765mm vs median reference89.38mm. One3.65cm transient had no simultaneous load-bearing contact and is not a grip.

## 2026-09-23T18:16:56+08:00 — user scope correction and bounded smoke boundary
- User clarified that current delivery is input alignment only; no actual training is required.
- The already-running bounded smoke had completed before this clarification:512 envs,100 updates,2,457,600 transitions, two references, fixed Cheyingtong. It produced0 natural successes across11,776 completed stochastic episodes and is retained only as a diagnostic artifact, not evidence about full-corpus learnability.
- Run root: `/home/jay/dexrobot/FromSSH/manoRL_raw_transfer_runs/two-source-two-reference-n512-u100-20260923T095718Z`.
- Final checkpoint SHA256: `40f59e5dc9795d17c35d4e8a924a9045423dfcd0ba9aead1a9d769f85c08624a`.
- A telemetry defect was observed: persisted update aliases used CLI placeholder cube1/01 for mixed-package episodes while the episode ledger retained correct identities. The fix derives aliases from resolved trajectory labels and persists real grouped metrics.
- No trainer or zero-residual process remains active; all four GPUs were released.
