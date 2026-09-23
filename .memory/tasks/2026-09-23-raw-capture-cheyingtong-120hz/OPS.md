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
