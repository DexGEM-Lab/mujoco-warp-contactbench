# OPS

## 2026-08-19T00:00:00Z
- Created feature worktree from dev@4558504.
- Confirmed synthesis checkpoint stepper calls deterministic_actions + env.step only; it does not call record_transition/post_interaction and therefore does not write PPO memory.
- Confirmed ordinary early phase is fixed early30; process_residual_actions zeros processed/cumulative residual during that interval.
- Confirmed each repeated synthesis retry launches a fresh child with attempt_seed = base_seed + attempt_round - 1.
- Implemented initial immutable positive-Z prefix transform and verified against /home/jay/data/manorl_mtp_pre60: original suffix/source indices remain exact and sampled Z is positive.
- Decision: augmented trajectories use prefix length as per-row early phase; ordinary trajectories preserve early30.
- Decision: 0.10 m deviation terminal is disabled only in the generated prefix and resumes at original pre60 frame 0.

## 2026-08-19T12:00:00Z
- Full pre60 catalog audit: 3,429 trajectories; generated prefix L=41..187 (median 110); original suffix/source indices/object reference exact; discrete splice velocity error max 0.
- Raw C2 acceleration extrapolation was falsified: noisy capture acceleration produced max 3.37 m/s and 139 rad/s. Replaced with exact discrete C1 velocity splice. Final audit max linear speed 0.79 m/s and max angular speed 8.30 rad/s, consistent with source frame0 intervals.
- Real MJX-Warp iphone audit showed low-positive-Z straight paths could contact the floor. Raised minimum elevation to 15 degrees and added a 4 cm endpoint-smooth wrist arc; 12/12 sampled prefixes then had no controlled-right-hand solved floor contact above 0.2 N.
- Viewer batch-loop experiment was rejected because background worlds had already progressed. Final visible loop uses one active env; terminal seed49 reset to seed50 with progress=0 and distinct start, then continued seed51+ in one window.
- Selected accepted parent row ee54c011-8237-5cc4-9061-cd2ea8e5163c from the 10,605-row production Lance: source banana_01_052, checkpoint SHA daa1d2e73a6bff496b09296d9400a53a75cdf1a7765f443bf24ad40be62964ad, object XY offset (-0.0137602, -0.0176767) m.
- End-to-end GPU exporter produced one successful compact v2_contact row from this parent on seed49. Manifest bound parent and prefix; rollout completed at 655 steps with success reason 1.
- End-to-end audit found movement metadata remained at source 60 instead of shifted 150; build_v2_row was corrected to serialize trajectory movement_start/end.
