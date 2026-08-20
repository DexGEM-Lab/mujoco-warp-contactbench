## Objective
Implement synthesis-only accepted-parent augmentation with two approach modes and a deterministic retreat phase. Each reset samples a fresh immutable reference, runs one active environment, preserves compact v2_contact replay, and remains safe for policy-transfer inference from the bound checkpoint.

## Final task contract
- Base reference: pre60/post250 at 120 Hz; ordinary training early30 and strict resume remain unchanged.
- Parent: one prior successful compact v2_contact row, bound by accepted-parent v3 to source identity/version/row, checkpoint SHA/update, object XY offset, raw source frame0 right q_ref[3:28], movement-end, contact qualification, and retreat anchor.
- Parent qualification: last solved right-hand/object contact (>0.2 N per contact pair) must not precede parent movement-end; movement-end+15 to final wrist XY must define a nonzero retreat direction.
- Retreat anchor: parent movement-end + 15 state frames (125 ms at 120 Hz), mapped through state-aligned reference.source_frame_index and verified against command mapping.
- Approach modes:
  - far: initial-object-relative XY 0.30–0.70 m, Z 0.08–0.30 m, source direction ±30°; seed=episode seed.
  - near: independently seeded retreat-like endpoint mapped from final-object-relative to initial object; seed=hash(episode seed, near-approach).
- Retreat endpoint family: original anchor→final XY distance + 0.03–0.15 m, direction ±30°, final wrist Z + 0.04–0.10 m; retreat seed=episode seed.
- Approach start q[0:3] is sampled XYZ; q[3:28] is exact raw source row frame0. Wrist rotation and all finger joints use discrete-C1 quintics to pre60 frame0. Prefix duration accounts for translation, wrist rotation, and max finger displacement.
- Prefix: no policy call; processed/cumulative residual zero; deviation terminal disabled; any solved right-hand/table or right-hand/object contact >0.2 N rejects the candidate.
- Task body: deterministic checkpoint policy and normal deviation semantics.
- Retreat: preserve and smoothly deform original tail XYZ from movement-end+15. No policy call or processed action; no new residual accumulation. Entry cumulative residual is quintic-decayed to exact zero by the final issued command, preventing controller-target discontinuity. Deviation remains enabled.
- Compact production contract: synthetic_mano_target_replay_visual_v2_contact with contact/reference/command mapping plus sibling manifest.

## Constraints
- Work only in feat/synthesis-approach-prefix.
- Preserve primary dev worktree and user-modified test.sh.
- Use explicit policy-transfer, never claim strict ABI equivalence for padding/assets.
- No subagents.
