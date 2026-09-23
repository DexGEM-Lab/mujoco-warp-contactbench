# Current model

## Phenomenon
The requested artifact is an aligned input boundary, not a trained policy. Raw captures from several people must become one physically unambiguous reference corpus: source motion is retained, but every future simulation uses exactly the same Cheyingtong right MANO hand and 120 Hz clock.

## Supported mechanisms
- Physical hand identity is explicit and independent of each source row. Source operator and ten source MANO betas are retained as provenance and never choose the hand asset.
- Remake is physically100Hz. A real12.621s row became12.625s at120Hz, retaining its final pose with a4ms terminal hold.
- Guangguan is physically120Hz by elapsed duration despite median adjacent intervals near8ms. A real4.809s row became4.816667s, retaining its final pose with one7.667ms edge hold.
- Full-capture selection preserves approach, annotated movement, release, and withdrawal.
- Compound bottle+cap rows are representable without relabeling: bottle is the explicit tracked object and cap remains a second free body with object-object collision.
- Two independently verified MTP catalogs compose deterministically while retaining separate source paths/versions and producing one aggregate checkpoint digest.

## Dataset accounting
- Remake v978:10,584 rows; all right-hand; all timestamps strictly increase;103 object/action pairs.
- Guangguan v530:5,164 rows. Thirty-five rows lack a right hand and nine have non-increasing timestamps. These44 rows are explicit future compiler rejections.
- The55 bottle+cap action18 rows compile with explicit `bottle:18` targeting.
- Expected accepted total if full compilation is requested:15,704 rows across139 unique object/action pairs.

## Direct validation
- Real Remake100Hz package: source1263 frames/12.621s → output1516 frames/12.625s.
- Real Guangguan120Hz package: source578 frames/4.809s → output579 frames/4.816667s.
- All55 bottle+cap rows compiled with no rejection.
- One-world bottle+cap MJX-Warp scene contained free bottle and cap joints, enabled object-object collision, and matched both raw initial positions within5e-9m.
- Fixed profile is Cheyingtong/f98 manifest SHA256 `d9fa818613cfc899041d4d195d14a8d5cfd8d45af8a00234b55e75c0b59aeb42`.

## Ruled out
- Selecting a physical MANO hand from `index.operator`.
- Requiring source betas to equal Cheyingtong betas.
- Reusing IsaacGym `q_state_ref` semantics.
- Treating Remake frames as already120Hz.
- Interpreting Guangguan median dt as a125Hz episode clock.
- Dropping the cap or forcing reference object poses after reset.
- Combining two sources by falsifying a single Lance source identity.

## Historical diagnostic boundary
A bounded100-update/two-reference trainer smoke was completed before the user clarified that training was unnecessary. It produced no natural completion and is not a deliverable claim about the aligned corpus. No further training or policy evaluation will run under the current task.

## Current justified claim
The input boundary is ready: canonical raw rows can be decoded with their source provenance, converted to a common duration-preserving120Hz clock, bound to one Cheyingtong physical hand, represented with complete free-body scenes, stored in verified source-specific MTPs, and loaded together without Lance in the trainer process.

## Remaining optional operation
Full compilation of15,704 accepted rows is an operational publication step, not missing input semantics. Run it only when the user requests the artifacts.
