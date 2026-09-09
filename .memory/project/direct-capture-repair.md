# Direct capture repair

Explicit source+patch replay and generated actual object-motion export:
`docs/direct_capture_repair.md`.

Two independent gaps in the Guangxue September9 capture: grasp geometry and
soft-friction creep. A better object-relative wrist/finger trajectory is needed
for acquisition; elliptic/impratio100 reduces later static slip for the tested
samples without changing mass, friction coefficients, gravity or gains. Keep
this solver profile local to the replay recipe, not a silent project default.
Source hand + improved solver still fails the held-bowl example.

Do not infer object pose slots from capture order when inspecting a runtime
trace. Runtime sorts scene objects: bowl precedes mayonnaisebottle/pitcherbase.
Assert body/joint names or use saved qpos addresses, not just matching nq.
Native coordinate mirrors do not guarantee valid native contact buffers.

Verified source v5 rows0/19/31/36/49 cover five action categories, one sample
per category; no all59 claim. Pitcher needs load compensation in the wrist
target and thumb-clear/sideways handle withdrawal; delivered command track is
frozen and open-loop. Moving bowls explicitly retime a verified grip primitive.
Exported Lance is measured generated motion with a separate actuator-command
track, not an interchangeable human-capture input. Live reproduction is source
UUID/version/row plus patch, as documented in the canonical guide.
