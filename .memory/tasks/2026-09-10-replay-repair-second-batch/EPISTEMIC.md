# Full59 repair: normal timing and grip refinement

## Objective
All59source-v5 rows need task-preserving contact repair. User now points out that
normal bowl motions should not contain the copied10s diagnostic hold, and permits
stronger grasp target closure. Keep source and first5published examples immutable.

## Evidence already established
Long-test registry outputs/full_repair/registry.json has59accepted actual physics
runs. Independent reviewer8797289e corroborated all raw outcomes/criteria, including
spatial pitcher outlet-over-bowl test and complete support/release. No hidden
physical failure was found. First5carryover entries lack uniform validation.json,
but reviewer recomputed their gates; new publication must supply equivalent checks
without changing original bundle. Parent inspected all59end states and critical
container keyframes; full59videos rendered.

## Unfinished quality issue
The long-test task motion reused donor49with its10s stability extension, making
moving-bowl outputs ~25s although original source clips are~7.1s. This is a
category error: verification time leaked into the user trajectory. Physical
success of the long version does not establish success after removing waits.
Normal output must remove long post-acquisition constant-command plateaus while
retaining source approach and all moving/contact commands. A separate appended
validation tail can test stability but must not enter normal exported duration.

## Next discriminating intervention
Generate shorter fixed command streams by plateau compression, with exact kept
command/source correspondence and no uncontrolled object actuation. First compare
representative bowl/bottle/pitcher with no extra grip versus joint-specific modest
closure preload after physical acquisition, tapered away before release. This
isolates timing sensitivity from stronger pinch. Do not expand degree values on
wrist or abduction axes. Propagate only variants with measured persistent grip,
no displacement/tilt regression, normal release and support.

## Runtime constraints
Explicit inherited elliptic/impratio100, same masses/mu/gains/gravity. One GPU
stream, currently free except desktop. Native model order must match trace.
Use primary .venv/PYTHONPATH for materialized assets; empty case submodule is not
an asset hash change and must not be reinitialized just to inspect saved data.

## Review/tool evidence
First full critic4cabcf68 died while heuristic 'id' column selection loaded raw
video/video_depth; corrected retry8797289e used local NPZ only and completed.
No source video column decoding needed. Native subagent resume parser is broken
for recorded acceptance fields; failed resumes did not start work. Evidence and
commands remain in OPS.md.
