# Current model

## Phenomenon
The fixed 35 accepted parents define 350 planned near/far distance-cell slots. Each slot has at most 12 same-cell candidate seeds. The scientific output is the actual paired success yield under the established synthesis setting, not a quota that physical parameters may be changed to satisfy.

## Fixed production identity
- The original 35 accepted-parent v3 descriptors are immutable inputs.
- Each parent binds its source/reference, checkpoint, object offset, raw frame-0 hand pose, and movement-end+15 anchor.
- The original 350 primary slots and their 4,200 planned candidate seeds remain fixed.
- Candidate failure advances only within the same parent/slot distance cell. It never changes parent or crosses cells.

## Fixed physical setting
- Approach uses the established 4 cm endpoint-smooth Z arc from `ApproachPrefixConfig`.
- Prefix has no policy inference, processed action, or accumulated residual.
- The accepted checkpoint policy runs through the unchanged task body.
- Both near and far require v4 retreat from movement-end+15; policy and new residual stop there while entry residual decays smoothly.
- Contact/deviation thresholds, near/far start distributions, and retreat bounds are unchanged.

## Bounded-yield semantics
- A candidate yields two rows only when both near and far succeed.
- A slot stops on its first accepted candidate or after all 12 candidates fail.
- Exhausting a slot records zero yield for that slot and execution continues to later slots.
- `attempts_complete` means all ten parent slots accepted or exhausted their budget.
- `complete` / `all_slots_succeeded` retain ordinary exporter meaning: every requested target succeeded.
- The final Lance merges all successful pairs after all 350 slots have terminal states; row count is `2 × successful_pairs`, at most 700.

## Runner design
- One parent worker owns one single-environment MJX/checkpoint runtime and atomic status.
- One fresh process per parent bounds Warp allocator lifetime; Server1 distributes parent workers across four GPUs.
- Resume skips accepted and exhausted slots and continues only unfinished budgets.
- Planned seeds are explicit; no seed+1 fallback exists.

## Negative evidence
- A diagnostic 10 cm arc increased yield but changed the physical setting without user authorization. That run was stopped at 204 rows, marked diagnostic-only, and must not be merged or published.
- Replacing parents, crossing distance cells, or extending beyond 12 candidates would measure a different process.
- Treating 700 as mandatory conflates target capacity with observed yield.

## Current claim
The correct production experiment is ready after synchronizing runner v3: run all 350 slots under the original 4 cm setting and 12-candidate limit, merge the actual paired yield, validate compact v2_contact, and report success rates by action, parent, distance cell, fallback rank, and failure reason.
