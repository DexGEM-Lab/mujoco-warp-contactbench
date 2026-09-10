# Container contact-template transfer for full source dataset

User expanded task to repair all59 source rows, focusing contact correction and stable holding, preserving non-contact motion where possible. This isolated worker slice owns bottle and pitcher receiver-template preparation, NOT GPU replay or the full-dataset ledger.

Source Lance v5: dexgem_vla_demo_guangxue_cma2lance_20260909_060700.lance.
Donor NAS: dexgem_vla_demo_guangxue_astra_repair_5samples_20260910, bottle row19 and pitcher row31 verified patches and actual recordings. New bottle rows20..30 (11) and pitcher32/33/34/35/37/38/39 (7). Keep donors unchanged.

Ownership: case/direct-replay-trajectory-repair/container-batch, child of feat/direct-replay-trajectory-repair. Only tools/ generator, patches/full_containers/, compact tests/docs and task OPS may be edited/committed. Parent case second-batch manages all GPU run stream and final integration. No edits in primary dev/first feature/NAS, no subagents. No RL or training code.

Acceptance: CPU-only generated UUID-bound patches/command tracks for18 receiver rows, with clearly recorded donor provenance, pose/time alignment, exact preserved non-contact prefix, bounded grasp/hold/placement edits. Existing replay_repaired_capture.py must accept the patch shapes and hashed assets. Do not claim success before parent runs actual physics. Return compact runner list and expected failure modes only where evidence supports them.

Use original source movement timing where possible; bottle template requires local stable grasp plus late pour wrist alignment and source withdrawal. Pitcher donor uses frozen1430target commands with successful source approach/load compensation/sideways release: transfer by receiver source hand/object frames, map motion phases and initial/goal differences, and preserve monotonic reference mapping. No global solver changes beyond explicit inherited elliptic/impratio100.
