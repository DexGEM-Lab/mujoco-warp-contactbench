# Current model

Only nominal-U1 A030005. Zero full passes/repeats. Historical reconstruction is no longer required. See PROMPT and latest OPS.

Fresh v3-control baseline reproduces middle loss at418: thumb9.208N/1.616mm penetration; middle0; index/ring/pinky3.674/1.588/2.072N; span80.865mm. Drift4.038deg, relative acceleration80.963deg/s². Lower leverage survives, load balance fails. Thumb>1N at278 precedes middle/ring/pinky acquisition334/310/300; target scheduling is not physical order.

155 buffers restored exactly between same-process8-frame branches at380/400/409. Paired zero max pose1.79e-7/velocity3.20e-5 pass. Two normal-gap directions, ±.004/±.002rad: correct signs, full rank, conditions2.406/1.684/2.279; each checkpoint fails30% nonlinearity.400 middle+.004 loses contact408, unlike+.002 (0.400N); task-force error53.9%.409 contact activation toggles, force77.5%/gap51.4%.380 angular symmetry75.7%, source not isolated; zero-repeat noise much smaller.

Commitment: reject inverse before generating any candidate. Single blocker is nonlinear local target-response map, not historical buffer absence. No global uncontrollability claim. No candidate predicted-vs-transfer result; reference wrench not computed after failed gate.

Archived spans78.989/79.259/79.734mm versus fresh78.660/79.236/79.711mm: supervisor clarified baseline same-basin/no-collapse; original80mm metric remains false. This baseline is not an accepted final grasp. Future authorized correction must affect acquisition before thumb high loading, not only350.

Evidence: outputs/action005_reference_contacts_v5/run_001 contains manifest/model, baseline raw trace/contacts, checkpoints,24 probes+6 zeros, sensitivities/result. interrupted_launch preserves approved operational interruption, not a completed prior experiment. Diagnostic doc gives metrics/command.27 focused tests pass. Unrelated largepose untouched.
