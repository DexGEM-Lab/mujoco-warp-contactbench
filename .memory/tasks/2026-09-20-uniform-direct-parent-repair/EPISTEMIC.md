# Current model

## Objective
Only nominal-U1 A030005, preserving pickup→over-bowl→deep inversion→return→place→release. Zero complete passes/repeats. Other actions and largepose untouched.

## Supported mechanism
Sliding topology preserves ring/pinky leverage (~80.9mm at418), unlike fixed donor patches. V3 still loses middle at410: thumb9.212N/1.617mm penetration, middle0N, relative acceleration81.23deg/s² at418. Geometric target displacement does not realize predicted contact geometry under coupled spring/contact loading. Static reserve is not realized wrench balance.

## Current boundary
V4 requested native sensitivities from complete v3 checkpoints. Historical v3 saved qpos/qvel/ctrl only. Approved fail-closed frozen-prefix reconstruction fails despite bitwise controls/matching initial states/U1 setting. qpos/qvel first exceed1e-6/1e-4 at29; maxima0.000872344/0.101655. Six acquisition frames change0.2N bearing masks. Frame380 force discrepancy0.070006N fails0.02N+1%;400/409 forces pass but cannot establish state identity. Numerical divergence source is not isolated. Historical compiled-model hash unavailable; v4 saves its model/hash.

## Decision and remaining question
Zero sensitivity probes/inverse/candidate replay. V4 complete native checkpoints380/400/409 represent a different reconstruction, not qualified historical v3 states. No derivative sign/rank/conditioning claim is justified. Single blocker: historical native states unavailable and reconstruction fails identity. Forward motion requires historical buffers or explicitly revised state-provenance contract, not silently relaxed tolerances/retries.

## Evidence
Latest OPS and diagnostic doc; outputs/action005_reference_contacts_v2/v3 preserved; v4 contains model/buffers/comparisons/raw trace/contacts/divergence. Planned central probes were not run.20 focused tests pass.
