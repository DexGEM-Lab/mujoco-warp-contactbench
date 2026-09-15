# Evidence

## 2026-09-15
Requested visualization under current right-only cheyingtong configuration. Finger scales/caps have2x multipliers, gamma=.9. DIP reaches asymptotic.1rad offset under unit action although configured hardcap.2rad. LocalGPU0 has6880MiB free; use only local rendering, no Server1 jobs touched. Selected slightly bent reference pose is illustrative, not a measured training frame.

## 2026-09-15T16:44:00+08:00 — visual artifact validation
Three range tests passed: recurrence-envelope agreement,DIP.1vs.2hardcap,jointlimit clipping,sweep endpoints. Preview inspection exposed hidden pivot and thumb occlusion; replaced sphere with projected orange marker and complementary side camera. Full render completed22jointsegments; MP4 verified1280x760,24fps,2112frames,88seconds.44 endpoint images and37 uniqueGIFframes (holds coalesced) present. Inspected fourjoint overview: thumbabduction/twist,indexPIP,pinkyMCPflex. Video fully decoded withoutFFmpeg errors. Static page and video returnHTTP200. User-facing assets copied to outputs/hand_residual_ranges_20260915; no Server1 connection/job changed.
