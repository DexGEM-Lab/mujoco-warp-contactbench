# Operations

Append-only evidence trail for direct Lance target replay.

- 2026-08-03: Task initialized from `dev` at `e1b6066` in linked worktree `feat/direct-lance-target-replay`.
- 2026-08-03: User added acceptance requirement: local one-display GUI smoke must explicitly export the observed `DISPLAY` value before launch; Server2 direct Lance execution remains prohibited.
- 2026-08-03: Local host `jay-4060` has physical X11 display `:1`, one connected primary monitor `DP-2` at 5120×2880, and Xauthority `/run/user/1000/gdm/Xauthority`. GUI acceptance will use `export DISPLAY=:1` and that authority.
- 2026-08-03: Healthy Server1 projected merged row 2239 into a one-row Lance fixture containing only the six direct replay columns. Local copy checksum matched (`1274bade188dbacce1b9552cc0af89c1a8773129712076fb978deb276d03b709` for the transfer tar); Lance version 1/count 1/schema readback passed.
- 2026-08-03: Local direct-Lance GPU headless replay passed all 519 transitions. Report preserved generated row UUID `635d...`, original source row 1613/version 295/UUID `7dda...`, max object-position error `3.8365e-3 m`, and replay max lift `0.211850177 m`.
- 2026-08-03: Local GUI command explicitly exported `DISPLAY=:1` and `XAUTHORITY=/run/user/1000/gdm/Xauthority`. `xwininfo` confirmed `MuJoCo : manorl_cube1_reference` on the sole connected display DP-2; 120-frame looping playback advanced repeatedly, then the tmux session was stopped and the process exited. `GUI_SMOKE_PASS`.
- 2026-08-03: Final direct-row contract rejects partial/malformed CCD provenance instead of silently defaulting physics, requires generated v2.2 rows at 200 Hz starting from timestamp zero, and preserves initial-state metrics. Nine focused direct-row tests pass; combined target-replay/viewer regression reports 34 passes; compile, CLI help, Black, and diff checks pass.
