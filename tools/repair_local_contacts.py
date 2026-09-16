#!/usr/bin/env python3
"""Replay archived Cheyingtong commands at120Hz, optionally with a local recipe."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from sim.manorl.local_contact_repair import (
    ASSET_COMMIT, CHEY_MANIFEST_SHA, MODEL_FIELDS, dump, edit_targets, evaluate, load_input, sha,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True, help="completed100Hz Cheyingtong43 output root")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--asset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True, help="new output root; existing row outputs are never overwritten")
    p.add_argument("--hz", type=int, choices=(100, 120), default=120)
    p.add_argument("--rows", nargs="+", help="default: all43, with previous failures first")
    p.add_argument("--recipe", type=Path, help="one source-bound local correction recipe")
    p.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--preflight", action="store_true")
    return p.parse_args()


def bind_model(a, inp):
    if sha(a.manifest) != CHEY_MANIFEST_SHA:
        raise ValueError("Cheyingtong manifest hash mismatch")
    manifest = json.loads(a.manifest.read_text())
    if manifest["hand_operator"] != "cheyingtong" or manifest["source_commit"] != ASSET_COMMIT:
        raise ValueError("unexpected physical hand")
    if "sim.manorl.assets" in sys.modules:
        raise RuntimeError("physical hand must be selected before asset imports")
    os.environ["MANORL_ASSET_MANIFEST"] = str(a.manifest.resolve())
    import sim.manorl.assets as assets
    assets.DEXSTREAM_ROOT = a.asset_root.resolve()
    names = inp.arrays["scene_object_names"].tolist()
    for name in names:
        assets.validate_asset_manifest(name, hand_side="right")
    mj, model = assets.compile_unified_model(object_types=names, object_collisions=True,
                                             physics_timestep=1 / (4 * inp.hz))
    model.opt.cone = mj.mjtCone.mjCONE_ELLIPTIC
    model.opt.impratio = 100.
    for field in MODEL_FIELDS:
        np.testing.assert_array_equal(getattr(model, field), np.asarray(inp.manifest["native"][field]),
                                      err_msg=f"physical model changed: {field}")
    if list(assets.hand_joint_names("right")) != inp.manifest["joint_names"]:
        raise ValueError("archived hand joint order differs")
    return assets, model


def one(a):
    if len(a.rows or []) != 1:
        raise ValueError("--one requires exactly one row")
    inp = load_input(a.baseline, a.rows[0], a.hz)
    recipe = (json.loads(a.recipe.read_text()) if a.recipe else
              dict(schema="manorl.local-contact-repair.v1", row_id=inp.row_id, phases=[], note="time-resampled baseline, no contact edit"))
    assets, model = bind_model(a, inp)
    desired, finger_delta, edit_info = edit_targets(inp, recipe, assets.hand_joint_names("right"))
    out = a.output / inp.row_id
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    dump(out / "recipe.json", recipe)
    provenance = dict(inp.provenance, recipe_sha256=sha(out / "recipe.json"), edit=edit_info,
                      joint_names=list(assets.hand_joint_names("right")),
                      input_kind="archived desired plus original controller/preload, not measured hand replay")
    dump(out / "provenance.json", provenance)
    if a.preflight:
        print(json.dumps(dict(row_id=inp.row_id, frames=inp.frames, provenance=provenance)))
        return
    from sim.manorl.local_contact_dynamics import replay
    trace, runtime = replay(inp, model, desired, finger_delta)
    run = out / "replay"
    run.mkdir()
    np.savez_compressed(run / "trace.npz", **trace)
    validation, diagnostic = evaluate(model, inp, trace)
    np.savez_compressed(run / "contact_validation.npz", **diagnostic["arrays"])
    with (run / "contact_chronology.jsonl").open("x") as f:
        for frame, contacts in enumerate(diagnostic["contacts"]):
            f.write(json.dumps(dict(frame=frame, time_s=frame / inp.hz, contacts=contacts)) + "\n")
    dump(run / "metrics.json", dict(schema="manorl.local-contact-repair-replay.v1", runtime=runtime,
                                    source=inp.manifest["source"], source_metadata=inp.metrics["source_metadata"],
                                    clock=inp.provenance, edit=edit_info, asset_provenance=assets.asset_provenance(),
                                    model_operator="cheyingtong", hand_side="right"))
    dump(run / "validation.json", validation)
    validation.update(trace_sha256=sha(run / "trace.npz"), edit=edit_info, hz=inp.hz,
                      recipe_sha256=sha(out / "recipe.json"), elapsed_s=time.monotonic() - start)
    dump(out / "result.json", validation)
    print(json.dumps(validation, ensure_ascii=False), flush=True)


def queue(a):
    rows = json.loads((a.baseline / "comparison.json").read_text())["rows"]
    if a.rows:
        selected = a.rows
    else:
        selected = [x["id"] for x in rows if not x["cheyingtong"]["physical_pose"]]
        selected += [x["id"] for x in rows if x["cheyingtong"]["physical_pose"]]
    if len(selected) != len(set(selected)) or any(name not in {r["id"] for r in rows} for name in selected):
        raise ValueError("invalid or duplicate row selector")
    if a.recipe and len(selected) != 1:
        raise ValueError("one recipe applies to one row")
    if any((a.output / name).exists() for name in selected):
        raise ValueError("selected output already exists; choose a new attempt directory")
    a.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    code_files = [root / "sim/manorl/local_contact_repair.py", root / "sim/manorl/local_contact_dynamics.py", Path(__file__).resolve()]
    pin = {str(p.relative_to(root)): sha(p) for p in code_files}
    dump(a.output / "code_pin.json", pin)
    dump(a.output / "job.json", dict(pid=os.getpid(), rows=selected, hz=a.hz,
                                    baseline=str(a.baseline.resolve()), manifest=str(a.manifest.resolve()),
                                    asset_root=str(a.asset_root.resolve()), created=datetime.now(timezone.utc).isoformat()))
    completed = []
    failures = []
    for name in selected:
        if {str(p.relative_to(root)): sha(p) for p in code_files} != pin:
            raise RuntimeError("runner code changed during queue; do not combine implementations")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--one", "--baseline", str(a.baseline.resolve()),
               "--manifest", str(a.manifest.resolve()), "--asset-root", str(a.asset_root.resolve()),
               "--output", str(a.output.resolve()), "--hz", str(a.hz), "--rows", name]
        if a.recipe:
            cmd += ["--recipe", str(a.recipe.resolve())]
        if a.preflight:
            cmd += ["--preflight"]
        dump(a.output / "status.json", dict(state="running", current=name, completed=completed, failures=failures))
        print(f"START {name}", flush=True)
        with (a.output / f"{name}.log").open("x") as log:
            process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            dump(a.output / "active_process.json", dict(row_id=name, pid=process.pid, command=cmd))
            code = process.wait()
        if code:
            failures.append(dict(row_id=name, exit_code=code, log=f"{name}.log"))
            dump(a.output / "status.json", dict(state="failed", current=name, completed=completed, failures=failures))
            raise RuntimeError(f"row {name} failed to execute: see {a.output / (name + '.log')}")
        completed.append(name)
        if not a.preflight:
            result = json.loads((a.output / name / "result.json").read_text())
            print(f"DONE {name} pass={result['physical_pose_pass']} failed={result['failed_gates']}", flush=True)
    results = [] if a.preflight else [json.loads((a.output / name / "result.json").read_text()) for name in selected]
    dump(a.output / "summary.json", dict(hz=a.hz, physics_hz=4 * a.hz, hand="cheyingtong", rows=results,
                                        completed=len(completed), functional_pass=sum(r["functional_pass"] for r in results),
                                        physical_pose_pass=sum(r["physical_pose_pass"] for r in results)))
    dump(a.output / "status.json", dict(state="complete", completed=completed, failures=failures))


if __name__ == "__main__":
    args = parse_args()
    one(args) if args.one else queue(args)
