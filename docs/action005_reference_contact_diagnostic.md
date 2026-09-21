# A030 reference-conditioned contact diagnostic

No accepted full action005 resulted from this experiment.

`tools/fit_action005_reference_contacts.py` fits donor contact patches around
A030's original wrist/object reference, with wrist corrections bounded by15mm
and8deg. Four-finger target corrections precede thumb corrections; the original
target prefix before225 and tail from1050 remain identical. These target timing
properties do not certify physical acquisition or release.

The native-U1 frame0 diagnostic fails atframe410: actual bottle tilt55.12deg
(reference44.99deg), lower finger contacts disappear and axial span collapses.
Atframe418, the first reference angle above50deg, ring/pinky remain absent.
Relative rotation reaches72.79deg in the reference50–90deg interval.

Fixed donor hand-surface patches overconstrain the fit: pinky residuals reach
13–19mm. A separate sliding-surface audit under20mm/15deg bounds fits shallow
five-finger geometry at50/90deg. At70deg, seeding from50deg improves the pinky
gap to0.460mm but leaves9.896mm axial error against a5mm allowed band.
A continuous topology is unresolved; these local fits do not prove infeasibility.

Evidence lives in ignored `outputs/action005_reference_contacts_v1/`: `fit.json`,
`trace.npz`, `telemetry.jsonl.gz`, `result.json`, `surface_feasibility.json`,
`surface_feasibility_70_neighbor.json`, and `REPORT.md`. The report contains
exact commands. Contact forces are120Hz last-substep samples; capacities are
checked at480Hz. Maxima were25contacts,55broadphase pairs and128constraints.
The broadphase guard is conservative, not an exact CCD occupancy metric.

The native replay remained finite with zero applied external forces, no
post-frame0 physical state writes, and maximum control discrepancy1.19e-7.
Full-action replays and video were withheld after the failed diagnostic.

Focused validation:

```sh
python -m pytest -q tests/test_action005_reference_contacts.py tests/test_action005_critical_grip.py
```

The next missing result is a continuous sliding-contact branch that closes
the70deg pinky gap while retaining lower axial leverage and opposed normals.
Only its positive native-U1 inversion test would justify full-action replays.
