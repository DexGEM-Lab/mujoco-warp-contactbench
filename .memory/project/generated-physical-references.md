# Generated physical-reference ingestion

Canonical contract: `docs/generated_physical_references.md`. Generated120Hz
Cheyingtong Lance keeps actual states, explicit physical operator/betas/manifest,
raw capture provenance and separate nominal/actuator/substep target semantics.
Sunke denotes a predecessor simulation hand, not the raw capture operator.

Default trajectory discovery still excludes generated data. Explicit
`generated_reference=True` / `--generated-reference` requires canonical v2.3
120/480Hz provenance, prestep frame0, zero padding and the matching explicit
physical-hand manifest. It decodes fixed right/left slots, keeps full episodes and
world coordinates, and does not reground or crop away perturbed starts. The flag
is bound into the MTP selection and trainer metadata. Existing raw paths are
unchanged.

Training tracks generated actual motion. Recorded generation observations use
the declared source-target/mask semantics and normal-only CPU-reconstructed
forces. They are not claimed to equal fresh training observations, and one
arrival-frame final actuator target does not reproduce480Hz internal wrist PID.
Use retained recipes/substep NPZ for physical reproduction.
