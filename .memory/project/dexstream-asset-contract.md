# DexStream asset compatibility

Canonical source/update procedure: `docs/manorl_asset_source.md`.

- Discover current object entries only at the bundle root. `calibration/versions/` contains same-name historical URDFs; recursive uniqueness tests can silently exclude every current object.
- Source object-asset-v2 declares root MJCF physical authority. ManoRL's URDF compatibility adapter is checked against it for inertia and collision geometry/poses. Viewer materials/textures come from root MJCF, since several URDFs omit texture bindings.
- Scale collision mesh vertices, then apply each URDF collision origin. The resulting body-frame coordinates must agree with compiled MuJoCo geometry. Marker-aligned egg/cup/rack/bin assets have nonidentity origins; mesh-author coordinates give wrong placement/observation geometry.
- Pin, manifest, setup LFS closure, and checkpoint physical identity move together. Native strict resume must reject old asset identities even when policy tensor dimensions match.

Regression boundary: `tests/manorl/test_current_asset_contract.py` plus the existing asset, scene-source, and submodule tests.
