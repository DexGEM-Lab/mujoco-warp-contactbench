ManoRL shared scene XML now explicitly sets ambient RGB .4 and diffuse RGB .65.
The tiled viewer uses the same constants; the hand-range fixture inherits them.
Compiled-model tests cover homogeneous/unified and visual/nonvisual models and
show physical arrays unchanged when the headlight element is removed. Focused
viewer and hand-range tests also pass. Canonical documentation is the Default
ManoRL rendering light section in docs/manorl_asset_source.md. No camera,
specular, training, contact, checkpoint identity or other-product changes.
