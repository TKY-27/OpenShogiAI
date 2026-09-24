# Model Weight License

Historically all generated weights were `pending-review` and unpublished. As
of 2026-09-24 the reviewed publication decision is recorded in
[distribution.md](distribution.md): the representative weights listed in
[configs/models/distribution.json](../../configs/models/distribution.json)
are published as CC BY 4.0 with per-model training-source attribution.

The principle below still holds for anything not in that manifest:

Training and validation generate local checkpoints and weight artifacts under
ignored storage. They are project-generated, but they are not published or
assigned a model license by that fact alone. The AGPL-3.0-only source license
does not apply automatically to model weights. Any new publication needs its
own provenance/rights review and a manifest update before release.
