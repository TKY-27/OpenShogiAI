# Phase 10R storage and acquisition plan

The storage root is `OPENSHOGI_DATA_ROOT`, falling back to the ignored `local/phase10r-data`. Raw archives, extraction trees, normalized JSONL, checkpoints, and holdout objects are outside Git. The committed registry and manifest contain only relative paths, checksums, sizes, and audit metadata.

## Hard guardrails

* Check free space before every object and while streaming. The hard floor is 150 GiB, configured exactly as `161061273600` bytes.
* Download one object at a time. The configured maximum is one parallel download and one request per second.
* Require a published size estimate or a bounded response limit before starting. The configured single-object ceiling is 20 GiB.
* Resume only from a `.part` object with its ETag or Last-Modified checkpoint. A changed validator is preserved as a stale partial; it is not silently appended.
* Verify final size and SHA-256 before renaming the object into the verified store. Never clean up a failed partial automatically.
* Use safe archive inventory before extraction. Reject traversal, symlinks, collisions, and decompression beyond the configured bound.
* Do not download the approximately 320 GB nodchip Hao release, the approximately 855 GB DL水匠15b release, or the large GCT release in Phase 10R-A.

## Progressive targets

| Target | Position scale | Download ceiling | Required gate |
| --- | ---: | ---: | --- |
| Sample | 1M | 1 GiB | Format, rights, archive, parser, and Rust legality proof. |
| Pilot | 10M | 10 GiB | Approved source only, exact checksums, overlap report. |
| Medium | 50M | 50 GiB | Deduplicated canonical positions and split-leakage report. |
| Large | 100M+ | 100 GiB initially | Recheck rights, free space, source overlap, and curriculum need. |

The Phase 10R-A acquisition stopped at two small approved archives: WCSC32 (297,175 bytes) and Denryu hardware-3 (2,200,967 bytes). Archive inventories are 519,128 and 12,016,210 uncompressed bytes respectively. Their extracted source files remain ignored local data; no normalized dataset was committed.

## Resumption commands

Validate the registry and print the no-write plan:

```sh
PYTHONPATH=training uv run python -m open_shogi_training.data phase10r-validate
PYTHONPATH=training uv run python -m open_shogi_training.data phase10r-dry-run --artifact wcsc32-kifu
```

Acquire only an artifact whose registry state is `approved` and whose training permission is `approved`:

```sh
OPENSHOGI_DATA_ROOT=/path/to/phase10r-data \
  PYTHONPATH=training uv run python -m open_shogi_training.data phase10r-acquire \
  --artifact wcsc32-kifu
```

No command in this plan authorizes a pending, denied, or holdout artifact.
