# AobaZero source audit

Review date: 2026-08-03 (Asia/Tokyo)

## Decision

Approval is limited to the 100 exact CSA files enumerated in
`configs/data_source_objects/aobazero_no_noise.yaml`: `w4745.csa` down to
`w4235.csa` in steps of five, excluding unavailable `w4495.csa`, `w4425.csa`, and
`w4420.csa`. The registry does not approve weights, bulk training archives, executables,
source releases, dynamically discovered URLs, or any other file.

This file-scoped sample is enabled and approved for machine-learning use and redistribution
as Public Domain material. Public accessibility alone was not treated as a rights grant.
The decision combines one immutable official project document that (a) identifies the
official AobaZero page as the publication location for collected game records, (b) says the
page publishes project-created game records and game samples, and (c) assigns Public Domain
status to project material other than the named USI engine. The pinned official sample index
then identifies every file in the selected exact 100-object subset as an AobaZero no-noise
self-play sample. This is a file-scoped provenance decision, not an inference from the
source-code license alone.

## Official evidence

- Official repository: <https://github.com/kobanium/aobazero>
- Immutable English README at audited commit:
  <https://raw.githubusercontent.com/kobanium/aobazero/5eb944165300d5b88924c917a147e80d9d173eed/README_en.md>
- Immutable Japanese README at the same audited commit:
  <https://raw.githubusercontent.com/kobanium/aobazero/5eb944165300d5b88924c917a147e80d9d173eed/README.md>
- Immutable license file at audited commit:
  <https://raw.githubusercontent.com/kobanium/aobazero/5eb944165300d5b88924c917a147e80d9d173eed/license.txt>
- Official distribution page: <http://www.yss-aya.com/aobazero/>
- Official no-noise sample index:
  <http://www.yss-aya.com/aobazero/no_noise/sample.html>
- Official robots policy: <http://www.yss-aya.com/robots.txt>

Rights quotation from the immutable official README (13 words):

> USI engine aobaz belongs to GPL v3. Others are in the public domain.

The matching Japanese rights sentence is also pinned:

> それ以外はpublic domainです。

The immutable Japanese README itself links the official distribution page while describing
the collected records, weights, and record samples published there. The no-noise index labels
the exact files as self-play samples and says they were not used for AobaZero's own training.
That latter description is not interpreted as a restriction on downstream use.

Pinned evidence identities observed on 2026-08-03 are:

- English README: `018bf5496101d0034f8a6559324c298b6adcec88d608e288b6bc49108b6975bc`
- Japanese README: `cfb7bafc4e7942ae8efef2dd9b86bc8d4585c80335400d76b53fa21446220181`
- repository license file: `3e5643e0fb6844379ef097542d2391885df782081bcf67b2bbf9b55a6a2a4a6f`
- sample index containing 209 CSA references: `178b6bfdd4a128eef14bc1ae7d057fae3781230cef9dc5cc3e2351291e15f9ad`

The live robots snapshot is intentionally not hash-pinned because access policy can change;
every acquisition run fetches it first and fails closed on denial. All other evidence is
hash-pinned and requires a new human review if it drifts.

## Robots, access, and transport

The audited `robots.txt` has a wildcard denial only for `/secret/`; the approved
`/aobazero/no_noise/` paths are not denied. Acquisition still performs a live bounded robots
check and fails closed if it cannot confirm access.

The official sample host serves these files over plain HTTP. HTTPS did not provide a usable
endpoint during review, so `allow_insecure_http` is explicitly true only for this source and
host. This permits on-path modification before the first trusted hash exists. The downloader
records SHA-256, ETag, Last-Modified, byte size, retrieval time, and evidence in an immutable
local object store, but those integrity records do not eliminate the initial transport risk.
No credentials or private information are sent.

## Catalog and operational limits

The pinned official sample index references 209 CSA objects (`w4745.csa` through
`w3705.csa` in steps of five) and includes all 100 selected objects. The catalog deliberately
approves only the bounded `w4745.csa` through `w4235.csa` subset, excluding `w4495.csa`,
`w4425.csa`, and `w4420.csa` after the server returned HTTP 404 for each on 2026-08-03.
The 36 selected lower candidates from `w4410.csa` through `w4235.csa` each returned HTTP 206
to a one-byte, 0.25 req/s audit request. The catalog is versioned and never generated
dynamically at acquisition time. Each acquisition request is limited to 1 MiB, the source
rate is 0.25 requests/second, concurrency is one, and one run is sample-only with at most 100
files. Redirects remain subject to the same scheme, host, port, and path policy.

Before any CSA request, the downloader saves the exact live robots file, both immutable
rights READMEs, immutable license file, and official sample index under the ignored root's
content-addressed `evidence/sha256/` tree. Every completed JSONL event records each evidence
URL, local relative path, retrieval time, byte size, media type, and SHA-256. Acquisition
fails if a pinned evidence digest changes or any exact catalog URL is absent from the parsed
CSA link/`Kifu.load` reference set. References outside the approved 100-object subset do not
expand the catalog.

Verdict: **approved only for the exact 100-object no-noise catalog**.
