# Contributing

Run the diagnostic bootstrap and full gate before submitting changes:

```sh
./scripts/bootstrap_macos.sh
make check
```

Keep production engine code independently implemented. Do not copy, port, translate, or link
code from another shogi engine or library. An external engine may be used only as a separately
installed process through a documented protocol such as USI.

Keep changes focused, add regression and failure-path tests, and record material rules,
algorithms, standards, or protocols in `docs/references.md`. Update architecture, provenance,
data, model, third-party, and operational documentation when behavior changes.

Do not commit raw or processed datasets, generated weights, checkpoints, teacher binaries,
teacher evaluation files, local artifacts, secrets, production data, or machine-specific
paths. New data sources require source-scoped rights evidence and explicit machine-learning and
redistribution decisions.

Project-owned contributions are accepted under `AGPL-3.0-only`. Contributions must not
misrepresent third-party or external-material licenses.
