# Phase 10R license and permission matrix

This matrix is a compact view of the detailed artifact rows in [`source-registry.yaml`](../../configs/phase10r/source-registry.yaml). `pending_permission` means “do not acquire for training”; it is not a temporary approval. The matrix records raw-record training permission separately from rights in derived weights and redistribution.

| Family / exact scope | Artifact state | Training | Derived weights | Redistribute raw/normalized | Commercial use | Attribution / boundary |
| --- | --- | --- | --- | --- | --- | --- |
| AobaZero exact Phase 3 100-object catalog | approved | approved | pending | approved under existing provenance | pending | Keep AobaZero/source-host attribution; no live-index expansion. |
| WCSC 1–29, 31–32 official archives | approved | approved | pending | pending | pending | Preserve program, date, event, or source attribution from the [CSA archive notice](https://www.computer-shogi.org/kifu/kifu.html). |
| WCSC 33–36 | reserved_holdout | reserved | pending | pending | pending | Recent finals are not acquired or inspected in this phase. |
| Denryu `kifu_dr5hdw3.zip` | approved | approved | pending | pending | pending | Official hardware-3 notice says unrestricted game-record use; preserve event/program/date/source attribution. |
| Other Denryu production/TSEC/hardware/designated archives | pending or reserved | pending/reserved | pending | pending | pending | Do not generalize hardware-3 wording. |
| GCT HCPE/HCPE3 and mixed-lineage releases | pending_permission | pending | pending | pending | pending | [Public release article](https://tadaoyamaoka.hatenablog.com/entry/2021/05/06/223701) is availability evidence, not a complete reuse grant. |
| nodchip Hao/Tanuki/Suisho repositories | pending_permission | pending | pending | pending | pending | Repository metadata and a code license do not settle dataset/teacher rights. |
| Taya/Yaneura 36SFEN/5247/KIF/PSV artifacts | pending_permission | pending | pending | pending | pending | Need exact artifact URL and permission chain. |
| QPD `QPD_train.7z` | pending_permission | pending | pending | pending | pending | [QPD notice](https://qhapaq.hatenablog.com/entry/2021/11/23/220251) requires disclosure of use; broader scope unresolved. |
| Floodgate filtered records | pending_permission | pending | pending | pending | pending | Require operator/program permission and exact filter manifest. |
| Existing public evaluation/test boundary | denied | denied | denied | denied | denied | Never train on or bulk-acquire the evaluation boundary. |
| DL水匠15b knowledge-distilled release | pending_permission | pending | pending | pending | pending | Transformation lineage is documented; complete data/derivative grant is not. |
| Lishogi API exports | pending_permission | pending | pending | pending | pending | API rate limits do not grant third-party ML rights; account authorization required. |
| Bonanza `fv.bin`/book/hash prior art | approved_local_only | local inspection only | denied | denied | denied | Independently implement concepts; do not import or ship third-party tables. |

## Permission-request status

Drafts are stored under [`docs/data/permission-requests/`](permission-requests/). They are intentionally unsent. Each draft asks separately about: training, derived weights, raw/normalized redistribution, commercial use, attribution, lineage, and holdout/public-test handling.
