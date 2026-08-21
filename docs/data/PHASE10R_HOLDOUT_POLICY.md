# Phase 10R holdout policy

Holdout status is assigned before normalization or overlap inspection. A recent or public evaluation artifact remains protected even when its raw-record license appears permissive. The policy is recorded in [`configs/phase10r/holdout-policy.yaml`](../../configs/phase10r/holdout-policy.yaml).

## Reserved scope

* WCSC33, WCSC34, WCSC35, and WCSC36 official final archives.
* Recent Denryu TSEC archives, including Denryu 4–6 TSEC records listed in the registry.
* Designated-position events, public evaluation/test sets, and any unpublished teacher derivative.
* Any artifact whose rights are pending but whose content could become a future final or public test boundary.

## Rules

1. Reserve artifact IDs and, when rights permit, record only archive hashes before any position inspection.
2. Do not train, normalize, deduplicate against, or inspect public-test positions from reserved artifacts.
3. Do not let source priority override holdout status.
4. Do not call a pending-permission artifact an approved holdout release; pending rights and holdout protection are separate states.
5. Split games before positions. A game/history identity must not cross train, validation, test, or reserved splits.
6. Canonical-position and transposition overlap checks are required before a reserved artifact can be released from holdout.

No reserved WCSC33–36 or Denryu TSEC archive was downloaded or inspected in Phase 10R-A.
