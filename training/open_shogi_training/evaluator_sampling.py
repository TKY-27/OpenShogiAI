"""Coverage-first, finite exposure accounting for the R4-C2 pool."""

from __future__ import annotations

from collections import Counter, deque

import numpy as np
import torch


def exposure_summary(data: dict, counts: torch.Tensor) -> dict:
    values = counts.numpy()

    def distribution(v):
        return {
            "pool": len(v),
            "seen": int(np.count_nonzero(v)),
            "exposures": int(v.sum()),
            "median": float(np.median(v)) if len(v) else 0,
            "p95": float(np.quantile(v, 0.95)) if len(v) else 0,
            "maximum": int(v.max()) if len(v) else 0,
        }

    result = distribution(values)
    result["coverage"] = result["seen"] / len(values)
    result["sources"] = {
        str(source): distribution(values[data["origins"] == source])
        for source in np.unique(data["origins"])
    }
    result["sequences"] = distribution(np.bincount(data["sequences"], weights=values))
    result["round_sources"] = {
        str(source): distribution(values[data["sources"] == source]) for source in (0, 1)
    }
    return result


def coverage_order(data: dict, counts: torch.Tensor, config: dict, generator) -> torch.Tensor:
    """Least-exposed first within each stratum; never refill exhausted pools by cycling.

    A sequence is an original trajectory where available, otherwise a documented
    inferred contiguous PSV segment. It is not a claim of independent source games.
    """
    policy = config["coverage_sampler"]
    size = policy["epoch_examples"]
    batch = config["batch_size"]
    limit = policy["maximum_per_sequence_batch"]
    maximum_sequence = policy["maximum_per_sequence"]
    if min(size, batch, limit, maximum_sequence, *policy["maximum_per_example"]) < 1:
        raise ValueError("invalid finite exposure limits")
    values = counts.numpy()
    sequences = data["sequences"]
    totals = np.bincount(sequences, weights=values).astype(np.int64)
    picked = []
    for source, fraction in enumerate(config["source_fractions"]):
        for group, group_fraction in (
            enumerate(config["sampling_fractions"]) if source == 0 else [(None, 1.0)]
        ):
            mask = (data["sources"] == source) & (values < policy["maximum_per_example"][source])
            if group is not None:
                mask &= data["groups"] == group
            members = np.flatnonzero(mask)
            random = torch.randperm(len(members), generator=generator).numpy()
            members = members[random]
            members = members[np.argsort(values[members], kind="stable")]
            quota = int(size * fraction * group_fraction)
            if quota == 0:
                continue
            used = 0
            for index in members:
                sequence = sequences[index]
                if totals[sequence] >= maximum_sequence:
                    continue
                picked.append(int(index))
                totals[sequence] += 1
                used += 1
                if used >= quota:
                    break
    if not picked:
        return torch.empty(0, dtype=torch.int64)
    order = torch.randperm(len(picked), generator=generator).numpy()
    pending = deque(picked[i] for i in order)
    result = []
    # Bounded one-pass packing. A short final batch is permitted; never duplicate
    # a row to fill it, and never mix many adjacent positions into one batch.
    while pending:
        available = len(pending)
        current, seen = [], Counter()
        for _ in range(available):
            index = pending.popleft()
            sequence = int(sequences[index])
            if seen[sequence] == limit:
                pending.append(index)
                continue
            current.append(index)
            seen[sequence] += 1
            if len(current) == batch:
                break
        if len(current) < batch and pending:
            # The remaining rows cannot form a compliant full batch. Leave them
            # unused for this finite pass; no count has been committed yet.
            break
        result.extend(current)
    return torch.tensor(result, dtype=torch.int64)
