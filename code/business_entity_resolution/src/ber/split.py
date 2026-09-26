from __future__ import annotations

import numpy as np
import pandas as pd


def split_entities(anchors, truth, tune_fraction, audit_fraction, seed):
    """Group anchors sharing a labelled target; never randomly split pairs."""
    ids = sorted(anchors.entity_id)
    parent = {x: x for x in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owners = {}
    for a in ids:
        for target in sorted(truth[a]):
            if target in owners:
                parent[find(a)] = find(owners[target])
            owners[target] = a
    groups = {}
    for a in ids:
        groups.setdefault(find(a), []).append(a)
    if len(groups) < 10:
        raise ValueError("Need at least 10 independent Source 1 groups for train/tune/audit splitting")
    fractions = np.array([1 - tune_fraction - audit_fraction, tune_fraction, audit_fraction])
    if np.any(fractions <= 0):
        raise ValueError("Train, tune and audit fractions must all be positive")
    # Stratify where possible by country and singleton status, with deterministic group allocation.
    country = anchors.set_index("entity_id").country_norm.to_dict()
    strata = {}
    for group, members in groups.items():
        key = (country[members[0]], any(truth[a] for a in members))
        strata.setdefault(key, []).append(group)
    rng = np.random.default_rng(seed)
    assignment = {}
    for keys in strata.values():
        keys = sorted(keys)
        rng.shuffle(keys)
        n = len(keys)
        if n >= 3:
            n_tune = max(1, int(round(n * tune_fraction)))
            n_audit = max(1, int(round(n * audit_fraction)))
            if n_tune + n_audit >= n:
                n_tune = n_audit = 1
            labels = ["tune"] * n_tune + ["audit"] * n_audit + ["train"] * (n - n_tune - n_audit)
        else:
            labels = list(rng.choice(["train", "tune", "audit"], size=n, p=fractions))
        assignment.update(zip(keys, labels))
    rows = [{"source1_entity_id": a, "group_id": find(a), "split": assignment[find(a)],
             "country": country[a], "true_count": len(truth[a])} for a in ids]
    frame = pd.DataFrame(rows)
    if set(frame.split) != {"train", "tune", "audit"}:
        raise ValueError("Unable to create nonempty splits; use more records or adjust split fractions")
    for label in ("train", "tune"):
        if frame.loc[frame.split == label, "true_count"].sum() == 0:
            raise ValueError(f"{label} split has no positive links; use another seed or more data")
    return frame


def target_partitions(targets, truth, split_frame):
    """Keep known tune/audit businesses out of train-only representation fitting.

    Unlinked distractors are assigned deterministically to train. Retrieval evaluation
    still searches the entire target collection, including competing businesses.
    """
    labels = split_frame.set_index("source1_entity_id").split.to_dict()
    ownership = {}
    for anchor, ids in truth.items():
        for target in ids:
            previous = ownership.setdefault(target, labels[anchor])
            if previous != labels[anchor]:
                raise AssertionError("Shared target crossed entity splits")
    return {row.entity_id: ownership.get(row.entity_id, "train") for row in targets.itertuples()}
