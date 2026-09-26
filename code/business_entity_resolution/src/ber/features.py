from __future__ import annotations

import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .retrieval import ROUTES


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def safe_similarity(function, a, b):
    return float(function(a, b)) if a and b else 0.0


def acronym(name):
    return "".join(t[0] for t in name.split() if t)


def comparison(a, b):
    an, bn = a["name_core"], b["name_core"]
    aa, ba = a["address_norm"], b["address_norm"]
    at, bt = an.split(), bn.split()
    overlap = set(at) & set(bt)
    result = {
        "name_edit": safe_similarity(Levenshtein.normalized_similarity, an, bn),
        "name_jaro_winkler": safe_similarity(JaroWinkler.normalized_similarity, an, bn),
        "name_token_jaccard": jaccard(at, bt),
        "name_token_containment": len(overlap) / min(len(set(at)), len(set(bt))) if at and bt else 0.0,
        "name_exact": float(bool(an) and an == bn),
        "name_original_exact": float(bool(a["name_norm"]) and a["name_norm"] == b["name_norm"]),
        "name_folded_exact": float(bool(a["name_folded"]) and a["name_folded"] == b["name_folded"]),
        "name_acronym": float(bool(an and bn) and (acronym(an) == bn.replace(" ", "") or acronym(bn) == an.replace(" ", ""))),
        "name_length_ratio": min(len(an), len(bn)) / max(len(an), len(bn), 1),
        "address_edit": safe_similarity(Levenshtein.normalized_similarity, aa, ba),
        "address_token_jaccard": jaccard(aa.split(), ba.split()),
        "address_exact": float(bool(aa) and aa == ba),
        "name_missing": float(not an or not bn), "address_missing": float(not aa or not ba),
    }
    for field in ("country_norm", "postal", "house", "unit"):
        left, right = a[field], b[field]
        known = bool(left and right)
        result[f"{field}_agree"] = float(known and left == right)
        result[f"{field}_conflict"] = float(known and left != right)
        result[f"{field}_missing"] = float(not known)
    result["name_address_interaction"] = result["name_edit"] * result["address_edit"]
    return result


def make_features(pairs, anchors, targets):
    left = anchors if isinstance(anchors, dict) else anchors.set_index("entity_id").to_dict("index")
    right = targets if isinstance(targets, dict) else targets.set_index("entity_id").to_dict("index")
    # Fixed column schema also works when a batch has no candidate pairs.
    blank = {key: "" for key in ["name_core", "name_norm", "name_folded", "address_norm", "country_norm", "postal", "house", "unit"]}
    columns = ROUTES + ["retrieval_score"] + list(comparison(blank, blank))
    rows = []
    for row in pairs.to_dict("records"):
        rows.append({**{key: row[key] for key in ROUTES + ["retrieval_score"]},
                     **comparison(left[row["source1_entity_id"]], right[row["candidate_entity_id"]])})
    result = pd.DataFrame(rows, columns=columns, dtype=np.float32)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("Non-finite matching feature")
    return result


def labels(pairs, truth):
    return np.array([int(b in truth[a]) for a, b in zip(pairs.source1_entity_id, pairs.candidate_entity_id)])


def training_subset(pairs, truth, max_negatives, seed):
    """Keep every retrieved positive; mix hardest and random negatives per anchor."""
    rng = np.random.default_rng(seed)
    y = labels(pairs, truth)
    chosen = []
    for _, positions in pairs.groupby("source1_entity_id", sort=True).indices.items():
        positive = positions[y[positions] == 1]
        negative = positions[y[positions] == 0]
        if max_negatives and len(negative) > max_negatives:
            order = sorted(negative, key=lambda i: (-pairs.iloc[i].retrieval_score, pairs.iloc[i].candidate_entity_id))
            hard_n = max_negatives // 2
            negative = np.r_[order[:hard_n], rng.choice(order[hard_n:], max_negatives - hard_n, replace=False)]
        chosen.extend(positive)
        chosen.extend(negative)
    return pairs.iloc[sorted(map(int, chosen))].reset_index(drop=True)
