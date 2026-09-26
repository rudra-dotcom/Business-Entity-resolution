from __future__ import annotations

import numpy as np


def entity_metrics(truth, predicted):
    truth, predicted = set(truth), set(predicted)
    tp = len(truth & predicted)
    fp, fn = len(predicted - truth), len(truth - predicted)
    # Challenge-specific empty/empty convention; otherwise count form of F_0.5.
    f = 1.0 if not truth and not predicted else (5 * tp / (5 * tp + 4 * fp + fn) if tp else 0.0)
    return {"tp": tp, "fp": fp, "fn": fn, "f0_5": f,
            "precision": tp / len(predicted) if predicted else (1.0 if not truth else 0.0),
            "recall": tp / len(truth) if truth else (1.0 if not predicted else 0.0),
            "true_count": len(truth), "predicted_count": len(predicted)}


def evaluate(anchors, truth, predictions, countries=None):
    rows = []
    for anchor in anchors:
        rows.append({"source1_entity_id": anchor, "country": (countries or {}).get(anchor, ""),
                     **entity_metrics(truth[anchor], predictions.get(anchor, set()))})
    tp, fp, fn = (sum(row[k] for row in rows) for k in ("tp", "fp", "fn"))
    singleton = [r for r in rows if r["true_count"] == 0]
    result = {"macro_f0_5": float(np.mean([r["f0_5"] for r in rows])) if rows else 0.0,
              "micro_precision": tp / (tp + fp) if tp + fp else 0.0,
              "micro_recall": tp / (tp + fn) if tp + fn else 0.0,
              "tp": tp, "fp": fp, "fn": fn, "entities": len(rows),
              "singleton_count": len(singleton),
              "singleton_accuracy": float(np.mean([r["f0_5"] for r in singleton])) if singleton else None}
    return result, rows


def candidate_metrics(anchors, truth, candidates, n_targets):
    anchors = list(anchors)
    counts = [len(candidates.get(a, [])) for a in anchors]
    total_true = sum(len(truth[a]) for a in anchors)
    retained = sum(len(truth[a] & set(candidates.get(a, []))) for a in anchors)
    per = [len(truth[a] & set(candidates.get(a, []))) / len(truth[a]) for a in anchors if truth[a]]
    ceiling = {a: truth[a] & set(candidates.get(a, [])) for a in anchors}
    return {"candidate_recall": retained / total_true if total_true else 1.0,
            "macro_candidate_recall": float(np.mean(per)) if per else 1.0,
            "recall_ceiling_macro_f0_5": evaluate(anchors, truth, ceiling)[0]["macro_f0_5"],
            "candidate_pairs": sum(counts), "mean_candidates": float(np.mean(counts)) if counts else 0.0,
            "p95_candidates": float(np.percentile(counts, 95)) if counts else 0.0,
            "max_candidates": max(counts, default=0),
            "reduction_ratio": 1 - sum(counts) / max(1, len(anchors) * n_targets)}


def pairs_to_lists(pairs):
    result = {}
    for row in pairs.itertuples(index=False):
        result.setdefault(row.source1_entity_id, set()).add(row.candidate_entity_id)
    return result


def predictions_at(pairs, scores, threshold):
    result = {}
    for row, score in zip(pairs.itertuples(index=False), scores):
        if score >= threshold:
            result.setdefault(row.source1_entity_id, set()).add(row.candidate_entity_id)
    return result


def tune_threshold(anchors, truth, pairs, scores, thresholds):
    scores = np.asarray(scores)
    if scores.shape != (len(pairs),) or not np.isfinite(scores).all():
        raise ValueError("Invalid matching scores")
    # Explicit predict-none option matters for singleton-heavy validation sets.
    values = sorted(set([0.0, 1.000001] + list(map(float, thresholds))))
    rows = []
    for t in values:
        metrics, _ = evaluate(anchors, truth, predictions_at(pairs, scores, t))
        rows.append({"threshold": t, **metrics})
    best = max(rows, key=lambda r: (r["macro_f0_5"], -r["fp"], r["threshold"]))
    return best, rows
