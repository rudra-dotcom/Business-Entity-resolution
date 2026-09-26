from __future__ import annotations

import copy
import csv
import gc
import json
import logging
import shutil
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import yaml

from .cleaning import clean_frame
from .data import load_dataset, export_lists, validate_submission
from .features import make_features, labels, training_subset
from .metrics import candidate_metrics, evaluate, pairs_to_lists, predictions_at, tune_threshold
from .models import PairMatcher
from .neural import LocalTransformer, pair_texts
from .retrieval import Fusion, RetrievalIndex, ROUTES, fit_vectorizers, route_ablation
from .split import split_entities, target_partitions
from .util import LOG, environment, grid, json_save, seed_all, sha256, write_tsv


def prepare_data(data_dir, split, labelled=True):
    data = load_dataset(data_dir, split, labelled)
    data.anchors, data.targets = clean_frame(data.anchors), clean_frame(data.targets)
    return data


def subset(frame, ids):
    return frame[frame.entity_id.isin(ids)].reset_index(drop=True)


def annotate_pairs(pairs, truth, countries, scores=None, threshold=None):
    result = pairs.copy()
    result["label"] = labels(pairs, truth)
    result["country"] = pairs.source1_entity_id.map(countries)
    if scores is not None:
        result["match_score"] = scores
    if threshold is not None:
        result["predicted_match"] = (np.asarray(scores) >= threshold).astype(int)
    return result


def error_rows(anchors, truth, predictions, candidates):
    rows = []
    for a in anchors:
        predicted, possible = set(predictions.get(a, [])), set(candidates.get(a, []))
        for target in sorted(truth[a] - predicted):
            rows.append({"source1_entity_id": a, "candidate_entity_id": target,
                         "error": "matching_false_negative" if target in possible else "retrieval_miss"})
        for target in sorted(predicted - truth[a]):
            rows.append({"source1_entity_id": a, "candidate_entity_id": target, "error": "false_positive"})
    return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_id", "error"])


def embedding_training_pairs(anchors, targets, truth, vectorizers, config, max_negatives, seed):
    # Bootstrap hard negatives from lexical retrieval; positives come only from train labels.
    lexical = dict(config)
    index = RetrievalIndex(targets, vectorizers, lexical)
    pairs, _ = index.retrieve(anchors)
    pairs["retrieval_score"] = pairs[ROUTES].max(axis=1)
    pairs = training_subset(pairs, truth, max_negatives, seed)
    present = set(zip(pairs.source1_entity_id, pairs.candidate_entity_id))
    additions = [{"source1_entity_id": a, "candidate_entity_id": b}
                 for a in anchors.entity_id for b in sorted(truth[a]) if (a, b) not in present]
    if additions:
        pairs = pd.concat([pairs, pd.DataFrame(additions)], ignore_index=True)
    return pairs


def retrieval_search(data, splits, config, run_dir):
    section = config["retrieval"]
    reports = run_dir / "reports"
    train_ids = set(splits.loc[splits.split == "train", "source1_entity_id"])
    tune_ids = set(splits.loc[splits.split == "tune", "source1_entity_id"])
    train_anchors, tune_anchors = subset(data.anchors, train_ids), subset(data.anchors, tune_ids)
    owners = target_partitions(data.targets, data.truth, splits)
    train_targets = subset(data.targets, {t for t, label in owners.items() if label == "train"})
    representation_records = pd.concat([train_anchors, train_targets], ignore_index=True)
    candidates = grid(section["grid"], section["max_trials"], config["seed"])
    embeddings = config["embedding"]
    embedding_trials = grid(embeddings["grid"], embeddings["max_trials"], config["seed"]) if embeddings["enabled"] else [{"epochs": 0}]
    trial_rows, weight_rows, histories = [], [], []
    best = None
    trial_number = 0
    for embedding_number, embedding_parameters in enumerate(embedding_trials):
        encoder = None
        embedding_path = None
        if embeddings["enabled"]:
            encoder = LocalTransformer(embeddings["model_path"], {**embeddings, "seed": config["seed"]})
            if embedding_parameters["epochs"]:
                bootstrap_vectors = fit_vectorizers(representation_records, section["base"])
                training_pairs = embedding_training_pairs(train_anchors, train_targets, data.truth,
                    bootstrap_vectors, section["base"], embeddings["negatives_per_anchor"], config["seed"])
                history = encoder.fit_pairs(*pair_texts(training_pairs, train_anchors, train_targets),
                                            labels(training_pairs, data.truth), embedding_parameters)
                histories += [{"embedding_trial": embedding_number, **row} for row in history]
            embedding_path = run_dir / "search_embeddings" / str(embedding_number)
            encoder.save(embedding_path)
        for variation in candidates:
            trial_number += 1
            params = {**section["base"], **variation}
            LOG.info("Retrieval trial %s: %s; embedding %s", trial_number, params, embedding_parameters)
            start = perf_counter()
            vectorizers = fit_vectorizers(representation_records, params)
            index = RetrievalIndex(data.targets, vectorizers, params, encoder)
            raw_train, train_work = index.retrieve(train_anchors)
            raw_tune, tune_work = index.retrieve(tune_anchors)
            # A held-out business must not enter supervised fitting as a negative either.
            raw_train = raw_train[raw_train.candidate_entity_id.isin(set(train_targets.entity_id))].reset_index(drop=True)
            fusion = Fusion.fit(raw_train, data.truth, params["fusion_l2"])
            weight_rows.append({"retrieval_trial": trial_number, "embedding_trial": embedding_number, **fusion.describe()})
            raw_metrics = candidate_metrics(tune_anchors.entity_id, data.truth, pairs_to_lists(raw_tune), len(data.targets))
            elapsed = perf_counter() - start
            for cap in section["candidate_caps"]:
                for threshold in section["thresholds"]:
                    selected = fusion.select(raw_tune, threshold, cap)
                    metrics = candidate_metrics(tune_anchors.entity_id, data.truth, pairs_to_lists(selected), len(data.targets))
                    feasible = (metrics["candidate_recall"] >= section["target_recall"] and
                                metrics["macro_candidate_recall"] >= section["target_macro_recall"])
                    row = {"retrieval_trial": trial_number, "embedding_trial": embedding_number,
                           "parameters": json.dumps(params, sort_keys=True),
                           "embedding_parameters": json.dumps(embedding_parameters, sort_keys=True),
                           "threshold": threshold, "cap": cap, "recall_target_met": feasible,
                           "raw_candidate_recall": raw_metrics["candidate_recall"],
                           "raw_mean_candidates": raw_metrics["mean_candidates"],
                           "index_and_retrieval_seconds": elapsed,
                           "mean_postings_visited": float(tune_work.postings_visited.mean()) if len(tune_work) else 0,
                           **metrics}
                    trial_rows.append(row)
                    # Deterministic tie-breaking; latency is reported but not a noisy selector.
                    key = (feasible, 0 if feasible else min(metrics["candidate_recall"], metrics["macro_candidate_recall"]),
                           -metrics["mean_candidates"], metrics["candidate_recall"])
                    if best is None or key > best["key"]:
                        best = {"key": key, "row": row, "params": params, "vectorizers": vectorizers,
                                "fusion": fusion, "threshold": threshold, "cap": cap,
                                "embedding_parameters": embedding_parameters,
                                "embedding_path": str(embedding_path) if embedding_path else None}
            write_tsv(reports / "retrieval_trials.tsv", trial_rows)
            write_tsv(reports / "retrieval_weights.tsv", weight_rows)
            write_tsv(reports / "embedding_training.tsv", histories,
                      columns=["embedding_trial", "epoch", "mean_loss", "training_pairs"])
            del index, raw_train, raw_tune
            gc.collect()
        if encoder:
            encoder.close()
            del encoder
            gc.collect()
    if best is None:
        raise ValueError("No retrieval configurations requested")
    if not best["row"]["recall_target_met"]:
        LOG.warning("No candidate configuration reached recall targets. Selected best recall; inspect retrieval_misses before relying on this run.")
    artifact = run_dir / "evaluation_artifacts"
    artifact.mkdir()
    if best["embedding_path"]:
        shutil.copytree(best["embedding_path"], artifact / "embedding")
    joblib.dump({key: best[key] for key in ["params", "vectorizers", "fusion", "threshold", "cap", "embedding_parameters"]}, artifact / "retrieval.joblib")
    json_save(reports / "selected_retrieval.json", best["row"])
    return best


def retrieve_selected(data, selected, config, run_dir):
    encoder = LocalTransformer(run_dir / "evaluation_artifacts" / "embedding", config["embedding"]) if config["embedding"]["enabled"] else None
    start = perf_counter()
    index = RetrievalIndex(data.targets, selected["vectorizers"], selected["params"], encoder)
    raw, work = index.retrieve(data.anchors)
    candidates = selected["fusion"].select(raw, selected["threshold"], selected["cap"])
    write_tsv(run_dir / "reports" / "retrieval_work.tsv", work)
    write_tsv(run_dir / "reports" / "candidate_pairs_long.tsv.gz", candidates)
    if encoder:
        encoder.close()
    del index, encoder
    gc.collect()
    return raw, candidates, perf_counter() - start


def report_retrieval(data, splits, raw, candidates, run_dir):
    rows, ablations, misses, entities = [], [], [], []
    candidate_lists = pairs_to_lists(candidates)
    raw_lists = pairs_to_lists(raw)
    for split in ["train", "tune", "audit"]:
        anchors = splits.loc[splits.split == split, "source1_entity_id"].tolist()
        rows.append({"split": split, **candidate_metrics(anchors, data.truth, candidate_lists, len(data.targets))})
        for route, lists in route_ablation(raw).items():
            ablations.append({"split": split, "route": route, **candidate_metrics(anchors, data.truth, lists, len(data.targets))})
        for a in anchors:
            for b in sorted(data.truth[a] - set(candidate_lists.get(a, []))):
                misses.append({"split": split, "source1_entity_id": a, "candidate_entity_id": b,
                               "stage": "fusion_or_cap" if b in raw_lists.get(a, []) else "raw_retrieval"})
            entities.append({"split": split, "source1_entity_id": a, "true_count": len(data.truth[a]),
                             "raw_count": len(raw_lists.get(a, [])), "candidate_count": len(candidate_lists.get(a, [])),
                             "retained_true": len(data.truth[a] & set(candidate_lists.get(a, [])))})
    reports = run_dir / "reports"
    write_tsv(reports / "candidate_summary.tsv", rows)
    write_tsv(reports / "route_ablation.tsv", ablations)
    write_tsv(reports / "candidate_entities.tsv", entities)
    write_tsv(reports / "retrieval_misses.tsv", misses,
              ["split", "source1_entity_id", "candidate_entity_id", "stage"])


def matching_search(data, splits, candidates, training_pool, config, run_dir):
    reports = run_dir / "reports"
    countries = data.anchors.set_index("entity_id").country.to_dict()
    split_ids = {s: splits.loc[splits.split == s, "source1_entity_id"].tolist() for s in ("train", "tune", "audit")}
    split_pairs = {s: candidates[candidates.source1_entity_id.isin(ids)].reset_index(drop=True) for s, ids in split_ids.items()}
    ownership = target_partitions(data.targets, data.truth, splits)
    train_targets = {target for target, partition in ownership.items() if partition == "train"}
    # Learn from a broader TRAIN-only pool. Otherwise a strong candidate filter can
    # remove every negative and leave the final classifier with only one class.
    permitted_pairs = training_pool[training_pool.candidate_entity_id.isin(train_targets)].reset_index(drop=True)
    train_pairs = training_subset(permitted_pairs, data.truth, config["matching"]["max_negatives_per_anchor"], config["seed"])
    x_train = make_features(train_pairs, data.anchors, data.targets)
    y_train = labels(train_pairs, data.truth)
    x_tune = make_features(split_pairs["tune"], data.anchors, data.targets)
    x_audit = make_features(split_pairs["audit"], data.anchors, data.targets)
    write_tsv(reports / "training_pairs.tsv.gz", annotate_pairs(train_pairs, data.truth, countries))
    for name, pairs, features in [("train", train_pairs, x_train), ("tune", split_pairs["tune"], x_tune),
                                  ("audit", split_pairs["audit"], x_audit)]:
        table = pd.concat([pairs[["source1_entity_id", "candidate_entity_id"]], features], axis=1)
        table["label"] = labels(pairs, data.truth)
        write_tsv(reports / "features" / f"{name}.tsv.gz", table)
    trial_rows, threshold_rows, summary_rows, country_rows = [], [], [], []
    choices, all_predictions = {}, {}
    for kind in config["matching"]["models"]:
        section = config["matching"][kind]
        best_key, best_info = None, None
        output_dir = run_dir / "evaluation_artifacts" / kind
        for trial, params in enumerate(grid(section["grid"], section["max_trials"], config["seed"])):
            LOG.info("Matching %s trial %s: %s", kind, trial, params)
            started = perf_counter()
            matcher = PairMatcher(kind, params, {**config["cross_encoder"], "seed": config["seed"]}, config["seed"])
            history = matcher.fit(train_pairs, x_train, y_train, data.anchors, data.targets)
            fit_seconds = perf_counter() - started
            started = perf_counter()
            scores = matcher.predict(split_pairs["tune"], x_tune, data.anchors, data.targets)
            predict_seconds = perf_counter() - started
            optimum, thresholds = tune_threshold(split_ids["tune"], data.truth, split_pairs["tune"], scores,
                                                  config["matching"]["thresholds"])
            trial_rows.append({"model": kind, "trial": trial, "parameters": json.dumps(params, sort_keys=True),
                               "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                               "training_positive_pairs": int(y_train.sum()),
                               "training_negative_pairs": int(len(y_train) - y_train.sum()), **optimum})
            threshold_rows += [{"model": kind, "trial": trial, **row} for row in thresholds]
            write_tsv(reports / "model_trials.tsv", trial_rows)
            write_tsv(reports / "threshold_trials.tsv", threshold_rows)
            write_tsv(reports / "scores" / f"{kind}_trial_{trial}_tune.tsv.gz",
                      annotate_pairs(split_pairs["tune"], data.truth, countries, scores, optimum["threshold"]))
            if history:
                write_tsv(reports / "training" / f"{kind}_trial_{trial}.tsv", history)
            key = (optimum["macro_f0_5"], -optimum["fp"])
            if best_key is None or key > best_key:
                best_key = key
                best_info = {"model": kind, "trial": trial, "parameters": params,
                             "threshold": optimum["threshold"], "tune_macro_f0_5": optimum["macro_f0_5"],
                             "tune_fp": optimum["fp"]}
                matcher.save(output_dir)
            matcher.close()
            del matcher
            gc.collect()
        if best_info is None:
            raise ValueError(f"No matching trials requested for {kind}")
        choices[kind] = best_info
        matcher = PairMatcher.load(output_dir, config["cross_encoder"].get("device"))
        if kind == "boosted_trees" and len(x_tune) and config["matching"].get("importance_repeats", 0):
            importance = []
            rng = np.random.default_rng(config["seed"])
            baseline = best_info["tune_macro_f0_5"]
            for feature in x_tune:
                drops = []
                for _ in range(config["matching"]["importance_repeats"]):
                    shuffled = x_tune.copy()
                    shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
                    scores = matcher.predict(split_pairs["tune"], shuffled, data.anchors, data.targets)
                    score = evaluate(split_ids["tune"], data.truth,
                        predictions_at(split_pairs["tune"], scores, best_info["threshold"]))[0]["macro_f0_5"]
                    drops.append(baseline - score)
                importance.append({"feature": feature, "mean_macro_f0_5_drop": float(np.mean(drops)),
                                   "std_drop": float(np.std(drops)), "split": "tune"})
            write_tsv(reports / "boosted_tree_permutation_importance.tsv", importance)
        for split, features in [("tune", x_tune), ("audit", x_audit)]:
            pairs = split_pairs[split]
            started = perf_counter()
            scores = matcher.predict(pairs, features, data.anchors, data.targets)
            predict_seconds = perf_counter() - started
            predictions = predictions_at(pairs, scores, best_info["threshold"])
            metrics, entity_rows = evaluate(split_ids[split], data.truth, predictions, countries)
            summary_rows.append({"model": kind, "split": split, "threshold": best_info["threshold"],
                                 "predict_seconds": predict_seconds, **metrics})
            all_predictions[(kind, split)] = scores
            write_tsv(reports / "scores" / f"{kind}_{split}.tsv.gz", annotate_pairs(pairs, data.truth, countries, scores, best_info["threshold"]))
            write_tsv(reports / "entities" / f"{kind}_{split}.tsv", entity_rows)
            write_tsv(reports / "errors" / f"{kind}_{split}.tsv",
                      error_rows(split_ids[split], data.truth, predictions, pairs_to_lists(pairs)))
            for country in sorted({countries[a] for a in split_ids[split]}):
                ids = [a for a in split_ids[split] if countries[a] == country]
                country_rows.append({"model": kind, "split": split, "country": country,
                                     **evaluate(ids, data.truth, predictions)[0]})
        matcher.close()
        del matcher
        gc.collect()
        write_tsv(reports / "model_comparison.tsv", summary_rows)
        write_tsv(reports / "country_scores.tsv", country_rows)
        json_save(reports / "selected_matchers.json", choices)
    # Choose solely on tuning results; the audit cannot silently choose the winner.
    winner = max(choices, key=lambda kind: (choices[kind]["tune_macro_f0_5"], -choices[kind]["tune_fp"],
                                          -config["matching"]["models"].index(kind)))
    comparison_rows = []
    for split in ("tune", "audit"):
        frame = annotate_pairs(split_pairs[split], data.truth, countries)
        decisions = []
        for kind in choices:
            scores = all_predictions[(kind, split)]
            frame[f"score_{kind}"] = scores
            frame[f"accept_{kind}"] = (scores >= choices[kind]["threshold"]).astype(int)
            decisions.append(frame[f"accept_{kind}"].to_numpy())
        frame["models_disagree"] = np.ptp(np.asarray(decisions), axis=0) if decisions else 0
        write_tsv(reports / f"comparison_pairs_{split}.tsv.gz", frame)
        for i, first in enumerate(choices):
            for second in list(choices)[i+1:]:
                a, b = frame[f"accept_{first}"].to_numpy(), frame[f"accept_{second}"].to_numpy()
                y = frame.label.to_numpy()
                comparison_rows.append({"split": split, "model_a": first, "model_b": second,
                    "disagreements": int(np.sum(a != b)), "both_wrong": int(np.sum((a != y) & (b != y))),
                    "only_a_correct": int(np.sum((a == y) & (b != y))),
                    "only_b_correct": int(np.sum((a != y) & (b == y)))})
    write_tsv(reports / "model_disagreements.tsv", comparison_rows,
              ["split", "model_a", "model_b", "disagreements", "both_wrong", "only_a_correct", "only_b_correct"])
    json_save(reports / "selected_model.json", {"model": winner, "selection_split": "tune", "ensemble": False})
    return choices, winner


def final_refit(data, selected, choices, config, run_dir):
    artifact = run_dir / "artifacts"
    artifact.mkdir()
    encoder = None
    if config["embedding"]["enabled"]:
        encoder = LocalTransformer(config["embedding"]["model_path"], {**config["embedding"], "seed": config["seed"]})
        if selected["embedding_parameters"]["epochs"]:
            vectorizers = fit_vectorizers(pd.concat([data.anchors, data.targets]), selected["params"])
            pairs = embedding_training_pairs(data.anchors, data.targets, data.truth, vectorizers,
                selected["params"], config["embedding"]["negatives_per_anchor"], config["seed"])
            history = encoder.fit_pairs(*pair_texts(pairs, data.anchors, data.targets), labels(pairs, data.truth), selected["embedding_parameters"])
            write_tsv(run_dir / "reports" / "embedding_final_training.tsv", history)
        encoder.save(artifact / "embedding")
    vectorizers = fit_vectorizers(pd.concat([data.anchors, data.targets]), selected["params"])
    index = RetrievalIndex(data.targets, vectorizers, selected["params"], encoder)
    raw, _ = index.retrieve(data.anchors)
    fusion = Fusion.fit(raw, data.truth, selected["params"]["fusion_l2"])
    pairs = fusion.select(raw, selected["threshold"], selected["cap"])
    write_tsv(run_dir / "reports" / "final_refit_candidate_metrics.tsv", [{"split": "all_train_after_refit",
        **candidate_metrics(data.anchors.entity_id, data.truth, pairs_to_lists(pairs), len(data.targets))}])
    joblib.dump({"params": selected["params"], "vectorizers": vectorizers, "fusion": fusion,
                 "threshold": selected["threshold"], "cap": selected["cap"],
                 "embedding_parameters": selected["embedding_parameters"]}, artifact / "retrieval.joblib")
    json_save(run_dir / "reports" / "final_retrieval_weights.json", fusion.describe())
    write_tsv(run_dir / "reports" / "final_retrieval_weights.tsv", [{"feature": k, "weight": v} for k, v in fusion.describe().items()])
    training_pool = raw.copy()
    training_pool["retrieval_score"] = fusion.score(raw)
    if encoder:
        encoder.close()
    del index, encoder, raw
    gc.collect()
    pairs = training_subset(training_pool, data.truth, config["matching"]["max_negatives_per_anchor"], config["seed"])
    del training_pool
    features, y = make_features(pairs, data.anchors, data.targets), labels(pairs, data.truth)
    final_features = pd.concat([pairs[["source1_entity_id", "candidate_entity_id"]], features], axis=1)
    final_features["label"] = y
    write_tsv(run_dir / "reports" / "features" / "final_train.tsv.gz", final_features)
    for kind, choice in choices.items():
        LOG.info("Refitting %s on all labelled training entities", kind)
        matcher = PairMatcher(kind, choice["parameters"], {**config["cross_encoder"], "seed": config["seed"]}, config["seed"])
        history = matcher.fit(pairs, features, y, data.anchors, data.targets)
        matcher.save(artifact / kind)
        if history:
            write_tsv(run_dir / "reports" / "training" / f"{kind}_final.tsv", history)
        matcher.close()
        del matcher
        gc.collect()


def predict(data_dir, artifact_dir, output_dir, device=None):
    artifact_dir, output_dir = Path(artifact_dir), Path(output_dir)
    manifest = json.loads((artifact_dir / "bundle.json").read_text())
    config, choices, winner = manifest["config"], manifest["matchers"], manifest["selected_model"]
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Prediction output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_data(data_dir, "test", labelled=False)
    state = joblib.load(artifact_dir / "retrieval.joblib")
    encoder = None
    if config["embedding"]["enabled"]:
        encoder = LocalTransformer(artifact_dir / "embedding", {**config["embedding"], **({"device": device} if device else {})})
    index = RetrievalIndex(data.targets, state["vectorizers"], state["params"], encoder)
    target_records = data.targets.set_index("entity_id").to_dict("index")
    matchers = {kind: PairMatcher.load(artifact_dir / kind, device) for kind in choices}
    parameter_count = (encoder.parameter_count if encoder else 0) + sum(
        matcher.model.parameter_count for matcher in matchers.values() if matcher.kind == "cross_encoder")
    if parameter_count > 8_000_000_000:
        raise ValueError("Combined neural models exceed eight billion parameters")
    handles, writers = {}, {}
    candidate_file = output_dir / "candidate_pairs.tsv"
    try:
        handles["candidate"] = candidate_file.open("w", newline="", encoding="utf-8")
        writers["candidate"] = csv.writer(handles["candidate"], delimiter="\t", lineterminator="\n")
        writers["candidate"].writerow(["source1_entity_id", "candidate_entity_ids"])
        for kind in matchers:
            directory = output_dir / "models" / kind
            directory.mkdir(parents=True)
            handles[kind] = (directory / "matching_results.tsv").open("w", newline="", encoding="utf-8")
            writers[kind] = csv.writer(handles[kind], delimiter="\t", lineterminator="\n")
            writers[kind].writerow(["source1_entity_id", "matched_entity_ids"])
            handles[f"scores_{kind}"] = (directory / "pair_scores.tsv").open("w", newline="", encoding="utf-8")
            writers[f"scores_{kind}"] = csv.writer(handles[f"scores_{kind}"], delimiter="\t", lineterminator="\n")
            writers[f"scores_{kind}"].writerow(["source1_entity_id", "candidate_entity_id", "match_score", "accepted"])
        work_rows = []
        size = config["query_batch_size"]
        for start in range(0, len(data.anchors), size):
            anchors = data.anchors.iloc[start:start + size].reset_index(drop=True)
            raw, work = index.retrieve(anchors)
            pairs = state["fusion"].select(raw, state["threshold"], state["cap"])
            lists = pairs_to_lists(pairs)
            work["final_candidates"] = work.source1_entity_id.map(lambda a: len(lists.get(a, [])))
            work_rows.extend(work.to_dict("records"))
            features = make_features(pairs, anchors, target_records)
            for a in anchors.entity_id:
                writers["candidate"].writerow([a, ",".join(sorted(lists.get(a, [])))])
            for kind, matcher in matchers.items():
                scores = matcher.predict(pairs, features, anchors, target_records)
                threshold = choices[kind]["threshold"]
                predictions = predictions_at(pairs, scores, threshold)
                for a in anchors.entity_id:
                    writers[kind].writerow([a, ",".join(sorted(predictions.get(a, [])))])
                for row, score in zip(pairs.itertuples(index=False), scores):
                    writers[f"scores_{kind}"].writerow([row.source1_entity_id, row.candidate_entity_id,
                                                        f"{score:.9g}", int(score >= threshold)])
            LOG.info("Predicted %s/%s test entities", min(start + size, len(data.anchors)), len(data.anchors))
        write_tsv(output_dir / "retrieval_work.tsv", work_rows)
    finally:
        for file in handles.values():
            file.close()
        for matcher in matchers.values():
            matcher.close()
        if encoder:
            encoder.close()
    validations = {}
    for kind in matchers:
        directory = output_dir / "models" / kind
        shutil.copy2(candidate_file, directory / "candidate_pairs.tsv")
        validations[kind] = validate_submission(directory / "matching_results.tsv", candidate_file, Path(data_dir) / "test")
    shutil.copy2(output_dir / "models" / winner / "matching_results.tsv", output_dir / "matching_results.tsv")
    json_save(output_dir / "validation.json", validations)
    json_save(output_dir / "selection.json", {"model": winner, "selection_split": "tune", "ensemble": False,
                                              "neural_parameter_count": parameter_count,
                                              "candidate_sha256": sha256(candidate_file)})
    return validations


def run(data_dir, config, run_dir, do_predict=True):
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory must be new or empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    reports = run_dir / "reports"
    reports.mkdir()
    file_handler = logging.FileHandler(run_dir / "run.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(file_handler)
    started = perf_counter()
    manifest = {"status": "running", "environment": environment(), "data_dir": str(Path(data_dir).resolve())}
    try:
        seed_all(config["seed"])
        json_save(run_dir / "config.json", config)
        manifest["data_hashes"] = {str(p.relative_to(data_dir)): sha256(p) for p in sorted(Path(data_dir).glob("*/*.tsv"))}
        json_save(run_dir / "manifest.json", manifest)
        data = prepare_data(data_dir, "train")
        write_tsv(reports / "cleaned_source1.tsv.gz", data.anchors)
        write_tsv(reports / "cleaned_targets.tsv.gz", data.targets)
        splits = split_entities(data.anchors, data.truth, config["split"]["tune_fraction"],
                                config["split"]["audit_fraction"], config["seed"])
        write_tsv(reports / "splits.tsv", splits)
        write_tsv(reports / "data_profile.tsv", [{"collection": name, "country": country, "rows": len(part),
                 "missing_names": int((part.name_norm == "").sum()),
                 "missing_addresses": int((part.address_norm == "").sum())}
                 for name, frame in [("source1", data.anchors), ("targets", data.targets)]
                 for country, part in frame.groupby("country", dropna=False)])
        selected = retrieval_search(data, splits, config, run_dir)
        raw, candidates, elapsed = retrieve_selected(data, selected, config, run_dir)
        report_retrieval(data, splits, raw, candidates, run_dir)
        train_ids = set(splits.loc[splits.split == "train", "source1_entity_id"])
        training_pool = raw[raw.source1_entity_id.isin(train_ids)].copy()
        training_pool["retrieval_score"] = selected["fusion"].score(training_pool)
        del raw
        choices, winner = matching_search(data, splits, candidates, training_pool, config, run_dir)
        del training_pool
        if config["final_refit"]:
            final_refit(data, selected, choices, config, run_dir)
        else:
            shutil.copytree(run_dir / "evaluation_artifacts", run_dir / "artifacts")
        json_save(run_dir / "artifacts" / "bundle.json", {"config": config, "matchers": choices,
                  "selected_model": winner, "final_refit": config["final_refit"], "ensemble": False,
                  "audit_note": "Audit evaluates the train-split models before any all-data refit; it is never used to select parameters or winner."})
        if do_predict:
            predict(data_dir, run_dir / "artifacts", run_dir / "output")
        manifest.update(status="complete", selected_model=winner, elapsed_seconds=perf_counter() - started)
        json_save(run_dir / "manifest.json", manifest)
        LOG.info("Run complete: %s", run_dir)
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", elapsed_seconds=perf_counter() - started)
        json_save(run_dir / "manifest.json", manifest)
        raise
    finally:
        LOG.removeHandler(file_handler)
        file_handler.close()
