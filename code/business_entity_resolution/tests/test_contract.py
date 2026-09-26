from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ber.cleaning import clean_frame
from ber.config import load_config
from ber.data import SOURCE_COLUMNS, export_lists, load_dataset, validate_submission
from ber.features import comparison
from ber.metrics import candidate_metrics, entity_metrics, evaluate, tune_threshold
from ber.retrieval import Fusion, RetrievalIndex, fit_vectorizers
from ber.split import split_entities, target_partitions
from ber.synthetic import create_dataset


PROJECT = Path(__file__).resolve().parents[1]


def record(id_, name, address="", country="France"):
    return dict(zip(SOURCE_COLUMNS, [id_, name, address, country]))


def test_metric_matches_challenge_and_singletons():
    assert entity_metrics({"a", "b"}, {"a", "b", "c"})["f0_5"] == pytest.approx(5/7)
    assert entity_metrics({"a", "b"}, {"a"})["f0_5"] == pytest.approx(5/6)
    assert entity_metrics(set(), set())["f0_5"] == 1
    assert entity_metrics(set(), {"a"})["f0_5"] == 0
    assert entity_metrics({"a"}, set())["f0_5"] == 0
    truth = {"one": {"a", "b"}, "singleton": set()}
    result, _ = evaluate(list(truth), truth, {"one": {"a"}})
    assert result["macro_f0_5"] == pytest.approx((5/6 + 1)/2)


def test_retrieval_ceiling_counts_missing_true_pairs():
    metrics = candidate_metrics(["q", "s"], {"q": {"a", "b"}, "s": set()}, {"q": {"a", "wrong"}}, 100)
    assert metrics["candidate_recall"] == .5
    assert metrics["recall_ceiling_macro_f0_5"] == pytest.approx((5/6 + 1)/2)
    assert metrics["reduction_ratio"] == .99


def test_threshold_includes_predict_none():
    pairs = pd.DataFrame({"source1_entity_id": ["q"], "candidate_entity_id": ["a"]})
    best, _ = tune_threshold(["q"], {"q": set()}, pairs, [1.0], [.5, .9])
    assert best["threshold"] > 1
    assert best["macro_f0_5"] == 1


def test_normalisation_preserves_numbers_scripts_and_unknown_country():
    frame = clean_frame(pd.DataFrame([record("S1-a", "Étoile & Services Pvt. Ltd.", "12 Rue des Écoles, 75001", "France"),
                                     record("S2-a", "उदय व्यापार", "21 Rue des Écoles, 75001", "New Country")]))
    assert frame.iloc[0].name_norm == "étoile and services private limited"
    assert frame.iloc[0].name_folded == "etoile and services"
    assert frame.iloc[1].name_norm == "उदय व्यापार"
    assert frame.iloc[1].name_folded == "उदय व्यापार"
    assert frame.iloc[1].country_norm == "new country"
    assert frame.house.tolist() == ["12", "21"]
    assert frame.postal.tolist() == ["75001", "75001"]
    evidence = comparison(frame.iloc[0].to_dict(), frame.iloc[1].to_dict())
    assert evidence["house_conflict"] == 1
    assert evidence["unit_missing"] == 1
    assert evidence["unit_agree"] == 0


def test_typo_retrieval_open_country_and_sparse_work():
    params = load_config(PROJECT / "configs/default.yaml")["retrieval"]["base"]
    targets = clean_frame(pd.DataFrame([record("S2-a", "Orion Tecnologies", "12 Rue Verte", "France"),
                                       record("S3-b", "Completely Unrelated", "94 Route Rouge", "Future Country")]))
    query = clean_frame(pd.DataFrame([record("S1-a", "Orion Technologies", "", "France")]))
    vectors = fit_vectorizers(targets, params)
    pairs, work = RetrievalIndex(targets, vectors, params).retrieve(query)
    assert "S2-a" in set(pairs.candidate_entity_id)
    assert pairs.loc[pairs.candidate_entity_id == "S2-a", "hit_name_char"].iloc[0] == 1
    assert work.postings_visited.sum() <= 3 * params["max_query_terms"] * params["max_posting"]
    assert all("Future Country" not in v.vocabulary_ for _, v in vectors.values())


def test_no_candidates_with_empty_fields_or_no_targets():
    params = load_config(PROJECT / "configs/default.yaml")["retrieval"]["base"]
    target = clean_frame(pd.DataFrame([record("S2-a", "Known")]))
    query = clean_frame(pd.DataFrame([record("S1-a", "")]))
    vectors = fit_vectorizers(target, params)
    assert RetrievalIndex(target, vectors, params).retrieve(query)[0].empty
    assert RetrievalIndex(target.iloc[:0], vectors, params).retrieve(query)[0].empty


def test_fusion_learns_and_applies_cap_without_injecting_truth():
    rows = []
    truth = {}
    for i in range(10):
        a = f"S1-{i}"
        truth[a] = {f"S2-{i}"}
        for b, score in [(f"S2-{i}", .95), (f"S3-{i}", .1)]:
            rows.append(dict(source1_entity_id=a, candidate_entity_id=b, name_char=score,
                             name_word=score, address=score, exact=0, embedding=0))
    pairs = pd.DataFrame(rows)
    model = Fusion.fit(pairs, truth, .001)
    assert np.all(model.weights >= 0)
    chosen = model.select(pairs, 0, 1)
    assert len(chosen) == 10
    assert all(b in truth[a] for a, b in zip(chosen.source1_entity_id, chosen.candidate_entity_id))
    assert model.select(pairs, 1.1, 1).empty


def test_group_split_and_representation_training_isolation(tmp_path):
    create_dataset(tmp_path / "data")
    data = load_dataset(tmp_path / "data", "train")
    anchors = clean_frame(data.anchors)
    # Force two anchors to share a target; the splitter must keep them together.
    shared = next(iter(data.truth["S1-train-0001"]))
    data.truth["S1-train-0002"].add(shared)
    split = split_entities(anchors, data.truth, .2, .2, 42)
    by_id = split.set_index("source1_entity_id").split.to_dict()
    assert by_id["S1-train-0001"] == by_id["S1-train-0002"]
    ownership = target_partitions(data.targets, data.truth, split)
    for a, ids in data.truth.items():
        assert all(ownership[b] == by_id[a] for b in ids)
    assert set(by_id.values()) == {"train", "tune", "audit"}


@pytest.mark.parametrize("fault", ["duplicate_id", "missing_row", "unknown_id", "match_outside_candidates", "wrong_header"])
def test_submission_rejects_contract_violations(tmp_path, fault):
    create_dataset(tmp_path / "data", test_count=3)
    data = load_dataset(tmp_path / "data", "test", labelled=False)
    anchors = data.anchors.entity_id.tolist()
    valid_id = data.targets.entity_id.iloc[0]
    matching, candidate = tmp_path / "matching.tsv", tmp_path / "candidate.tsv"
    export_lists(matching, anchors, {anchors[0]: [valid_id]}, "matched_entity_ids")
    export_lists(candidate, anchors, {anchors[0]: [valid_id]}, "candidate_entity_ids")
    validate_submission(matching, candidate, tmp_path / "data/test")
    text = matching.read_text()
    if fault == "duplicate_id":
        text = text.replace(valid_id, f"{valid_id},{valid_id}")
    elif fault == "missing_row":
        text = "\n".join(text.splitlines()[:-1]) + "\n"
    elif fault == "unknown_id":
        text = text.replace(valid_id, "S2-nonexistent")
    elif fault == "wrong_header":
        text = text.replace("matched_entity_ids", "matches")
    else:
        export_lists(candidate, anchors, {}, "candidate_entity_ids")
    matching.write_text(text)
    with pytest.raises(ValueError):
        validate_submission(matching, candidate, tmp_path / "data/test")
