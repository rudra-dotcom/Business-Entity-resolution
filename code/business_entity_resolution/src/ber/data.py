from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


@dataclass
class Dataset:
    anchors: pd.DataFrame
    targets: pd.DataFrame
    truth: dict[str, set[str]] | None


def read_source(path, prefix):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Add the challenge TSV files; see README.md.")
    frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    missing = set(SOURCE_COLUMNS) - set(frame)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}; files must be tab-separated")
    frame = frame[SOURCE_COLUMNS].copy()
    ids = frame.entity_id
    if ids.duplicated().any() or not ids.str.startswith(prefix).all():
        raise ValueError(f"{path}: duplicate IDs or IDs without the {prefix} prefix")
    if ids.str.contains(r"[\s,]", regex=True).any():
        raise ValueError(f"{path}: IDs must not contain whitespace or commas")
    return frame


def parse_id_list(value):
    if not value:
        return []
    ids = value.split(",")
    if any(not x or x != x.strip() for x in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"Malformed or duplicate ID list: {value!r}")
    return ids


def load_dataset(root, split, labelled=True):
    root = Path(root) / split
    sources = [read_source(root / f"{split}_source{i}.tsv", f"S{i}-") for i in (1, 2, 3)]
    targets = pd.concat(sources[1:], ignore_index=True)
    truth = None
    if labelled:
        file = root / f"{split}_ground_truth.tsv"
        frame = pd.read_csv(file, sep="\t", dtype=str, keep_default_na=False)
        if list(frame.columns) != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError(f"{file}: incorrect ground-truth columns")
        if frame.source1_entity_id.duplicated().any():
            raise ValueError("Duplicate Source 1 IDs in ground truth")
        truth = {row.source1_entity_id: set(parse_id_list(row.matched_entity_ids))
                 for row in frame.itertuples(index=False)}
        if set(truth) != set(sources[0].entity_id):
            raise ValueError("Ground truth must contain exactly one row per training Source 1 record")
        valid = set(targets.entity_id)
        if any(ids - valid for ids in truth.values()):
            raise ValueError("Ground truth references IDs absent from training Sources 2/3")
    return Dataset(sources[0], targets, truth)


def export_lists(path, anchors, lists, column):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", column])
        for anchor in anchors:
            writer.writerow([anchor, ",".join(sorted(set(lists.get(anchor, []))))])


def validate_submission(matching, candidate, test_dir):
    test_dir = Path(test_dir)
    anchors = set(read_source(test_dir / "test_source1.tsv", "S1-").entity_id)
    valid = set(read_source(test_dir / "test_source2.tsv", "S2-").entity_id)
    valid |= set(read_source(test_dir / "test_source3.tsv", "S3-").entity_id)
    parsed = []
    for path, column in [(matching, "matched_entity_ids"), (candidate, "candidate_entity_ids")]:
        with Path(path).open(newline="", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter="\t")
            if next(reader, None) != ["source1_entity_id", column]:
                raise ValueError(f"{path}: incorrect header or separator")
            result = {}
            for line, row in enumerate(reader, 2):
                if len(row) != 2:
                    raise ValueError(f"{path}:{line}: expected exactly two tab-separated fields")
                anchor, value = row
                if anchor in result or anchor not in anchors:
                    raise ValueError(f"{path}:{line}: duplicate or unknown Source 1 ID {anchor}")
                ids = set(parse_id_list(value))
                if ids - valid:
                    raise ValueError(f"{path}:{line}: unknown Source 2/3 IDs {sorted(ids - valid)}")
                result[anchor] = ids
            if set(result) != anchors:
                raise ValueError(f"{path}: missing Source 1 rows")
            parsed.append(result)
    for anchor in anchors:
        if not parsed[0][anchor] <= parsed[1][anchor]:
            raise ValueError(f"{anchor}: final matches must be a subset of candidates")
    return {"status": "PASS", "source1_rows": len(anchors),
            "candidate_pairs": sum(map(len, parsed[1].values())),
            "matched_pairs": sum(map(len, parsed[0].values()))}
