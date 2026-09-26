"""Independent stdlib-only validation; the official validator can also be used."""
import argparse
import csv
from pathlib import Path


def validate(matching, candidate, test_dir):
    def source_ids(source):
        path = Path(test_dir) / f"test_source{source}.tsv"
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file, delimiter="\t")
            if "entity_id" not in (reader.fieldnames or []):
                raise ValueError(f"{path}: entity_id column missing")
            ids = [r["entity_id"] for r in reader]
        if len(ids) != len(set(ids)) or any(not x.startswith(f"S{source}-") for x in ids):
            raise ValueError(f"{path}: duplicate or incorrect IDs")
        return set(ids)

    anchors = source_ids(1)
    valid = source_ids(2) | source_ids(3)
    output = []
    for path, column in [(matching, "matched_entity_ids"), (candidate, "candidate_entity_ids")]:
        result = {}
        with Path(path).open(newline="", encoding="utf-8") as file:
            reader = csv.reader(file, delimiter="\t", strict=True)
            if next(reader, None) != ["source1_entity_id", column]:
                raise ValueError(f"{path}: incorrect header or separator")
            for line, row in enumerate(reader, 2):
                if len(row) != 2:
                    raise ValueError(f"{path}:{line}: expected exactly two fields")
                a, text = row
                ids = text.split(",") if text else []
                if a not in anchors or a in result:
                    raise ValueError(f"{path}:{line}: unknown or duplicate anchor")
                if len(ids) != len(set(ids)) or any(x not in valid for x in ids):
                    raise ValueError(f"{path}:{line}: duplicate or unknown candidate IDs")
                result[a] = set(ids)
        if set(result) != anchors:
            raise ValueError(f"{path}: missing Source 1 rows")
        output.append(result)
    if any(not output[0][a] <= output[1][a] for a in anchors):
        raise ValueError("A final match was absent from candidate_pairs.tsv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--test-dir", required=True)
    args = parser.parse_args()
    try:
        validate(args.matching, args.candidate, args.test_dir)
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(1, f"FAIL: {error}\n")
    print("PASS")
