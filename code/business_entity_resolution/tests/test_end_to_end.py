import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ber.config import load_config
from ber.data import validate_submission
from ber.packaging import package
from ber.pipeline import predict, run
from ber.synthetic import create_dataset, create_tiny_transformer, smoke_config


def test_all_models_training_reload_export_and_package(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("faiss")
    pytest.importorskip("transformers")
    project = Path(__file__).resolve().parents[1]
    dataset, base, run_dir = tmp_path / "dataset", tmp_path / "tiny", tmp_path / "run"
    create_dataset(dataset, train_count=40, test_count=6)
    create_tiny_transformer(base)
    config = smoke_config(load_config(project / "configs/default.yaml"), base)
    run(dataset, config, run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    summary = pd.read_csv(run_dir / "reports/model_comparison.tsv", sep="\t")
    assert set(summary.model) == {"logistic", "boosted_trees", "cross_encoder"}
    assert set(summary.split) == {"tune", "audit"}
    assert summary.macro_f0_5.between(0, 1).all()
    trials = pd.read_csv(run_dir / "reports/model_trials.tsv", sep="\t")
    assert trials.groupby("model").size().min() == 2
    assert len(pd.read_csv(run_dir / "reports/retrieval_trials.tsv", sep="\t")) > 1
    from ber.data import load_dataset
    from ber.split import target_partitions
    data = load_dataset(dataset, "train")
    splits = pd.read_csv(run_dir / "reports/splits.tsv", sep="\t", keep_default_na=False)
    ownership = target_partitions(data.targets, data.truth, splits)
    training = pd.read_csv(run_dir / "reports/training_pairs.tsv.gz", sep="\t")
    assert all(ownership[t] == "train" for t in training.candidate_entity_id)
    output = run_dir / "output"
    candidates = pd.read_csv(output / "candidate_pairs.tsv", sep="\t", keep_default_na=False)
    candidate_pairs = {(row.source1_entity_id, b) for row in candidates.itertuples()
                       for b in row.candidate_entity_ids.split(",") if b}
    assert len(candidates) == 6  # Includes all France queries and singletons.
    for kind in config["matching"]["models"]:
        directory = output / "models" / kind
        validate_submission(directory / "matching_results.tsv", directory / "candidate_pairs.tsv", dataset / "test")
        scores = pd.read_csv(directory / "pair_scores.tsv", sep="\t")
        assert set(zip(scores.source1_entity_id, scores.candidate_entity_id)) == candidate_pairs
        assert len(scores) == len(candidate_pairs)
        assert scores.match_score.between(0, 1).all()
    # Actual encoder weights, not just a mocked API, must have been updated.
    from safetensors.torch import load_file
    original = load_file(base / "model.safetensors")
    trained = load_file(run_dir / "artifacts/embedding/model.safetensors")
    assert any(not np.array_equal(value.numpy(), trained[key].numpy()) for key, value in original.items())
    cross_trained = load_file(run_dir / "artifacts/cross_encoder/transformer/model.safetensors")
    assert any(not np.array_equal(value.numpy(), cross_trained[f"bert.{key}"].numpy())
               for key, value in original.items() if f"bert.{key}" in cross_trained)
    # Inference replay must produce identical candidate and final submission bytes.
    replay = tmp_path / "replay"
    predict(dataset, run_dir / "artifacts", replay, device="cpu")
    for name in ["candidate_pairs.tsv", "matching_results.tsv"]:
        assert (output / name).read_bytes() == (replay / name).read_bytes()
    result = subprocess.run([sys.executable, str(project / "utils/validate_submission.py"),
                             "--matching", str(output / "matching_results.tsv"), "--candidate", str(output / "candidate_pairs.tsv"),
                             "--test-dir", str(dataset / "test")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    archive = tmp_path / "team_submission.zip"
    package(run_dir, dataset, archive)
    with zipfile.ZipFile(archive) as file:
        names = set(file.namelist())
        assert {"output/matching_results.tsv", "output/candidate_pairs.tsv", "Documentation_template.md", "REPRODUCE.md"} <= names
        assert "code/business_entity_resolution/src/ber/pipeline.py" in names
        assert any("base_models/model_0/model.safetensors" in name for name in names)
        file.extractall(tmp_path / "unpacked")
    unpacked = tmp_path / "unpacked"
    base.rename(tmp_path / "original_base_unavailable")
    isolated_environment = {**os.environ, "PYTHONPATH": str(unpacked / "code/business_entity_resolution/src")}
    subprocess.run([sys.executable, "-m", "ber", "predict", "--data-dir", str(dataset), "--artifacts",
                    str(unpacked / "code/business_entity_resolution/artifacts"), "--output-dir", str(tmp_path / "packaged_replay"),
                    "--device", "cpu"], check=True, capture_output=True, text=True, cwd=unpacked, env=isolated_environment)
    assert (output / "matching_results.tsv").read_bytes() == (tmp_path / "packaged_replay/matching_results.tsv").read_bytes()
    # Complete retraining must also work using only the archived code/checkpoints and input data.
    subprocess.run([sys.executable, "-m", "ber", "run", "--data-dir", str(dataset), "--config",
                    "code/business_entity_resolution/reproduce.yaml", "--run-dir", str(tmp_path / "packaged_training")],
                   check=True, capture_output=True, text=True, cwd=unpacked, env=isolated_environment)
    assert (output / "matching_results.tsv").read_bytes() == (tmp_path / "packaged_training/output/matching_results.tsv").read_bytes()
    subprocess.run([sys.executable, "-m", "ber", "package", "--data-dir", str(dataset), "--run-dir",
                    str(tmp_path / "packaged_training"), "--output", str(tmp_path / "repackaged.zip")],
                   check=True, capture_output=True, text=True, cwd=unpacked, env=isolated_environment)
    assert (tmp_path / "repackaged.zip").is_file()
