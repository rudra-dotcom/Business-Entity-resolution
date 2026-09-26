"""Invented fixtures for plumbing tests only. Never claim their scores as accuracy."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import SOURCE_COLUMNS
from .util import json_save, write_tsv


def create_dataset(directory, train_count=60, test_count=15):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Synthetic dataset directory is not empty: {directory}")
    for split, count in [("train", train_count), ("test", test_count)]:
        rows = {1: [], 2: [], 3: []}
        truth = []
        for i in range(count):
            country = (["India", "US"] if split == "train" else ["India", "US", "France"])[i % (2 if split == "train" else 3)]
            postal = "411001" if country == "India" else ("75001" if country == "France" else "10001")
            token = f"{split}venture{i:04d}"
            name = f"{token} Technologies Private Limited"
            street = "Rue des Écoles" if country == "France" else "Maple Road"
            address = f"{10+i} {street}, District {i % 4}, {postal}"
            a = f"S1-{split}-{i:04d}"

            def record(id_, name_, address_):
                return dict(zip(SOURCE_COLUMNS, [id_, name_, address_, country]))

            rows[1].append(record(a, name, address))
            matches = []
            # Every fourth entity is a singleton; different countries get singletons.
            if i % 4 != 0:
                b, c = f"S2-{split}-{i:04d}", f"S3-{split}-{i:04d}"
                rows[2].append(record(b, f"{token} Tecnologies Pvt. Ltd.", address.replace("Road", "Rd")))
                rows[3].append(record(c, f"Technologies {token}", address if i % 5 else ""))
                matches += [b, c]
                if i % 7 == 0:
                    extra = f"S2-{split}-extra-{i:04d}"
                    rows[2].append(record(extra, f"{token} Technologies", address))
                    matches.append(extra)
            # Hard negative: same distinctive token, conflicting activity and address.
            rows[2].append(record(f"S2-{split}-noise-{i:04d}", f"{token} Foods", f"{210+i} Lake Road, {postal}"))
            rows[3].append(record(f"S3-{split}-noise-{i:04d}", f"Neighbour{i:04d} Technologies", address))
            truth.append({"source1_entity_id": a, "matched_entity_ids": ",".join(matches)})
        for source, records in rows.items():
            write_tsv(directory / split / f"{split}_source{source}.tsv", pd.DataFrame(records, columns=SOURCE_COLUMNS))
        if split == "train":
            write_tsv(directory / split / "train_ground_truth.tsv", truth)
        else:
            write_tsv(directory / "fixture_only" / "test_truth_DO_NOT_USE_FOR_TRAINING.tsv", truth)
    json_save(directory / "SYNTHETIC_ONLY.json", {"purpose": "pipeline verification, not model-performance evidence"})


def create_tiny_transformer(directory):
    from transformers import BertConfig, BertModel, BertTokenizerFast
    import torch
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Tiny model directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "name", "address", "country", ":", "|",
             "technologies", "tecnologies", "private", "limited", "pvt", "ltd", "foods", "road", "rd",
             "maple", "lake", "district", "india", "us", "france", "rue", "des", "ecoles"]
    words += [str(i) for i in range(300)] + ["411001", "75001", "10001"]
    words += [f"{s}venture{i:04d}" for s in ["train", "test"] for i in range(100)]
    (directory / "vocab.txt").write_text("\n".join(words) + "\n")
    tokenizer = BertTokenizerFast(vocab_file=str(directory / "vocab.txt"), do_lower_case=True)
    tokenizer.save_pretrained(directory)
    torch.manual_seed(42)
    model = BertModel(BertConfig(vocab_size=len(words), hidden_size=32, num_hidden_layers=1,
                                 num_attention_heads=4, intermediate_size=48, max_position_embeddings=256))
    model.save_pretrained(directory, safe_serialization=True)
    json_save(directory / "provenance.json", {"model_id": "local-random-tiny-bert-fixture", "revision": "seed-42",
                                              "license": "MIT", "synthetic_only": True,
                                              "purpose": "exercise real transformer code without downloading pretrained weights"})


def smoke_config(base, model_path):
    import copy
    config = copy.deepcopy(base)
    config["query_batch_size"] = 5
    config["retrieval"].update(max_trials=2, candidate_caps=[5, 10], thresholds=[0, 0.1],
                               target_recall=0.9, target_macro_recall=0.9)
    config["retrieval"]["grid"] = {"route_k": [8, 12], "fusion_l2": [0.001]}
    config["retrieval"]["base"].update(max_features=5000, max_posting=500, hnsw_m=8, ef_construction=30, ef_search=20)
    for key in ["embedding", "cross_encoder"]:
        config[key].update(model_path=str(Path(model_path).resolve()), device="cpu", torch_threads=1,
                            max_length=64, batch_size=16, gradient_checkpointing=False)
    config["embedding"].update(enabled=True, max_trials=1, negatives_per_anchor=3)
    config["embedding"]["grid"] = {"epochs": [1], "learning_rate": [0.001], "batch_size": [16]}
    config["matching"].update(models=["logistic", "boosted_trees", "cross_encoder"], max_negatives_per_anchor=8,
                              thresholds=[0.1, 0.3, 0.5, 0.7, 0.9], importance_repeats=1)
    config["matching"]["logistic"] = {"max_trials": 2, "grid": {"C": [0.1, 1.0], "class_weight": [None]}}
    config["matching"]["boosted_trees"] = {"max_trials": 2, "grid": {"learning_rate": [0.1], "max_iter": [20, 30],
        "max_leaf_nodes": [7], "min_samples_leaf": [5], "l2_regularization": [1.0]}}
    config["matching"]["cross_encoder"] = {"max_trials": 2, "grid": {"learning_rate": [0.001, 0.002], "epochs": [1],
        "batch_size": [16], "gradient_accumulation": [2]}}
    return config
