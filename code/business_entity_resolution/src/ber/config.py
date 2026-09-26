from __future__ import annotations

import copy
from pathlib import Path

import yaml


def load_config(path):
    path = Path(path)
    obj = yaml.safe_load(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError("Config must be a YAML mapping")
    if "extends" in obj:
        parent = load_config(path.parent / obj.pop("extends"))
        obj = merge(parent, obj)
    validate_config(obj)
    return obj


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def validate_config(config):
    required = {"seed", "final_refit", "query_batch_size", "split", "retrieval", "embedding", "cross_encoder", "matching"}
    if required - config.keys():
        raise ValueError(f"Missing config sections: {sorted(required - config.keys())}")
    if config["query_batch_size"] < 1:
        raise ValueError("query_batch_size must be positive")
    for name in config["matching"]["models"]:
        if name not in {"logistic", "boosted_trees", "cross_encoder"}:
            raise ValueError(f"Unknown matching model: {name}")
    if not config["matching"]["models"]:
        raise ValueError("At least one matching model is required")
    if len(config["matching"]["models"]) != len(set(config["matching"]["models"])):
        raise ValueError("Matching model names must be unique")
    for section in [config["retrieval"], config["embedding"]] + [config["matching"][m] for m in config["matching"]["models"]]:
        if section["max_trials"] < 1 or not all(isinstance(v, list) and len(v) for v in section["grid"].values()):
            raise ValueError("Each search needs positive max_trials and nonempty parameter lists")
    if not config["retrieval"]["candidate_caps"] or min(config["retrieval"]["candidate_caps"]) < 1:
        raise ValueError("candidate_caps must be positive")
    base = config["retrieval"]["base"]
    if set(config["retrieval"]["grid"]) - set(base):
        raise ValueError("Every retrieval grid key must also be declared in retrieval.base")
    for name in ["route_k", "max_features", "max_posting", "max_query_terms", "max_exact_block", "hnsw_m", "ef_construction", "ef_search"]:
        if any(value < 1 for value in [base[name]] + config["retrieval"]["grid"].get(name, [])):
            raise ValueError(f"{name} must be positive; unbounded posting scans are not supported")
    if "cross_encoder" in config["matching"]["models"] and min(config["matching"]["cross_encoder"]["grid"]["epochs"]) < 1:
        raise ValueError("Cross-encoder trials must train the classification head for at least one epoch")
