from __future__ import annotations

import hashlib
import itertools
import json
import logging
import platform
import random
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("ber")


def json_save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")


def write_tsv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    frame.to_csv(path, sep="\t", index=False, float_format="%.9g")


def grid(parameters, limit=None, seed=42):
    keys = list(parameters)
    combinations = list(itertools.product(*(parameters[k] for k in keys)))
    if limit and len(combinations) > limit:
        # Include the first (baseline) combination, then a reproducible random search.
        rng = random.Random(seed)
        combinations = [combinations[0]] + rng.sample(combinations[1:], limit - 1)
    return [dict(zip(keys, values)) for values in combinations]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def environment():
    names = ["numpy", "scipy", "pandas", "scikit-learn", "torch", "transformers",
             "faiss-cpu", "rapidfuzz", "PyYAML", "joblib", "tokenizers", "safetensors"]
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]
