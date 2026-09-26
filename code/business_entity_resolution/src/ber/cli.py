from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .util import json_save


def main():
    parser = argparse.ArgumentParser(description="Business entity resolution: local retrieval, learned matching and auditable reports")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("run", help="Tune retrieval and three matchers, audit, refit and predict")
    fit.add_argument("--data-dir", type=Path, required=True)
    fit.add_argument("--config", type=Path, required=True)
    fit.add_argument("--run-dir", type=Path, required=True)
    fit.add_argument("--train-only", action="store_true", help="Do not require or predict test data")
    pred = commands.add_parser("predict", help="Regenerate test outputs from saved artifacts, without retraining")
    pred.add_argument("--data-dir", type=Path, required=True)
    pred.add_argument("--artifacts", type=Path, required=True)
    pred.add_argument("--output-dir", type=Path, required=True)
    pred.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"])
    valid = commands.add_parser("validate", help="Check submission IDs, coverage and candidate containment")
    valid.add_argument("--matching", type=Path, required=True)
    valid.add_argument("--candidate", type=Path, required=True)
    valid.add_argument("--test-dir", type=Path, required=True)
    download = commands.add_parser("download-models", help="Explicitly download pretrained weights; no business lookup")
    download.add_argument("--directory", type=Path, default=Path("models"))
    download.add_argument("--revision", default=None, help="Hub commit SHA; omitted means resolve current SHA and record it")
    fixture = commands.add_parser("make-synthetic", help="Create invented fixtures, not competition data")
    fixture.add_argument("--directory", type=Path, required=True)
    fixture.add_argument("--train-count", type=int, default=60)
    fixture.add_argument("--test-count", type=int, default=15)
    smoke = commands.add_parser("smoke", help="Exercise all five routes and all three models with tiny local transformers")
    smoke.add_argument("--directory", type=Path, required=True)
    smoke.add_argument("--config", type=Path, required=True, help="Path to configs/default.yaml")
    pack = commands.add_parser("package", help="Build a self-contained challenge submission ZIP")
    pack.add_argument("--run-dir", type=Path, required=True)
    pack.add_argument("--data-dir", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # External model repositories are accessed only through this explicit command.
    if args.command == "download-models":
        from huggingface_hub import HfApi, snapshot_download
        from .util import sha256
        model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        target = args.directory / "multilingual-minilm"
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(f"Model directory already exists: {target}")
        info = HfApi().model_info(model_id, revision=args.revision)
        license_ = info.card_data.get("license") if info.card_data else None
        if license_ not in {"mit", "apache-2.0"}:
            raise ValueError(f"Model licence is not allowed: {license_}")
        snapshot_download(model_id, revision=info.sha, local_dir=target,
                          allow_patterns=["config.json", "tokenizer*", "special_tokens_map.json", "sentencepiece*", "vocab.txt",
                                          "merges.txt", "*.safetensors", "pytorch_model.bin", "README.md", "LICENSE*"])
        hashes = {str(p.relative_to(target)): sha256(p) for p in target.rglob("*") if p.is_file() and ".cache" not in p.parts}
        json_save(target / "provenance.json", {"model_id": model_id, "revision": info.sha,
                                               "license": license_, "files_sha256": hashes})
        print(f"Downloaded pretrained weights only to {target}; pinned revision {info.sha}")
    elif args.command == "make-synthetic":
        from .synthetic import create_dataset
        create_dataset(args.directory, args.train_count, args.test_count)
    elif args.command == "validate":
        from .data import validate_submission
        print(json.dumps(validate_submission(args.matching, args.candidate, args.test_dir), indent=2))
    elif args.command == "predict":
        from .pipeline import predict
        predict(args.data_dir, args.artifacts, args.output_dir, args.device)
    elif args.command == "package":
        from .packaging import package
        package(args.run_dir, args.data_dir, args.output)
    else:
        from .config import load_config
        from .pipeline import run
        config = load_config(args.config)
        if args.command == "smoke":
            from .synthetic import create_dataset, create_tiny_transformer, smoke_config
            if args.directory.exists() and any(args.directory.iterdir()):
                raise FileExistsError(f"Smoke directory is not empty: {args.directory}")
            create_dataset(args.directory / "dataset")
            create_tiny_transformer(args.directory / "tiny_model")
            config = smoke_config(config, args.directory / "tiny_model")
            run(args.directory / "dataset", config, args.directory / "run")
        elif args.command == "run":
            for key in ("embedding", "cross_encoder"):
                config[key]["model_path"] = str(Path(config[key]["model_path"]).resolve())
            run(args.data_dir, config, args.run_dir, do_predict=not args.train_only)
