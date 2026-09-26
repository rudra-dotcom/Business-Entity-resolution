from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import yaml

from .data import validate_submission


def package(run_dir, data_dir, destination):
    run_dir, destination = Path(run_dir), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing archive: {destination}")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Only a completed run can be packaged")
    validate_submission(run_dir / "output" / "matching_results.tsv", run_dir / "output" / "candidate_pairs.tsv", Path(data_dir) / "test")
    project = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="ber-package-") as directory:
        root = Path(directory)
        code = root / "code" / "business_entity_resolution"
        # A relocated submission is itself a runnable project. Do not recursively
        # copy its old artifacts/data when packaging a newly reproduced run.
        shutil.copytree(project, code, ignore=shutil.ignore_patterns(
            "__pycache__", ".pytest_cache", "*.egg-info", ".venv", ".git",
            "artifacts", "base_models", "models", "dataset", "runs", "output", "*.zip"))
        shutil.copytree(run_dir / "artifacts", code / "artifacts")
        output = root / "output"
        output.mkdir()
        for name in ["matching_results.tsv", "candidate_pairs.tsv"]:
            shutil.copy2(run_dir / "output" / name, output / name)
        shutil.copytree(run_dir / "reports", root / "reports")
        shutil.copy2(run_dir / "manifest.json", root / "run_manifest.json")
        bundle = json.loads((run_dir / "artifacts" / "bundle.json").read_text())
        config = bundle["config"]
        paths = {}
        for role in ["embedding", "cross_encoder"]:
            needed = config["embedding"]["enabled"] if role == "embedding" else "cross_encoder" in config["matching"]["models"]
            if not needed:
                continue
            source = Path(config[role]["model_path"]).resolve()
            if not source.is_dir():
                raise FileNotFoundError(f"Initial weights needed for self-contained retraining are missing: {source}")
            if str(source) not in paths:
                relative = Path("code/business_entity_resolution/base_models") / f"model_{len(paths)}"
                shutil.copytree(source, root / relative, ignore=shutil.ignore_patterns(".cache"))
                paths[str(source)] = str(relative)
            config[role]["model_path"] = paths[str(source)]
        (code / "reproduce.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        template = (project / "Documentation_template.md").read_text()
        (root / "Documentation_template.md").write_text(template + "\n\n## This run\n\n"
            + f"Selected model: `{bundle['selected_model']}` (chosen on tuning data).\n\n"
            + "Exact settings: `code/business_entity_resolution/reproduce.yaml`. Measured results: `reports/model_comparison.tsv`; candidate metrics: `reports/candidate_summary.tsv`.\n\n"
            + "Run provenance and input hashes are in `run_manifest.json`. If its inputs are synthetic fixtures, these scores are plumbing checks only.\n")
        (root / "REPRODUCE.md").write_text("""# Reproduce this package

Place the official `dataset/train/` and `dataset/test/` files beside this file.
Use Python 3.12 in a new virtual environment. From this archive's root:

```sh
python -m pip install -r code/business_entity_resolution/requirements.txt
python -m pip install -e code/business_entity_resolution --no-deps
ber predict --data-dir dataset --artifacts code/business_entity_resolution/artifacts --output-dir regenerated_output
```

This replays the saved models without downloads or retraining. All final matcher input pairs are emitted in `candidate_pairs.tsv`.
To repeat the complete search and training instead, using the included initial neural weights:

```sh
ber run --data-dir dataset --config code/business_entity_resolution/reproduce.yaml --run-dir reproduced_run
```

GPU/CPU floating-point differences can change scores near thresholds. Check pinned dependencies, recorded hardware, input hashes and the original seed.
""")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root))
    print(f"Created {destination}")
