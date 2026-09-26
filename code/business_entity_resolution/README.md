# Business entity resolution pipeline

This project takes the challenge's seven TSV files and produces both required submission files.
It trains and compares logistic regression, histogram gradient-boosted trees, and a transformer
cross-encoder independently. An ensemble is deliberately not selected: the reports preserve
scores and disagreements so a later experiment can determine whether combining models helps.

The implementation has been exercised with invented records and tiny, randomly initialised local
transformers. That verifies execution, training, saved-model replay and formatting. It does **not**
establish accuracy, best hyperparameters, training time or memory requirements on the real dataset.
The supplied pretrained checkpoint is a recommended starting representation, not a verified winner.

## Installation and files

Use Python 3.12 (supported by these dependency pins: 3.11–3.12). Commands below assume the repository root is the current directory.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/business_entity_resolution/requirements.txt
python -m pip install -e code/business_entity_resolution --no-deps
```

The exact direct dependency versions are pinned. Every run also records installed versions, Python,
platform, configuration, seed and SHA-256 hashes of the input TSVs. PyTorch's pinned version supports
CPU and Apple Silicon; an appropriate CUDA installation may be needed on a GPU machine.

Expected data layout:

```text
dataset/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv
  test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv
```

Input fields are loaded as strings with `sep="\t"`; empty strings and leading zeroes are preserved.
IDs and complete ground-truth coverage are validated. No dataset or real business information is supplied here.

Download the one shared multilingual pretrained encoder explicitly:

```sh
ber download-models --directory models
```

This downloads `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, records its exact Hub
commit and file hashes, and verifies its declared Apache-2.0 licence. Use `--revision COMMIT_SHA`
to request a particular revision. Both neural models start from this checkpoint; they become separate
models when trained. The cross-encoder has a newly initialised one-logit classification head, trained
on this challenge's labels. It is not used as an untrained off-the-shelf identity classifier.

Training and inference use `local_files_only=True` and `trust_remote_code=False`. They make no
business-identity, geocoding, registry, web-data or hosted-embedding requests. A local replacement model
must include a `provenance.json` with `model_id`, `revision` and `license` (`mit` or `apache-2.0`), use a
compatible Hugging Face AutoModel/tokenizer, and support masked mean pooling for the bi-encoder.
Individual and combined loaded neural parameter counts are checked against eight billion.

## Run the full experiment

```sh
ber run --data-dir dataset --config code/business_entity_resolution/configs/default.yaml --run-dir runs/experiment_001
```

Use `--train-only` if the test files are not yet available. Each run needs an empty/new directory;
existing experiments are never overwritten. Completed trial reports are written incrementally. A failed
run preserves its logs and manifest error; restart into a new directory after fixing the problem.
There is no automatic training resume from partial optimizer state.

The default search is intentionally bounded. It samples 8 retrieval configurations from the listed
Cartesian product, compares 2 embedding variants, evaluates candidate thresholds/caps, then tests
6 logistic, 12 boosted-tree and 4 cross-encoder configurations. `max_trials` caps each search and the
first configuration is always included. Increase the limits to explore more combinations; it is a
deterministic random search, not a claim to exhaust all possible hyperparameters or find a global optimum.
Neural trials can be expensive: use a GPU where possible and inspect logs before increasing the search.

An explicit smaller baseline, excluding neural routes/models, is also available:

```sh
ber run --data-dir dataset --config code/business_entity_resolution/configs/quick.yaml --run-dir runs/classical_001
```

`quick.yaml` runs only four retrieval routes and two matchers. Use `default.yaml` for the requested
complete three-model comparison. There is no silent fallback when a neural dependency or model is missing.

## What the pipeline does

1. **Clean records locally.** Unicode normalisation, case folding, punctuation/space handling, legal
   abbreviation expansion, an additional suffix-stripped name, and an accent-folded lookup representation.
   Original fields remain available to the neural models and in the cleaned reports. Street abbreviation
   handling is conservative (`Rd`, `Ave`, etc.); ambiguous `St` is not globally replaced. Postal/leading
   house/unit extraction yields evidence rather than a rule forcing a match. Missing is different from conflicting.
2. **Make train/tune/audit splits by Source 1 entity.** Anchors sharing a labelled target stay together.
   Country/singleton strata are used where possible. Known tune/audit-linked targets are excluded from
   train-only TF-IDF fitting, embedding fine-tuning, and supervised negative examples for fusion/matching.
   All targets remain searchable distractors during evaluation; held-out targets may be indexed/scored
   but do not supply training pairs.
3. **Build and evaluate retrieval indexes.** Character TF-IDF on names; word TF-IDF on names;
   word or character TF-IDF on addresses; exact composite keys; optional learned dense embeddings with HNSW.
4. **Learn candidate fusion.** A nonnegative regularised logistic function learns five route coefficients
   and an intercept from train-pair labels. Queries receive equal aggregate training weight. Missing route
   evidence has score zero; a route's positive cosine is recomputed for records recovered by other routes,
   using only the bounded union. Label information is never injected into tune, audit or test candidates.
5. **Tune shortlist policy.** Thresholds and caps are selected on tuning entities. Configurations must
   meet both micro and positive-entity macro candidate-recall targets. Among feasible configurations,
   select the smallest mean shortlist, breaking ties by recall. If none meet the targets, select best
   recall and emit a warning. Review the reports; a successful run does not mean the recall target was met.
6. **Train three separate matchers.** Logistic regression and boosted trees use pair features. A
   cross-encoder jointly reads the two raw records and learns a binary identity score. It does not consume
   the handcrafted feature matrix. All three evaluate the same final candidate pairs.
7. **Select hyperparameters and acceptance thresholds on tune only.** Score the full truth, including
   retrieval misses. Select the model to submit on tune only; audit results cannot silently change that selection.
8. **Evaluate the chosen variant of each model on audit.** Record per-pair and per-entity diagnostics,
   country results, singletons, false positives, missed matches and cross-model disagreements.
9. **Optionally refit on all labelled data, then predict test.** Enabled by default. Hyperparameters,
   shortlist policy and matching thresholds remain fixed, while TF-IDF statistics, neural weights,
   fusion coefficients and matcher parameters are refit. Audit scores describe the pre-refit models;
   all-data refitting can shift score distributions, so inspect threshold robustness before relying on it.

The audit is a held-out estimate for this experiment. Repeatedly using it to choose new models or
ensemble weights makes it a development set. A later ensemble needs its own grouped out-of-fold
training predictions and fresh evaluation; simply fitting a blender on these audit labels would leak.

## Indexing and cost

Text indexes are weighted posting lists: `term -> (target_row, TF-IDF weight)` entries. For each query,
only a configured number of its strongest terms is visited, and terms whose posting lists exceed
`max_posting` are skipped. Route scores are accumulated for touched records, followed by top-k retrieval.
With fixed caps the posting visits per route are bounded by `max_query_terms * max_posting`.
No dense all-Source-1 by all-target similarity matrix is built. The reported search may be approximate
because skipped common terms can change the ranking. These misses are explicitly measured on validation.

Exact keys include `(country, postal, house)`, `(country, normalised_name)` and
`(country, normalised_address)` when sufficient fields exist. Oversized exact blocks are skipped and
counted, not silently truncated by file order. Other routes can recover their records. HNSW stores a
graph over normalised dense vectors; `ef_search`, graph degree and construction settings control its
speed/memory/recall trade-off. No logarithmic worst-case lookup guarantee is claimed.

Country is an open string. There is **no hard country filter** on text or embedding routes and no
US/India-only encoding. Country-specific parsing hints cover common numeric postal forms with a
fallback for unknown labels. Generic parsing can be wrong; conflicts are model features, not forced rejections.
France and all other test Source 1 records are always represented in the output.

Index construction and dataset reading are linear in input size apart from index-building overhead.
Target vectors/indexes live in RAM. Training candidate tables and trial features also live in RAM;
this is a single-machine challenge implementation, not a distributed billion-record deployment.
Test queries are batched. Report posting visits, touched records, candidate counts and runtime rather
than treating a small exported candidate file as proof that retrieval was cheap.

## What learns weights, and what is tuned

| Component | Learned from labels | Tuned settings |
|---|---|---|
| TF-IDF | No; computes document-frequency weights from training text | n-gram lengths, word/character address representation, vocabulary size |
| Inverted indexes / exact lookup | No learned model parameters | posting cap, query-term cap, route top-k, minimum similarity, exact block cap |
| Bi-encoder | Optional transformer fine-tuning using positive pairs and lexical hard negatives | epochs, learning rate, batch size, accumulation, regularisation, cosine centre/temperature |
| HNSW | No supervised training | graph degree, construction effort, search effort |
| Candidate fusion | Five nonnegative route coefficients plus an intercept | L2 regularisation, shortlist threshold, shortlist cap |
| Logistic matcher | Scaled-feature coefficients and intercept | regularisation `C`, class weighting, acceptance threshold |
| Boosted-tree matcher | Tree splits and leaf predictions, including feature interactions | learning rate, iterations, leaf count, minimum leaf size, L2, acceptance threshold |
| Cross-encoder matcher | Transformer and pair-classification-head parameters | epochs, learning rate, batch size, gradient accumulation, weight decay, positive class weight, acceptance threshold |

All grid-valued settings can be changed in YAML. Constants in `retrieval.base`, neural sections and
sampling settings can also be varied across runs. Matcher training uses the broader raw training
retrieval pool, keeping all retrieved positives and a reproducible mixture of hard and random
negatives per anchor. This prevents an effective shortlist filter from deleting all negative training
examples. Tune/audit/test inference always uses only the final filtered candidates. Singleton anchors provide negatives.
Only embedding training can add positive pairs missed by lexical bootstrapping; these use train labels
and never become artificially injected validation/test candidates. Transformer losses are logged per epoch.

Pair features include separate name/address TF-IDF scores, Levenshtein, Jaro–Winkler, token overlap,
containment, acronym evidence, full/core/accent-folded name agreement, length ratio, address agreement,
postal/house/unit/country agreement/conflict/missing flags, and a name–address interaction. Source IDs
are identifiers only and never predictive features. No fixed “name always matters more” rule is imposed.

Logistic coefficients and scalers are exported. Boosted trees do not have one universal weight per feature;
tune-set permutation importance reports the drop in macro F0.5 after shuffling each feature, at a fixed
threshold. Correlated features can make this importance misleading. Neural weights are saved in
Safetensors, not millions of TSV rows. All model outputs are scores; do not assume perfect probability
calibration before a future probability-averaging ensemble.

## Evaluation and reports

For each Source 1 entity, `F0.5 = 5*TP/(5*TP + 4*FP + FN)`, with empty truth + empty prediction
explicitly scoring 1. Average over **entities**, including singletons. This is not sklearn's macro
average over binary pair classes. Threshold search includes predict-none. Candidate metrics compare
against the full ground truth, even when the matcher never saw a missed pair.

Everything below is under the run directory:

| File | What to inspect |
|---|---|
| `config.json`, `manifest.json`, `run.log` | Exact settings, input hashes, dependency versions, progress and failures |
| `reports/data_profile.tsv`, `splits.tsv` | Missingness, country coverage, entity groups and split membership |
| `reports/cleaned_*.tsv.gz` | Original and cleaned fields for inspection |
| `reports/retrieval_trials.tsv` | Every tested retrieval/policy combination, recall, candidate count, reduction and runtime |
| `reports/retrieval_weights.tsv` | Learned five-route coefficients and intercept for each retrieval trial |
| `reports/embedding_training.tsv` | Fine-tuning losses for retrieval representations |
| `reports/candidate_summary.tsv`, `candidate_entities.tsv` | Recall ceiling and counts by split/entity |
| `reports/route_ablation.tsv`, `retrieval_misses.tsv` | Which routes retrieve truth, and whether misses happen before or after fusion |
| `reports/retrieval_work.tsv` | Per-query posting visits, skipped terms, oversized blocks, timing and route counts |
| `reports/candidate_pairs_long.tsv.gz` | Final candidate pairs, route scores, route-presence flags and fusion scores |
| `reports/training_pairs.tsv.gz`, `features/*.tsv.gz` | Actual training pairs, labels and matcher feature values |
| `reports/model_trials.tsv`, `threshold_trials.tsv` | Tested matcher settings, timings and all tested acceptance thresholds |
| `reports/model_comparison.tsv`, `country_scores.tsv` | Tune/audit macro F0.5, precision, recall, singletons and country breakdowns |
| `reports/scores/*.tsv.gz`, `entities/*.tsv`, `errors/*.tsv` | Pair scores, entity-level scores, false merges and missed matches |
| `reports/comparison_pairs_{tune,audit}.tsv.gz` | Aligned scores/decisions from all three models for later ensemble analysis |
| `reports/model_disagreements.tsv` | Whether different models correct different mistakes |
| `reports/boosted_tree_permutation_importance.tsv` | Feature importance measured on tuning entities, if enabled |
| `reports/training/*.tsv` | Cross-encoder epoch losses |
| `reports/final_retrieval_weights.tsv` | Candidate coefficients after all-data refit, if enabled |
| `evaluation_artifacts/` | Models used for the reported held-out comparison |
| `artifacts/` | Final fitted models, vectorizers, fusion, selected policies and replay manifest |
| `artifacts/logistic/feature_weights.tsv` | Final logistic coefficients and feature scaling |
| `output/models/<model>/` | Each model's own submission and exact scored-pair log |

Files ending in `.tsv.gz` are compressed TSVs: `pandas.read_csv(path, sep="\t")` reads them directly.
Runtime columns are hardware dependent. Tiny fixtures deliberately make easy patterns and should never
be compared to a real leaderboard. There is no best-model claim until real-data experiments are run.

## Replay, validate and package

```sh
ber predict --data-dir dataset --artifacts runs/experiment_001/artifacts --output-dir runs/replay_001

python code/business_entity_resolution/utils/validate_submission.py \
  --matching runs/experiment_001/output/matching_results.tsv \
  --candidate runs/experiment_001/output/candidate_pairs.tsv \
  --test-dir dataset/test

ber package --run-dir runs/experiment_001 --data-dir dataset --output team_submission.zip
```

The included validator uses only the Python standard library; also run the organiser's supplied
validator when available. Both outputs have one row for **every** test Source 1 entity, empty lists for
no results, valid distinct S2/S3 IDs, exact column names and tab separators. Every final match must be
a candidate. The candidate set is exactly the set fed into **each final matching classifier**, after all
retrieval fusion/filtering; the raw retrieval pool is reported separately. No artificial true pairs are
added during test inference and no cap of one match per source is imposed.

The ZIP contains `output/`, `code/business_entity_resolution/`, `Documentation_template.md`, reports,
saved trained artifacts and the original local neural checkpoints needed to repeat training. Shared
base checkpoints are copied once. Consequently, a full neural package can be large. Its `REPRODUCE.md`
explains offline prediction replay and complete retraining from the supplied dataset. Dependencies must
be installed; model/data lookup is unnecessary. Retain the initial local model directories until packaging.

## Software verification

```sh
ber smoke --directory runs/smoke_local --config code/business_entity_resolution/configs/default.yaml
python -m pytest code/business_entity_resolution/tests -q
```

The smoke command generates invented US/India training records, France-inclusive test records,
singletons, duplicates, hard negatives, and a tiny local BERT. It actually fine-tunes a bi-encoder and
cross-encoder, runs hyperparameter trials for all three models, refits, predicts, and validates outputs.
Tests check the exact metric, typo retrieval, unseen countries, missing fields, split isolation,
learned fusion, bad-file rejection, equality between scored and exported candidate sets, neural weight
updates, byte-identical CPU prediction replay and ZIP replay. Transformer randomness/GPU kernels may
prevent byte identity across different hardware; prefer comparing scores with tolerances there.

## Primary references

- [TF-IDF / character and word n-grams](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [FAISS HNSW and other index types](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
- [Multilingual MiniLM model card, pooling and Apache-2.0 licence](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- [Decision-threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html)

Project code and its newly learned classical model artifacts use MIT. Downloaded pretrained/fine-tuned
model artifacts retain their original applicable licence; third-party library licences remain separate.
