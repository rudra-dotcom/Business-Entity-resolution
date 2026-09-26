# Business Entity Resolution — methodology

The organiser's original template was not supplied. This document covers the requested methodology,
candidate generation, architecture, feature engineering and reproducibility topics. Transfer it into
the official template if the organiser requires different headings.

## Problem and inputs

For each deduplicated Source 1 entity, predict zero or more matching Source 2/3 records. All input/output
tables are tab-separated. IDs are never model features. Country is an open string, and no country is
excluded from output. Only the challenge-provided records and labels are used for entity training;
there are no business database, internet identity, API resolution or geocoding lookups.

## Normalisation

Retain originals plus Unicode/case/whitespace/punctuation normalisation, legal abbreviations, an
additional core name, and accent-folded text. Extract tentative postal, leading house and unit values.
Treat missing values separately from conflicting values. Conservative rules do not replace ambiguous
abbreviations or invent missing address components.

## Candidate generation and learned fusion

Retrieve through name character TF-IDF, name word TF-IDF, address TF-IDF, exact composite keys, and
normalised dense embeddings searched by HNSW. Bound lexical query terms/posting lengths and skip
oversized exact blocks, logging the work and recall cost. Combine the union with a train-fitted,
nonnegative regularised logistic fusion over five route scores. Tune a threshold and candidate cap
to minimise shortlist size while satisfying micro and macro candidate-recall targets. Report failure
to meet those targets rather than hiding retrieval misses.

TF-IDF uses training-text frequency statistics. Neural retrieval compares frozen pretrained weights
against fine-tuning on training positives and hard negatives. Index structures themselves do not
learn supervised identity weights. Raw retrieval candidates are kept conceptually separate from
`candidate_pairs.tsv`, which is the exact final matcher input set.

## Matching architectures and learning

Train and compare three independent models: scaled logistic regression, histogram gradient boosting,
and a transformer cross-encoder jointly reading both records. The first two use similarity and
agreement/conflict/missing features. The cross-encoder learns a new binary head and transformer
weights from the supplied labels. Train matchers on all retrieved positives and a bounded hard/random
mixture of negatives from the broader training retrieval pool, including singleton negatives. Held-out
businesses are excluded even as supervised negatives. Validation/test scoring uses only the final
filtered candidate sets. No ensemble is automatically formed.

The default neural base is an Apache-2.0 multilingual MiniLM checkpoint. Downloads record the exact
repository revision, file hashes and licence; inference is local. Combined loaded neural models must
remain below eight billion parameters. Full model settings and licences are preserved with artifacts.

## Search, validation and score

Split by anchor groups, keeping shared-target anchors together. Train-only linked records fit lexical
statistics and neural representations. Tune retrieval configurations, fusion regularisation,
shortlist policy, matcher hyperparameters and acceptance thresholds on the tuning split. Pick the
submission model on tuning results. Evaluate the chosen variant of each family once on the audit
split. All test predictions use unlabeled test records only.

Compute F0.5 per anchor as `5 TP / (5 TP + 4 FP + FN)`; empty truth plus empty prediction scores 1.
Average over anchors including singletons. Retrieval misses are never excluded from the denominator.
Report candidate recall, macro positive-anchor candidate recall, F0.5 ceiling, reduction ratio,
candidate counts, posting visits, inference time, country results and singleton performance.

Hyperparameter search is bounded and seeded, with every tested configuration and threshold logged.
Searches are not exhaustive unless their budget covers the grid. Optional final refitting uses all
labelled data with previously selected settings fixed. Audit numbers refer to pre-refit models.

## Outputs and reproducibility

The two challenge outputs contain one row per test anchor and distinct valid target IDs; final matches
are a subset of candidates. Per-model predictions, per-pair scores, feature tables, errors, learned
fusion coefficients, logistic weights, tree permutation importance and neural losses support later
comparison. A future ensemble must be trained without using the audit labels for fitting.

The submission package includes code under `src/`, README instructions, pinned dependencies,
model artifacts, original checkpoints needed for retraining, configuration, input hashes and reports.
An independent standard-library validator checks the submission format. Neural weight files use
Safetensors. The included synthetic tests verify code paths only, not real-world model accuracy.

## Limitations

Single-machine in-memory indexes and training tables need profiling on the actual dataset. Posting
caps and approximate search can miss true matches. Address extraction is heuristic. The test's
unseen country can differ substantially from training. Validation thresholds can shift after full
refitting. No real challenge score or best model can be reported before the supplied dataset is run.
