# Verification record

Verified locally on 26 September 2026 with Python 3.12.4 on Apple Silicon, using the exact direct
dependency versions in `requirements.txt`. No challenge dataset or production pretrained checkpoint
was available for these tests. All data and initial transformer weights used below were synthetic.

Commands completed successfully:

```sh
.venv/bin/python -m pytest code/business_entity_resolution/tests -q
.venv/bin/python -m ber smoke --directory runs/final_smoke --config code/business_entity_resolution/configs/default.yaml
```

Result: **14 tests passed**. The environment emitted a non-failing joblib physical-core detection
warning and fell back to logical cores. The smoke run completed eight retrieval-policy evaluations,
two hyperparameter trials for each of the three matching models, all-data refitting, and validated
predictions for all 15 test anchors, including France.

| Requirement | Verification evidence |
|---|---|
| Read challenge TSVs and preserve empty values | Strict data loading and malformed-output tests |
| Cleaning with Unicode and open countries | Accent folding, Indic combining marks, unknown country and address-conflict tests |
| Efficient candidate retrieval | Typo recovery, bounded posting-visit checks, sparse indexes and live HNSW smoke path |
| Learn candidate weights | Nonnegative fusion training test, per-trial coefficient reports and shortlist cap checks |
| Learn all three matchers | Actual logistic/tree fits and transformer gradient updates; both encoder and cross-encoder weights checked against the initial checkpoint |
| Explore hyperparameters | More than one retrieval policy and two trials per matcher in the integration test |
| Correct validation isolation | Anchor/shared-target split tests; known held-out businesses excluded from supervised negatives |
| Correct challenge score | Explicit TP/FP/FN examples, per-anchor macro averaging and singleton conventions |
| Record experiment outputs | Trial/threshold TSVs, feature tables, weights, loss histories, per-model scores, errors and comparison reports |
| Submit exactly the scored candidates | Each model's scored-pair set equals the exported candidate-pair set |
| Validate all required output rules | Invalid-ID/duplicate/missing-row/header/subset rejection tests and independent stdlib validator |
| Reload saved artifacts | CPU replay produces byte-identical main matching and candidate files |
| Self-contained submission ZIP | Archive includes code, outputs, documentation, artifacts and original neural checkpoints |
| Reproduce after relocation | Original base-model directory is made unavailable; extracted code performs both inference and complete retraining with the same selected output |

The test fixture cross-encoder uses gradient accumulation, and the neural tests execute real training
rather than replacing models with mocks. The initial tiny model is random: its metrics do not measure
the usefulness of the recommended pretrained model. The chosen default search ranges, actual accuracy,
France generalisation, runtime, memory demand, and any ensemble benefit remain to be assessed on the
real supplied dataset. GPU execution has not been verified by these CPU tests.
