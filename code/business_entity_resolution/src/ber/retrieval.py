from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer

from .cleaning import exact_keys
from .metrics import pairs_to_lists

ROUTES = ["name_char", "name_word", "address", "exact", "embedding"]
PAIR_COLUMNS = ["source1_entity_id", "candidate_entity_id"] + ROUTES


class SparseTextIndex:
    """Weighted postings; never constructs an all-query by all-target matrix.

    Frequent postings and query terms may be skipped to bound work. These are
    explicit, logged recall/speed tradeoffs, not claims of exact nearest neighbours.
    """
    def __init__(self, vectorizer, matrix, max_posting, max_terms):
        self.vectorizer = vectorizer
        self.matrix = matrix.tocsr()
        self.postings = matrix.tocsc()
        self.max_posting = max_posting
        self.max_terms = max_terms

    def query(self, vector, k, minimum):
        scores = defaultdict(float)
        visited = skipped = 0
        terms = sorted(zip(vector.indices, vector.data), key=lambda x: (-x[1], x[0]))
        for term, weight in terms[:self.max_terms]:
            start, end = self.postings.indptr[term:term + 2]
            if self.max_posting and end - start > self.max_posting:
                skipped += 1
                continue
            for record, value in zip(self.postings.indices[start:end], self.postings.data[start:end]):
                scores[int(record)] += float(weight * value)
            visited += int(end - start)
        best = heapq.nsmallest(k, ((i, s) for i, s in scores.items() if s >= minimum),
                              key=lambda item: (-item[1], item[0]))
        return best, {"postings_visited": visited, "terms_skipped": skipped,
                      "records_touched": len(scores)}


def fit_vectorizers(records, config):
    vectorizers = {}
    specifications = {
        "name_char": ("name_folded", "char_wb", (config["char_min"], config["char_max"])),
        "name_word": ("name_folded", "word", (1, config["word_max"])),
        "address": ("address_folded", config["address_analyzer"],
                    (2, 4) if config["address_analyzer"] == "char_wb" else (1, 2)),
    }
    for route, (field, analyzer, ngrams) in specifications.items():
        options = {"tokenizer": str.split, "token_pattern": None} if analyzer == "word" else {}
        vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=ngrams, min_df=1,
                                     max_features=config["max_features"], sublinear_tf=True,
                                     dtype=np.float32, norm="l2", **options)
        texts = records[field].tolist()
        try:
            vectorizer.fit(texts)
        except ValueError as exc:
            if "empty vocabulary" not in str(exc):
                raise
            # Empty fields yield zero query vectors rather than a spurious shared token.
            vectorizer.fit(["ber_empty_vocabulary_placeholder"])
        vectorizers[route] = (field, vectorizer)
    return vectorizers


class RetrievalIndex:
    def __init__(self, targets, vectorizers, config, encoder=None):
        self.targets = targets.reset_index(drop=True)
        self.config = config
        self.vectorizers = vectorizers
        self.ids = self.targets.entity_id.tolist()
        self.id_to_pos = {x: i for i, x in enumerate(self.ids)}
        self.sparse = {}
        for route, (field, vectorizer) in vectorizers.items():
            matrix = vectorizer.transform(self.targets[field]) if len(targets) else csr_matrix((0, len(vectorizer.vocabulary_)), dtype=np.float32)
            self.sparse[route] = SparseTextIndex(vectorizer, matrix, config["max_posting"], config["max_query_terms"])
        self.exact = defaultdict(list)
        for i, row in enumerate(self.targets.to_dict("records")):
            for key in exact_keys(row):
                self.exact[key].append(i)
        self.encoder = encoder
        self.vectors = self.ann = None
        if encoder is not None and len(targets):
            import faiss
            self.vectors = encoder.encode(self.targets.record_text.tolist())
            self.ann = faiss.IndexHNSWFlat(self.vectors.shape[1], config["hnsw_m"], faiss.METRIC_INNER_PRODUCT)
            self.ann.hnsw.efConstruction = config["ef_construction"]
            self.ann.hnsw.efSearch = config["ef_search"]
            faiss.omp_set_num_threads(config.get("ann_threads", 1))
            self.ann.add(np.ascontiguousarray(self.vectors, dtype=np.float32))

    def retrieve(self, anchors):
        output, stats = [], []
        if anchors.empty:
            return pd.DataFrame(columns=PAIR_COLUMNS + [f"hit_{route}" for route in ROUTES]), pd.DataFrame()
        query_matrices = {route: vectorizer.transform(anchors[field])
                          for route, (field, vectorizer) in self.vectorizers.items()}
        query_dense = self.encoder.encode(anchors.record_text.tolist()) if self.encoder is not None else None
        k = self.config["route_k"]
        for qi, query in enumerate(anchors.to_dict("records")):
            start = perf_counter()
            pool = set()
            route_members = {}
            counters = {"postings_visited": 0, "terms_skipped": 0, "records_touched": 0,
                        "oversized_exact_blocks": 0}
            for route in self.sparse:
                found, work = self.sparse[route].query(query_matrices[route][qi], k,
                                                      self.config["min_similarity"])
                members = {i for i, _ in found}
                route_members[route] = members
                pool |= members
                for key, value in work.items():
                    counters[key] += value
            exact = set()
            for key in exact_keys(query):
                members = self.exact.get(key, [])
                if len(members) > self.config["max_exact_block"]:
                    counters["oversized_exact_blocks"] += 1
                    continue
                exact.update(members)
            route_members["exact"] = exact
            pool |= exact
            if self.ann is not None:
                similarities, positions = self.ann.search(query_dense[qi:qi+1], min(k, len(self.ids)))
                found = {int(i) for i, score in zip(positions[0], similarities[0])
                         if i >= 0 and score >= self.config["embedding_min_similarity"]}
                route_members["embedding"] = found
                pool |= found
            else:
                route_members["embedding"] = set()
            # No hard country filter: mismatched/unknown labels and all unseen countries survive.
            positions = sorted(pool)
            values = {}
            for route, index in self.sparse.items():
                if positions:
                    values[route] = (index.matrix[positions] @ query_matrices[route][qi].T).toarray().ravel()
                else:
                    values[route] = np.array([])
            dense_scores = np.clip(self.vectors[positions] @ query_dense[qi], 0, 1) if self.ann is not None else np.zeros(len(positions))
            for j, position in enumerate(positions):
                output.append({"source1_entity_id": query["entity_id"], "candidate_entity_id": self.ids[position],
                               **{route: float(values[route][j]) for route in self.sparse},
                               "exact": float(position in exact), "embedding": float(dense_scores[j]),
                               **{f"hit_{route}": int(position in route_members[route]) for route in ROUTES}})
            stats.append({"source1_entity_id": query["entity_id"], "raw_candidates": len(positions),
                          "query_seconds": perf_counter() - start, **counters,
                          **{f"count_{route}": len(members) for route, members in route_members.items()}})
        columns = PAIR_COLUMNS + [f"hit_{route}" for route in ROUTES]
        return pd.DataFrame(output, columns=columns), pd.DataFrame(stats)


@dataclass
class Fusion:
    weights: np.ndarray
    bias: float
    converged: bool = True

    @classmethod
    def fit(cls, pairs, truth, regularization):
        if pairs.empty:
            raise ValueError("No retrieval pairs: relax blocking limits or inspect input fields")
        x = pairs[ROUTES].to_numpy(dtype=float)
        y = np.array([float(b in truth[a]) for a, b in zip(pairs.source1_entity_id, pairs.candidate_entity_id)])
        if len(set(y)) != 2:
            raise ValueError("Retrieval training needs both positive and negative pairs; increase route_k or provide more data")
        counts = pairs.source1_entity_id.value_counts()
        sample_weights = pairs.source1_entity_id.map(lambda a: 1 / counts[a]).to_numpy()
        sample_weights /= sample_weights.sum()

        def objective(params):
            weights, bias = params[:-1], params[-1]
            z = x @ weights + bias
            loss = np.sum(sample_weights * (np.logaddexp(0, z) - y * z))
            loss += regularization * np.sum(weights ** 2) / 2
            residual = sample_weights * (expit(z) - y)
            grad = np.r_[x.T @ residual + regularization * weights, residual.sum()]
            return loss, grad

        fit = minimize(objective, np.r_[np.ones(len(ROUTES)), -3.0], method="L-BFGS-B", jac=True,
                       bounds=[(0, None)] * len(ROUTES) + [(None, None)], options={"maxiter": 1000})
        if not fit.success:
            raise RuntimeError(f"Candidate fusion optimisation failed: {fit.message}")
        return cls(fit.x[:-1], float(fit.x[-1]))

    def score(self, pairs):
        return expit(pairs[ROUTES].to_numpy(dtype=float) @ self.weights + self.bias)

    def select(self, pairs, threshold, cap):
        selected = pairs.copy()
        selected["retrieval_score"] = self.score(pairs)
        selected = selected[selected.retrieval_score >= threshold]
        selected = selected.sort_values(["source1_entity_id", "retrieval_score", "candidate_entity_id"],
                                        ascending=[True, False, True])
        selected = selected.groupby("source1_entity_id", sort=False).head(cap).reset_index(drop=True)
        return selected

    def describe(self):
        return {**dict(zip(ROUTES, map(float, self.weights))), "bias": self.bias}


def route_ablation(pairs):
    return {route: pairs_to_lists(pairs[pairs[f"hit_{route}"] == 1]) for route in ROUTES}
