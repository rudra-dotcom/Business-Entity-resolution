from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .neural import LocalTransformer, pair_texts
from .util import json_save, write_tsv


class PairMatcher:
    def __init__(self, kind, parameters, neural_config, seed):
        self.kind, self.parameters, self.neural_config, self.seed = kind, parameters, neural_config, seed
        self.model = None
        self.feature_names = None

    def fit(self, pairs, features, labels, anchors, targets):
        if len(np.unique(labels)) != 2:
            raise ValueError("Matcher training needs both positive and negative candidate pairs")
        self.feature_names = list(features)
        if self.kind == "logistic":
            self.model = make_pipeline(StandardScaler(), LogisticRegression(
                C=self.parameters["C"], class_weight=self.parameters["class_weight"],
                max_iter=2000, random_state=self.seed))
            self.model.fit(features, labels)
            iterations = self.model[-1].n_iter_.max()
            if iterations >= 2000:
                raise RuntimeError("Logistic regression did not converge")
            return []
        if self.kind == "boosted_trees":
            self.model = HistGradientBoostingClassifier(**self.parameters, early_stopping=False, random_state=self.seed)
            self.model.fit(features, labels)
            return []
        if self.kind == "cross_encoder":
            self.model = LocalTransformer(self.neural_config["model_path"], self.neural_config, classifier=True)
            first, second = pair_texts(pairs, anchors, targets)
            return self.model.fit_pairs(first, second, labels, self.parameters)
        raise ValueError(f"Unknown matcher {self.kind}")

    def predict(self, pairs, features, anchors, targets):
        if pairs.empty:
            return np.empty(0)
        if list(features) != self.feature_names:
            raise ValueError("Feature schema changed between training and inference")
        if self.kind == "cross_encoder":
            return self.model.predict_pairs(*pair_texts(pairs, anchors, targets))
        return self.model.predict_proba(features)[:, 1]

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        json_save(directory / "matcher.json", {"kind": self.kind, "parameters": self.parameters,
                                               "feature_names": self.feature_names, "neural_config": self.neural_config,
                                               "seed": self.seed, "learned_artifact_license": "MIT" if self.kind != "cross_encoder" else self.model.provenance["license"]})
        if self.kind == "cross_encoder":
            self.model.save(directory / "transformer")
        else:
            joblib.dump(self.model, directory / "model.joblib")
        if self.kind == "logistic":
            weights = self.model[-1].coef_[0]
            write_tsv(directory / "feature_weights.tsv", [{"feature": name, "standardised_coefficient": weight,
                       "scaler_mean": mean, "scaler_scale": scale}
                       for name, weight, mean, scale in zip(self.feature_names, weights,
                                                          self.model[0].mean_, self.model[0].scale_)])
            json_save(directory / "intercept.json", {"intercept": float(self.model[-1].intercept_[0])})

    @classmethod
    def load(cls, directory, device=None):
        import json
        directory = Path(directory)
        info = json.loads((directory / "matcher.json").read_text())
        obj = cls(info["kind"], info["parameters"], info["neural_config"], info["seed"])
        obj.feature_names = info["feature_names"]
        if obj.kind == "cross_encoder":
            config = obj.neural_config.copy()
            if device:
                config["device"] = device
            obj.model = LocalTransformer(directory / "transformer", config, classifier=True)
        else:
            obj.model = joblib.load(directory / "model.joblib")
        return obj

    def close(self):
        if self.kind == "cross_encoder" and self.model:
            self.model.close()
