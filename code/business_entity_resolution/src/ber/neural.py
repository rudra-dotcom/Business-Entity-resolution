"""Local Hugging Face training/inference. No network calls in this module."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from .util import LOG, chunks, json_save


def imports():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import torch
        from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Neural models require the pinned requirements.txt; run pip install -r requirements.txt") from exc
    return torch, AutoModel, AutoModelForSequenceClassification, AutoTokenizer


def device_for(torch, requested):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_provenance(path):
    path = Path(path)
    file = path / "provenance.json"
    if not file.is_file():
        raise FileNotFoundError(f"{file} missing. Run 'ber download-models --directory models', or provide an audited local model and provenance.json (see README).")
    data = json.loads(file.read_text())
    if data.get("license", "").lower() not in {"mit", "apache-2.0"}:
        raise ValueError("Model must declare MIT or Apache-2.0 in provenance.json")
    return data


class LocalTransformer:
    def __init__(self, directory, config, classifier=False):
        torch, AutoModel, AutoClassifier, AutoTokenizer = imports()
        self.torch = torch
        self.config = config.copy()
        self.classifier = classifier
        self.provenance = read_provenance(directory)
        torch.manual_seed(config.get("seed", 42))
        torch.set_num_threads(config.get("torch_threads", 2))
        self.device = device_for(torch, config.get("device", "auto"))
        self.tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
        kwargs = {"local_files_only": True, "trust_remote_code": False}
        if classifier:
            self.model = AutoClassifier.from_pretrained(str(directory), num_labels=1, **kwargs)
        else:
            self.model = AutoModel.from_pretrained(str(directory), **kwargs)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        if self.parameter_count > 8_000_000_000:
            raise ValueError("Model exceeds the challenge's eight-billion-parameter limit")
        if config.get("gradient_checkpointing", False):
            self.model.gradient_checkpointing_enable()
        self.model.to(self.device)

    def tokens(self, first, second=None):
        maximum = min(self.config["max_length"], getattr(self.model.config, "max_position_embeddings", 512) - 2)
        batch = self.tokenizer(first, text_pair=second, padding=True, truncation=True,
                               max_length=maximum, return_tensors="pt")
        return {key: value.to(self.device) for key, value in batch.items()}

    def pooled(self, tokens):
        output = self.model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(output.dtype)
        mean = (output * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.torch.nn.functional.normalize(mean, dim=1)

    def encode(self, texts):
        if self.classifier:
            raise TypeError("Cross-encoders do not produce retrieval embeddings")
        self.model.eval()
        output = []
        with self.torch.no_grad():
            for batch in chunks(texts, self.config["batch_size"]):
                output.append(self.pooled(self.tokens(batch)).cpu().float().numpy())
        return np.concatenate(output).astype(np.float32) if output else np.empty((0, self.model.config.hidden_size), dtype=np.float32)

    def predict_pairs(self, first, second):
        self.model.eval()
        output = []
        with self.torch.no_grad():
            for start in range(0, len(first), self.config["batch_size"]):
                end = start + self.config["batch_size"]
                logits = self.model(**self.tokens(first[start:end], second[start:end])).logits.flatten()
                output.append(self.torch.sigmoid(logits).cpu().float().numpy())
        return np.concatenate(output) if output else np.empty(0, dtype=np.float32)

    def fit_pairs(self, first, second, labels, parameters):
        torch = self.torch
        labels = np.asarray(labels, dtype=np.float32)
        if len(np.unique(labels)) != 2:
            raise ValueError("Neural training needs both matching and nonmatching pairs")
        batch_size = parameters.get("batch_size", self.config["batch_size"])
        accumulation = parameters.get("gradient_accumulation", 1)
        if batch_size < 1 or accumulation < 1:
            raise ValueError("Batch size and gradient accumulation must be positive")
        epochs = parameters["epochs"]
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=parameters["learning_rate"],
                                      weight_decay=parameters.get("weight_decay", 0.01))
        rng = np.random.default_rng(self.config.get("seed", 42))
        history = []
        total_batches = math.ceil(len(labels) / batch_size)
        for epoch in range(epochs):
            self.model.train()
            order = rng.permutation(len(labels))
            losses = []
            optimizer.zero_grad()
            for step, positions in enumerate(chunks(order, batch_size)):
                a = [first[i] for i in positions]
                b = [second[i] for i in positions]
                if self.classifier:
                    logits = self.model(**self.tokens(a, b)).logits.flatten()
                else:
                    av = self.pooled(self.tokens(a))
                    bv = self.pooled(self.tokens(b))
                    logits = ((av * bv).sum(1) - parameters.get("cosine_center", 0.5)) * parameters.get("temperature", 10.0)
                y = torch.as_tensor(labels[positions], device=self.device)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, y, pos_weight=torch.tensor(parameters.get("positive_weight", 1.0), device=self.device))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Neural training loss became non-finite")
                losses.append(float(loss.detach().cpu()))
                denominator = min(accumulation, total_batches - (step // accumulation) * accumulation)
                (loss / denominator).backward()
                if (step + 1) % accumulation == 0 or step + 1 == total_batches:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
            row = {"epoch": epoch + 1, "mean_loss": float(np.mean(losses)), "training_pairs": len(labels)}
            history.append(row)
            LOG.info("%s epoch %s/%s loss %.5f", "cross-encoder" if self.classifier else "bi-encoder", epoch + 1, epochs, row["mean_loss"])
        self.model.eval()
        return history

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory, safe_serialization=True)
        self.tokenizer.save_pretrained(directory)
        json_save(directory / "provenance.json", {**self.provenance, "parameter_count": self.parameter_count,
                                                   "fine_tuning_data": "provided challenge training records only"})
        json_save(directory / "runtime.json", self.config)

    def close(self):
        self.model.to("cpu")
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
        if self.device == "mps":
            self.torch.mps.empty_cache()


def pair_texts(pairs, anchors, targets):
    left = anchors if isinstance(anchors, dict) else anchors.set_index("entity_id").to_dict("index")
    right = targets if isinstance(targets, dict) else targets.set_index("entity_id").to_dict("index")
    return ([left[a]["record_text"] for a in pairs.source1_entity_id],
            [right[b]["record_text"] for b in pairs.candidate_entity_id])
