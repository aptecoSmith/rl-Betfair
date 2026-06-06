"""Lightweight, DB-free vector search tier.

No database, no server: embeddings live in a flat `.npy` next to a small `.json` sidecar,
and search is brute-force cosine in numpy (instant at our scale). Pluggable embedder:
  - Model2VecEmbedder  - static embeddings, true semantic recall (optional install).
  - HashingEmbedder    - deterministic numpy fallback, zero extra deps/model.
`get_embedder()` picks the best available. Everything degrades gracefully; if numpy is
absent the caller simply falls back to BM25 (this module just won't import).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str):
    return _TOKEN.findall(text.lower())


class HashingEmbedder:
    """Deterministic bag-of-hashed-tokens vector. Lexical, but zero-dependency and stable."""
    name = "hashing-v1"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in _tokens(t):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class Model2VecEmbedder:
    """Static-embedding model (model2vec). Optional - only used if installed."""
    name = "model2vec"

    def __init__(self, model="minishlab/potion-base-8M"):
        from model2vec import StaticModel  # raises ImportError if absent
        self._m = StaticModel.from_pretrained(model)
        self.dim = self._m.dim

    def embed(self, texts):
        v = np.asarray(self._m.encode(list(texts)), dtype=np.float32)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return v / norms


def get_embedder():
    try:
        return Model2VecEmbedder()
    except Exception:
        return HashingEmbedder()


def _content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


class VectorStore:
    def __init__(self, base: Path, embedder=None):
        self.base = Path(base)
        self.npy = self.base.with_suffix(".embeddings.npy")
        self.meta_path = self.base.with_suffix(".embeddings.json")
        self.embedder = embedder or get_embedder()
        self.ids: list[str] = []
        self.hashes: list[str] = []
        self.matrix = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._load()

    def _load(self):
        if self.npy.exists() and self.meta_path.exists():
            meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
            if meta.get("embedder") == self.embedder.name and meta.get("dim") == self.embedder.dim:
                self.ids = meta["ids"]
                self.hashes = meta["hashes"]
                self.matrix = np.load(self.npy)

    def save(self):
        np.save(self.npy, self.matrix)
        self.meta_path.write_text(json.dumps(
            {"embedder": self.embedder.name, "dim": self.embedder.dim,
             "ids": self.ids, "hashes": self.hashes}), encoding="utf-8")

    def update(self, items):
        """items: list of (id, text). Re-embeds only changed/new ids; drops removed ids."""
        existing = {i: (h, n) for n, (i, h) in enumerate(zip(self.ids, self.hashes))}
        new_ids, new_hashes, rows = [], [], []
        to_embed, embed_idx = [], []
        for _id, text in items:
            h = _content_hash(text)
            new_ids.append(_id)
            new_hashes.append(h)
            if _id in existing and existing[_id][0] == h:
                rows.append(self.matrix[existing[_id][1]])
            else:
                rows.append(None)
                to_embed.append(text)
                embed_idx.append(len(rows) - 1)
        if to_embed:
            vecs = self.embedder.embed(to_embed)
            for k, idx in enumerate(embed_idx):
                rows[idx] = vecs[k]
        self.ids = new_ids
        self.hashes = new_hashes
        self.matrix = (np.vstack(rows).astype(np.float32) if rows
                       else np.zeros((0, self.embedder.dim), dtype=np.float32))
        return len(to_embed)

    def search(self, query: str, k: int = 10):
        if self.matrix.shape[0] == 0:
            return []
        q = self.embedder.embed([query])[0]
        sims = self.matrix @ q
        order = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in order]

    def rerank(self, query: str, candidate_ids):
        """Hybrid step: reorder a BM25 candidate set by vector cosine."""
        if self.matrix.shape[0] == 0:
            return [(c, 0.0) for c in candidate_ids]
        pos = {i: n for n, i in enumerate(self.ids)}
        q = self.embedder.embed([query])[0]
        scored = []
        for c in candidate_ids:
            if c in pos:
                scored.append((c, float(self.matrix[pos[c]] @ q)))
            else:
                scored.append((c, 0.0))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored
