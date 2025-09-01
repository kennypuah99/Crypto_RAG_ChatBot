from __future__ import annotations
import re, json, hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

def cache_key(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

@dataclass
class Doc:
    id: str
    text: str
    metadata: Dict[str, Any]

class HybridIndex:
    def __init__(self, dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.dense_model_name = dense_model_name
        self.embedder = SentenceTransformer(dense_model_name)
        self.docs: List[Doc] = []
        self.bm25 = None
        self.embeddings = None

    def add(self, docs: List[Doc]):
        self.docs.extend(docs)

    def build(self):
        # Build only if we have docs
        if not self.docs:
            self.bm25, self.embeddings = None, None
            return
        corpus = [d.text for d in self.docs]
        tokenized = [c.split() for c in corpus]
        self.bm25 = BM25Okapi(tokenized)
        self.embeddings = self.embedder.encode(
            corpus, convert_to_numpy=True, normalize_embeddings=True
        )

    def ready(self) -> bool:
        return (self.bm25 is not None) and (self.embeddings is not None) and (len(self.docs) > 0)

    def search(self, query: str, k: int = 8, alpha: float = 0.5, filters: Dict[str, Any] | None = None):
        # If index isn't ready, return empty (UI/pipeline should guide the user)
        if not self.ready():
            return []

        # Dense embedding for query
        query_vec = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]

        # BM25 + dense scores
        q_tokens = query.split()
        try:
            bm25_scores = self.bm25.get_scores(q_tokens)
        except Exception:
            # Fallback if BM25 hiccups (e.g., empty tokens)
            bm25_scores = np.zeros(len(self.docs), dtype=float)
        dense_scores = (self.embeddings @ query_vec)

        # NumPy 2.x-safe normalization
        def _norm(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            rng = np.ptp(x)
            return (x - x.min()) / (rng + 1e-8)

        bm25_norm = _norm(bm25_scores)
        dense_norm = _norm(dense_scores)
        scores = alpha * bm25_norm + (1 - alpha) * dense_norm

        # Optional metadata filters
        idxs = np.arange(len(self.docs))
        if filters:
            def ok(d: Doc) -> bool:
                for kf, vf in filters.items():
                    if kf not in d.metadata:
                        return False
                    dv = str(d.metadata[kf]).lower()
                    if isinstance(vf, (list, tuple, set)):
                        if not any(str(x).lower() in dv for x in vf):
                            return False
                    else:
                        if str(vf).lower() not in dv:
                            return False
                return True

            keep = [i for i in idxs if ok(self.docs[int(i)])]
            if not keep:
                return []
            idxs = np.array(keep, dtype=int)
            scores = scores[idxs]

        # Top-k results
        order = np.argsort(-scores)[:k]
        return [(self.docs[int(idxs[i])], float(scores[i])) for i in order]

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: List[Tuple[Doc, float]], top_k: int = 5) -> List[Tuple[Doc, float]]:
        if not docs:
            return []
        pairs = [(query, d.text) for d, _ in docs]
        scores = self.model.predict(pairs)
        rescored = list(zip([d for d,_ in docs], [float(s) for s in scores]))
        rescored.sort(key=lambda x: -x[1])
        return rescored[:top_k]

def select_fewshots(query: str, fewshots: List[Dict[str, str]], embedder: SentenceTransformer, n: int = 2):
    if not fewshots:
        return []
    qv = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    ex_vecs = embedder.encode([fs["q"] for fs in fewshots], convert_to_numpy=True, normalize_embeddings=True)
    sims = ex_vecs @ qv
    order = np.argsort(-sims)[:n]
    return [fewshots[i] for i in order]
