from __future__ import annotations
from typing import List, Dict, Any

from openai import OpenAI
from .utils import HybridIndex, Reranker, Doc, select_fewshots
from .ingest import build_docs_from_paths, build_docs_from_urls
from prompts import SYSTEM_PROMPT, FEWSHOTS

class CryptoRAGPipeline:
    def __init__(self, dense_model: str = "sentence-transformers/all-MiniLM-L6-v2", reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.index = HybridIndex(dense_model_name=dense_model)
        self.reranker = Reranker(reranker_model)
        self.client: OpenAI | None = None

    def set_openai(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def add_local_files(self, paths: List[str]):
        docs = build_docs_from_paths(paths, source_label="local")
        self.index.add(docs)

    def add_urls(self, urls: List[str]):
        docs = build_docs_from_urls(urls, source_label="web")
        self.index.add(docs)

    def build(self):
        self.index.build()

    def route(self, query: str) -> str:
        q = query.lower()
        if any(k in q for k in ["price", "market cap", "marketcap", "ath", "all-time high", "24h", "fear greed", "greed index"]):
            return "tools"
        return "rag"

    def build_prompt(self, query: str, contexts: List[Doc]) -> str:
        fs = select_fewshots(query, FEWSHOTS, self.index.embedder, n=2)
        few = "\n\n".join([f"Q: {x['q']}\nA: {x['a']}" for x in fs])
        ctx = "\n\n".join([f"[{i+1}] {c.text[:1200]}" for i, c in enumerate(contexts)])
        prompt = f"""{SYSTEM_PROMPT}

Few-shot examples:
{few}

Context (use to answer if relevant; cite [#]):
{ctx}

User question: {query}

Answer:"""
        return prompt

    def answer_stream(self, query: str, contexts: List[Doc], model: str = "gpt-4o-mini"):
        assert self.client is not None, "LLM client not set"
        prompt = self.build_prompt(query, contexts)
        with self.client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":SYSTEM_PROMPT},
                      {"role":"user","content":prompt}],
            stream=True,
            temperature=0.3,
            max_tokens=400
        ) as stream:
            for event in stream:
                if hasattr(event, "choices") and event.choices:
                    delta = event.choices[0].delta
                    if delta and delta.content:
                        yield delta.content

    def ask(self, query: str, k: int = 8, alpha: float = 0.5, top_k_rerank: int = 5, filters: Dict[str, Any] | None = None, stream: bool = True):
        route = self.route(query)
        if route == "tools":
            return {"route": "tools", "contexts": []}

        # Try auto-build if needed
        if not self.index.ready():
            self.index.build()
            if not self.index.ready():
                return {"route": "not_ready", "contexts": [], "reason": "index_empty" if len(self.index.docs)==0 else "build_failed"}

        hits = self.index.search(query, k=k, alpha=alpha, filters=filters)
        if not hits:
            return {"route": "not_ready", "contexts": [], "reason": "no_results"}

        reranked = self.reranker.rerank(query, hits, top_k=top_k_rerank)
        top_contexts = [d for d,_ in reranked]
        return {"route": "rag", "contexts": top_contexts}
