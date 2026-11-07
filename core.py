"""ScholarLens core: ingestion, retrieval (vector + BM25 hybrid) and LLM helpers.

Pipeline
  indexing : PDF -> pages -> overlapping chunks -> embeddings -> ChromaDB (with paper + page metadata)
  querying : question -> (rewrite if follow-up) -> retrieve top-k -> grounded prompt -> LLM -> cited answer
"""
import hashlib
import os
import re
import time
import uuid
from functools import lru_cache

import chromadb
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

EMBED_MODEL = "all-MiniLM-L6-v2"
RRF_K = 60  # constant used in Reciprocal Rank Fusion

# What to look for when comparing papers / building paper cards
ASPECTS = {
    "Problem & motivation": "problem addressed, motivation, limitations of prior work",
    "Method / architecture": "proposed method, model architecture, approach",
    "Datasets & training": "datasets used for training and evaluation, training setup, hyperparameters",
    "Results & metrics": "experimental results, evaluation metrics, performance compared to baselines",
    "Limitations & future work": "limitations, weaknesses, future work",
}


# ----------------------------------------------------------------------------
# Embeddings + chunking
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_embedder():
    """Small, free embedding model that runs on CPU (384-dimensional vectors)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def chunk_text(text, size=800, overlap=150):
    """Split text into overlapping chunks so ideas at a boundary are not cut in half."""
    text = re.sub(r"\s+", " ", text).strip()
    overlap = max(0, min(overlap, size // 2))
    chunks, start = [], 0
    while start < len(text):
        piece = text[start:start + size]
        if piece.strip():
            chunks.append(piece)
        start += size - overlap
    return chunks


# ----------------------------------------------------------------------------
# Index: vector store + keyword index
# ----------------------------------------------------------------------------
class Index:
    """ChromaDB-backed index. path=None gives a throw-away in-memory index (used by evaluate.py)."""

    def __init__(self, path="db", name="papers"):
        if path:
            self.client = chromadb.PersistentClient(path=path)
        else:
            self.client = chromadb.EphemeralClient()
            name = f"{name}_{uuid.uuid4().hex[:8]}"
        self.col = self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        self._bm25 = None
        self._rows = []

    # ---- ingestion ---------------------------------------------------------
    def ingest_pdf(self, source, paper, size=800, overlap=150):
        """Index a PDF (path or file-like). Re-indexing the same paper replaces it. Returns #chunks."""
        reader = PdfReader(source)
        self.delete(paper)
        docs, metas = [], []
        for i, page in enumerate(reader.pages):
            for c in chunk_text(page.extract_text() or "", size, overlap):
                docs.append(c)
                metas.append({"paper": paper, "page": i + 1})
        if not docs:
            return 0
        embs = get_embedder().encode(docs, batch_size=64, normalize_embeddings=True).tolist()
        pid = hashlib.md5(paper.encode()).hexdigest()[:8]
        ids = [f"{pid}-{n}" for n in range(len(docs))]
        for s in range(0, len(docs), 500):  # add in batches
            self.col.add(
                ids=ids[s:s + 500],
                documents=docs[s:s + 500],
                embeddings=embs[s:s + 500],
                metadatas=metas[s:s + 500],
            )
        self._bm25 = None
        return len(docs)

    def delete(self, paper):
        try:
            self.col.delete(where={"paper": paper})
        except Exception:
            pass
        self._bm25 = None

    def papers(self):
        """{paper: {"chunks": n, "pages": n}}"""
        data = self.col.get(include=["metadatas"])
        out = {}
        for m in data["metadatas"] or []:
            p = out.setdefault(m["paper"], {"chunks": 0, "pages": set()})
            p["chunks"] += 1
            p["pages"].add(m["page"])
        return {k: {"chunks": v["chunks"], "pages": len(v["pages"])} for k, v in sorted(out.items())}

    def count(self):
        return self.col.count()

    # ---- retrieval ---------------------------------------------------------
    @staticmethod
    def _where(papers):
        return {"paper": {"$in": list(papers)}} if papers else None

    def vector_search(self, query, k=5, papers=None):
        """Semantic search: nearest chunk vectors to the question vector."""
        n = self.col.count()
        if n == 0:
            return []
        emb = get_embedder().encode([query], normalize_embeddings=True).tolist()
        res = self.col.query(
            query_embeddings=emb,
            n_results=min(k, n),
            where=self._where(papers),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            hits.append({"text": d, "paper": m["paper"], "page": m["page"], "score": max(0.0, 1.0 - dist)})
        return hits

    def _ensure_bm25(self):
        if self._bm25 is None:
            data = self.col.get(include=["documents", "metadatas"])
            self._rows = [
                {"text": d, "paper": m["paper"], "page": m["page"]}
                for d, m in zip(data["documents"], data["metadatas"])
            ]
            self._bm25 = BM25Okapi([tokenize(r["text"]) for r in self._rows]) if self._rows else False

    def keyword_search(self, query, k=5, papers=None):
        """Keyword search (BM25): good at exact terms like 'WMT 2014', 'BLEU', numbers, names."""
        self._ensure_bm25()
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in order:
            r = self._rows[i]
            if papers and r["paper"] not in papers:
                continue
            if scores[i] <= 0:
                break
            hits.append({**r, "score": float(scores[i])})
            if len(hits) >= k:
                break
        return hits

    def hybrid_search(self, query, k=5, papers=None, pool=20):
        """Combine vector + keyword rankings with Reciprocal Rank Fusion (RRF)."""
        rankings = [self.vector_search(query, pool, papers), self.keyword_search(query, pool, papers)]
        fused = {}
        for ranking in rankings:
            for rank, h in enumerate(ranking):
                key = (h["paper"], h["page"], h["text"])
                entry = fused.setdefault(key, {"hit": h, "rrf": 0.0})
                entry["rrf"] += 1.0 / (RRF_K + rank + 1)
        best = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)[:k]
        top = 2.0 / (RRF_K + 1)  # best possible RRF score (rank 1 in both lists)
        return [{**e["hit"], "score": min(1.0, e["rrf"] / top)} for e in best]

    def search(self, query, k=5, papers=None, mode="hybrid"):
        if mode == "vector":
            return self.vector_search(query, k, papers)
        return self.hybrid_search(query, k, papers)


# ----------------------------------------------------------------------------
# LLM helpers
# ----------------------------------------------------------------------------
ANSWER_PROMPT = """You are ScholarLens, an assistant that answers questions about research papers.
Rules:
- Use ONLY the context excerpts below. If they do not contain the answer, say you could not find it in the provided papers.
- Cite every claim like [paper.pdf, p.3], using the exact paper name and page from the excerpt header.
- Be concise and precise. Use short bullet points when listing or comparing things.

Context:
{context}

Question: {question}
Answer:"""

REWRITE_PROMPT = """Rewrite the final user question as a standalone search query, resolving pronouns and references using the conversation. Return only the rewritten query, nothing else.

Conversation:
{history}

Final question: {question}
Standalone query:"""

COMPARE_PROMPT = """You are comparing research papers on one aspect: {aspect}.
Use ONLY the excerpts below, which are grouped by paper.

Output exactly:
1. A markdown table with columns: Paper | {aspect}. One row per paper, 1-3 short sentences each, with citations like [paper.pdf, p.3]. Write "Not stated in retrieved text" if the excerpts do not cover it.
2. Then a heading "**Key differences**" followed by 2-4 bullet points.

Excerpts:
{context}
"""

CARD_PROMPT = """Create a concise paper card for "{paper}" using ONLY the excerpts below.
Use exactly this format:
**One-line summary:** ...
**Problem:** ...
**Method:** ...
**Data / experiments:** ...
**Key results:** ...
**Limitations:** ...
Each field is 1-2 sentences with a citation like [{paper}, p.N]. If a field is not in the excerpts, write "Not stated in the retrieved text".

Excerpts:
{context}
"""


def llm(prompt, api_key=None, temperature=0.2):
    from google import genai
    from google.genai import types

    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("No Gemini API key. Paste it in the sidebar or set GEMINI_API_KEY.")
    client = genai.Client(api_key=key)
    resp = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),  # change if Google renames it
        contents=prompt,
        config=types.GenerateContentConfig(temperature=temperature),
    )
    return resp.text or ""


def format_context(hits):
    return "\n\n".join(f"[{h['paper']}, p.{h['page']}]\n{h['text']}" for h in hits)


def rewrite_query(question, history, api_key=None):
    """Turn a follow-up like 'what about its limitations?' into a standalone search query."""
    lines = []
    for role, content in history[-6:]:
        who = "User" if role == "user" else "Assistant"
        lines.append(f"{who}: {content[:400]}")
    try:
        out = llm(REWRITE_PROMPT.format(history="\n".join(lines), question=question), api_key, 0.0)
        return out.strip().strip('"') or question
    except Exception:
        return question


def answer_question(index, question, api_key=None, k=5, papers=None, mode="hybrid",
                    history=None, temperature=0.2):
    t0 = time.time()
    standalone = rewrite_query(question, history, api_key) if history else question
    hits = index.search(standalone, k, papers, mode)
    if not hits:
        return {"answer": "I could not find anything relevant in the indexed papers.",
                "sources": [], "standalone": standalone, "latency": time.time() - t0}
    answer = llm(ANSWER_PROMPT.format(context=format_context(hits), question=question), api_key, temperature)
    return {"answer": answer, "sources": hits, "standalone": standalone, "latency": time.time() - t0}


def compare_papers(index, papers, aspect, api_key=None, k=4, mode="hybrid", temperature=0.2):
    t0 = time.time()
    query = ASPECTS.get(aspect, aspect)
    blocks, sources = [], []
    for p in papers:
        hits = index.search(query, k, [p], mode)
        sources.extend(hits)
        blocks.append(f"=== {p} ===\n{format_context(hits)}")
    answer = llm(COMPARE_PROMPT.format(aspect=aspect, context="\n\n".join(blocks)), api_key, temperature)
    return {"answer": answer, "sources": sources, "latency": time.time() - t0}


def paper_card(index, paper, api_key=None, mode="hybrid", temperature=0.2):
    seen, hits = set(), []
    for query in ASPECTS.values():
        for h in index.search(query, 2, [paper], mode):
            key = (h["page"], h["text"])
            if key not in seen:
                seen.add(key)
                hits.append(h)
    answer = llm(CARD_PROMPT.format(paper=paper, context=format_context(hits)), api_key, temperature)
    return {"answer": answer, "sources": hits}
