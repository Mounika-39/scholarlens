# 🔬 ScholarLens: Research Paper RAG Assistant

Ask, compare and summarise research papers. Every answer is grounded in your PDFs and cited with **paper name + page**.

## Features
- **Multi-paper Q&A** with inline citations and an expandable "Sources" panel (with match scores)
- **Hybrid retrieval**: semantic vectors + BM25 keywords, fused with Reciprocal Rank Fusion (switchable to vector-only)
- **Follow-up questions**: the app rewrites "what about its limitations?" into a standalone search query
- **Compare mode**: side-by-side table of papers on method, datasets, results, limitations
- **Paper cards**: one-click structured summary of any paper
- **Persistent index** (ChromaDB on disk), per-paper delete, "search within" filter
- **Evaluation script**: measures Hit@k and MRR for vector vs hybrid and for different chunk sizes

## Architecture
```
PDF -> pages -> overlapping chunks -> embeddings (all-MiniLM-L6-v2) -> ChromaDB (+ paper/page metadata)

question -> [rewrite if follow-up] -> vector search + BM25 -> RRF fusion -> top-k chunks
         -> grounded prompt ("answer ONLY from context, cite [paper, p.N]") -> Gemini -> cited answer
```

## Run it
```bash
python -m venv venv
venv\Scripts\activate            # Windows  (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt

python download_samples.py       # fetches 3 sample papers into ./papers
set GEMINI_API_KEY=your_key      # Windows; Mac/Linux: export GEMINI_API_KEY=your_key (or paste it in the sidebar)
streamlit run app.py
```
Free API key: https://aistudio.google.com/apikey

In the app: sidebar -> **Index ./papers folder** -> ask questions. The first run downloads the embedding model (~90 MB).

If you get a model-not-found error, set a current model name: `set GEMINI_MODEL=<model-name>` (see Google AI Studio for the list).

## Measure it (this gives you resume numbers)
```bash
python evaluate.py
```
No API key needed (it tests retrieval only). It compares chunk sizes 400/800/1200 and vector vs hybrid, then writes `results/eval_results.md`. Put **your own measured numbers** in this README and on your resume.

Add your own questions to `eval_questions.json`: `question`, `paper` (file name) and `keywords` (any of them appearing in the right chunk counts as a hit).

## Deploy (free)
1. Push this folder to GitHub (`papers/`, `db/`, `results/` are git-ignored).
2. Streamlit Community Cloud -> New app -> select the repo, main file `app.py`.
3. App settings -> Secrets: `GEMINI_API_KEY = "your_key"`.
4. Users upload PDFs in the sidebar. Note: the free tier has limited memory and storage, so the index may reset on restart.

## Results (fill in after running evaluate.py)
| Chunk size | Mode | Hit@4 | MRR |
|---|---|---|---|
| 400 | vector | | |
| 800 | vector | | |
| 800 | hybrid | | |

## Resume bullets (use only numbers you measured)
- Built **ScholarLens**, a RAG assistant over research papers with cross-paper comparison and page-level citations (sentence-transformers, ChromaDB, Gemini, Streamlit).
- Implemented **hybrid retrieval** (vector + BM25 with Reciprocal Rank Fusion) and follow-up query rewriting.
- Built an **evaluation harness** (Hit@k, MRR) over N questions; tuned chunk size and retrieval mode to raise Hit@4 from X% to Y%.
- Deployed on Streamlit Community Cloud: <your link>.

## Interview questions this project prepares you for
- Explain the full pipeline from PDF to cited answer.
- Why chunk with overlap? How did you choose the chunk size? (cite your eval table)
- Why hybrid search? When does BM25 beat embeddings, and the reverse? What is RRF?
- Why must the question and chunks use the same embedding model?
- How do you reduce hallucination? (grounded prompt, citations, "say not found", low temperature)
- How do you debug a wrong answer? (inspect retrieved chunks first: retrieval problem vs generation problem)
- How did you evaluate it? What are Hit@k and MRR? What are the limits of keyword-based ground truth?
- What are the limitations? (scanned PDFs need OCR, tables and equations extract badly, no reranker, in-process embedding on CPU)
- What would you add next? (cross-encoder reranker, OCR, table extraction, LLM-as-judge answer evaluation)

## Files
| File | Purpose |
|---|---|
| `core.py` | chunking, ChromaDB index, vector/BM25/hybrid search, prompts, LLM calls |
| `app.py` | Streamlit UI (Ask, Compare, Paper cards, Library) |
| `evaluate.py` | retrieval evaluation, writes `results/` |
| `eval_questions.json` | 15 sample questions for the 3 sample papers |
| `download_samples.py` | fetches sample papers from arXiv |
