<div align="center">

# 🔬 ScholarLens

**Ask, compare and summarise research papers, with every answer cited to the paper and page it came from.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-7C5CFF)
![Gemini](https://img.shields.io/badge/LLM-Gemini-4285F4?logo=google&logoColor=white)
![Retrieval](https://img.shields.io/badge/Retrieval-Hybrid%20(BM25%20%2B%20Vectors)-06B6D4)

</div>

---

## 📌 What is ScholarLens?

Reading several research papers to answer one question is slow, and general chatbots tend to **make up** answers or cite things that aren't in your papers.

**ScholarLens is a Retrieval-Augmented Generation (RAG) assistant for research papers.** You upload PDFs, and it answers questions using **only text retrieved from those papers**, citing `[paper.pdf, p.N]` for every claim. It can also compare papers side by side and generate a structured summary card for each paper.

## ✨ Features

| Feature | What it does |
|---|---|
| 💬 **Cited Q&A** | Answers grounded in your PDFs, with an expandable *Sources* panel showing the exact chunks and match scores |
| 🔀 **Hybrid retrieval** | Combines semantic vector search with BM25 keyword search using Reciprocal Rank Fusion (switchable to vector-only) |
| 🔁 **Follow-up questions** | Rewrites "what about its limitations?" into a standalone search query using the chat history |
| ⚖️ **Compare mode** | Side-by-side table of papers on problem, method, datasets, results or limitations |
| 🗂️ **Paper cards** | One-click structured summary of any paper |
| 📚 **Library** | Persistent index on disk, per-paper delete and a "search within" filter |
| 📏 **Evaluation harness** | Measures Hit@k and MRR for vector vs hybrid search across chunk sizes |

<!-- Add screenshots to a docs/ folder, then uncomment:
![Ask tab](docs/ask.png)
![Compare tab](docs/compare.png)
-->

---

## 🏗️ Architecture

```text
 1. INDEXING  (once per paper)
 ─────────────────────────────────────────────────────────────

   PDF files
      │
      ▼
   Extract text page by page
      │
      ▼
   Split into overlapping chunks        (800 chars, 150 overlap)
      │
      ▼
   Embed each chunk                     (all-MiniLM-L6-v2, 384 numbers)
      │
      ▼
   ChromaDB  ◄── stores vectors + text + {paper, page} metadata


 2. QUERYING  (every question)
 ─────────────────────────────────────────────────────────────

   User question
      │
      ▼
   Follow-up? ── yes ──►  Rewrite as a standalone query
      │ no                          │
      └──────────────┬──────────────┘
                     ▼
              Hybrid search           (both read from ChromaDB)
           ┌─────────┴─────────┐
           ▼                   ▼
     Vector search      BM25 keyword search
      (by meaning)         (exact terms)
           └─────────┬─────────┘
                     ▼
       Reciprocal Rank Fusion (RRF)   merge the two rankings
                     │
                     ▼
               Top-k chunks
                     │
                     ▼
   Grounded prompt: "answer only from this context, cite sources"
                     │
                     ▼
                Gemini LLM
                     │
                     ▼
   Answer with [paper, p.N] citations + source chunks in the UI
```

---

## ⚙️ How it works

| Step | What happens | Code (`core.py`) |
|---|---|---|
| 1. **Extract** | Read text from every PDF page and remember the page number | `Index.ingest_pdf` |
| 2. **Chunk** | Split into 800-character pieces with 150 characters of overlap so ideas at a boundary are not cut in half | `chunk_text` |
| 3. **Embed** | Convert each chunk into a 384-number vector that captures its meaning | `get_embedder` |
| 4. **Store** | Save vectors, text and `{paper, page}` metadata in ChromaDB on disk | `Index.ingest_pdf` |
| 5. **Rewrite** | If the question is a follow-up, turn it into a standalone query | `rewrite_query` |
| 6. **Retrieve** | Run vector search and BM25 search, then merge the two rankings with RRF | `hybrid_search` |
| 7. **Generate** | Send the top-k chunks and the question to the LLM with a grounded prompt | `answer_question` |
| 8. **Cite** | The answer cites `[paper, p.N]` and the UI shows the source chunks | `app.py` |

### Key design decisions

| Decision | Why |
|---|---|
| **Hybrid search (BM25 + vectors)** | Vectors capture meaning; BM25 catches exact terms such as numbers, model names and dataset names, which embeddings can blur |
| **Reciprocal Rank Fusion** | Merges rankings by position, so no score normalisation between the two methods is needed |
| **800 / 150 chunking** | Chosen from the evaluation below: 400 loses context, 1200 blurs topics |
| **Grounded prompt + "say not found"** | Reduces hallucination by forcing answers from retrieved text and giving the model a safe way to refuse |
| **Low temperature (0.2)** | Keeps factual answers consistent |
| **Metadata on every chunk** | Enables citations, per-paper filtering and per-paper comparison |

---

## 📊 Evaluation & results

Retrieval was evaluated on **15 questions across 3 papers** (*Attention Is All You Need*, *BERT*, *RAG*). A retrieved chunk counts as a hit if it comes from the expected paper **and** contains one of the expected keywords.

- **Hit@4**: share of questions where a correct chunk appears in the top 4
- **MRR**: mean of 1/rank of the first correct chunk (rewards ranking the right chunk first)

| Chunk size | Overlap | Mode | Chunks | Hit@4 | MRR |
|---:|---:|---|---:|---:|---:|
| 400 | 60 | vector | 535 | 0.73 | 0.46 |
| 400 | 60 | hybrid | 535 | 0.93 | 0.74 |
| 800 | 150 | vector | 287 | 0.87 | 0.68 |
| **800** | **150** | **hybrid** | **287** | **0.93** | **0.82** |
| 1200 | 200 | vector | 197 | 0.80 | 0.48 |
| 1200 | 200 | hybrid | 197 | 0.93 | 0.75 |

**Findings**

- **Hybrid beats vector-only at every chunk size.** At chunk size 400, Hit@4 rises from 0.73 to 0.93.
- **800 characters is the best chunk size.** At 800, hybrid also improves MRR from 0.68 to 0.82 over vector-only.
- **Smaller chunks lose context and larger chunks blur topics**, which matches the trade-off seen in the vector-only rows.

**Caveats:** the test set is small (one question is about 6.7 percentage points) and ground truth is keyword-based, so treat the numbers as evidence and not as a benchmark.

Reproduce with `python evaluate.py` (no API key needed). Results are written to `results/`.

---

## 🚀 Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Mac / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download 3 sample papers into ./papers (optional)
python download_samples.py
```

**Set your free Gemini API key** from [Google AI Studio](https://aistudio.google.com/apikey):

```powershell
# Windows PowerShell
$env:GEMINI_API_KEY="your_key_here"

# Mac / Linux
# export GEMINI_API_KEY=your_key_here
```

**Run the app**

```bash
streamlit run app.py
```

In the app: sidebar → **Index ./papers folder** (or upload your own PDFs) → ask questions. The first run downloads the embedding model (about 90 MB). You can also paste the API key into the sidebar instead of using an environment variable.

> **Model not found (404)?** Model names change over time. Set a current one, for example `$env:GEMINI_MODEL="gemini-3.8-flash"`, or check the list in Google AI Studio.

### Run the evaluation

```bash
python evaluate.py
```

Add your own test questions to `eval_questions.json` using the fields `question`, `paper` (file name) and `keywords`.

---

## 📁 Project structure

```
scholarlens/
├── app.py                 # Streamlit UI: Ask, Compare, Paper cards, Library
├── core.py                # Chunking, ChromaDB index, vector / BM25 / hybrid search, prompts, LLM calls
├── evaluate.py            # Retrieval evaluation (Hit@k, MRR) across chunk sizes and modes
├── eval_questions.json    # 15 evaluation questions for the sample papers
├── download_samples.py    # Fetches sample papers from arXiv
├── requirements.txt
└── .streamlit/config.toml # Dark theme
```

## 🧰 Tech stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| PDF parsing | pypdf |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`, runs locally on CPU) |
| Vector database | ChromaDB (persistent, cosine similarity) |
| Keyword search | rank-bm25 |
| LLM | Google Gemini via `google-genai` |

---

<div align="center">


</div>
