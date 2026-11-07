"""ScholarLens: research-paper RAG assistant (Streamlit UI).

Run:  streamlit run app.py
"""
import glob
import html
import os
import time

import pandas as pd
import streamlit as st

from core import ASPECTS, Index, answer_question, compare_papers, get_embedder, paper_card

st.set_page_config(page_title="ScholarLens", page_icon="🔬", layout="wide")

CSS = """
<style>
.block-container{padding-top:1.4rem;max-width:1200px}
.hero{background:linear-gradient(120deg,#4f46e5 0%,#7c5cff 45%,#06b6d4 100%);padding:28px 32px;border-radius:18px;color:#fff;margin-bottom:18px}
.hero h1{margin:0;font-size:2.1rem;color:#fff}
.hero p{margin:6px 0 0;opacity:.93;font-size:1.02rem}
.stat{background:rgba(124,92,255,.10);border:1px solid rgba(124,92,255,.35);border-radius:14px;padding:14px 18px}
.stat .n{font-size:1.7rem;font-weight:700}
.stat .l{opacity:.7;font-size:.85rem}
.src{border:1px solid rgba(150,150,170,.30);border-radius:12px;padding:12px 14px;margin:8px 0;background:rgba(150,150,170,.06)}
.chip{display:inline-block;background:#7c5cff;color:#fff;border-radius:999px;padding:2px 10px;font-size:.78rem;margin-right:6px}
.chip.alt{background:#06b6d4}
.score{float:right;font-size:.78rem;opacity:.75}
.bar{height:4px;border-radius:4px;background:rgba(150,150,170,.25);margin:8px 0}
.bar>span{display:block;height:4px;border-radius:4px;background:linear-gradient(90deg,#7c5cff,#06b6d4)}
.src-text{font-size:.86rem;opacity:.88;line-height:1.5}
.step{border:1px dashed rgba(150,150,170,.5);border-radius:14px;padding:16px;height:100%}
.step b{color:#9d8cff}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

for key, default in {"chat": [], "cards": {}, "compare": None, "flash": None}.items():
    st.session_state.setdefault(key, default)


@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~90 MB)...")
def get_index():
    get_embedder()
    return Index(path="db")


def default_key():
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            key = ""
    return key


def stat(n, label):
    return f'<div class="stat"><div class="n">{n}</div><div class="l">{label}</div></div>'


def src_cards(hits):
    out = []
    for h in hits:
        pct = int(round(100 * max(0.0, min(1.0, h["score"]))))
        text = html.escape(h["text"][:600]) + ("…" if len(h["text"]) > 600 else "")
        out.append(
            f'<div class="src"><span class="chip">{html.escape(h["paper"])}</span>'
            f'<span class="chip alt">p.{h["page"]}</span><span class="score">{pct}% match</span>'
            f'<div class="bar"><span style="width:{pct}%"></span></div>'
            f'<div class="src-text">{text}</div></div>'
        )
    st.markdown("".join(out), unsafe_allow_html=True)


def render_msg(m):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and m.get("sources"):
            st.caption(f'{m["latency"]:.1f}s · searched for: “{m["standalone"]}”')
            with st.expander(f'Sources ({len(m["sources"])})'):
                src_cards(m["sources"])


idx = get_index()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 🔬 ScholarLens")
    api_key = st.text_input("Gemini API key", type="password", value=default_key(),
                            help="Free key: aistudio.google.com/apikey")
    with st.expander("Retrieval settings"):
        mode = st.radio("Retrieval mode", ["hybrid", "vector"], horizontal=True,
                        help="hybrid = semantic vectors + BM25 keywords fused with RRF")
        top_k = st.slider("Chunks per question (top-k)", 2, 10, 5)
        chunk_size = st.slider("Chunk size (characters)", 300, 2000, 800, 100)
        overlap = st.slider("Chunk overlap (characters)", 0, 400, 150, 25)
        temperature = st.slider("LLM temperature", 0.0, 1.0, 0.2, 0.1)

    st.markdown("**Add papers**")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True,
                             label_visibility="collapsed")
    if st.button("Index uploaded papers", type="primary", disabled=not files, width="stretch"):
        bar, total = st.progress(0.0), 0
        for i, f in enumerate(files):
            with st.spinner(f"Indexing {f.name}..."):
                try:
                    total += idx.ingest_pdf(f, f.name, chunk_size, overlap)
                except Exception as e:
                    st.error(f"{f.name}: {e}")
            bar.progress((i + 1) / len(files))
        st.session_state.cards = {}
        st.session_state.flash = f"Indexed {len(files)} paper(s), {total} chunks."
        st.rerun()

    folder_pdfs = sorted(glob.glob(os.path.join("papers", "*.pdf")))
    if folder_pdfs and st.button(f"Index ./papers folder ({len(folder_pdfs)})", width="stretch"):
        total = 0
        with st.spinner("Indexing papers folder..."):
            for p in folder_pdfs:
                total += idx.ingest_pdf(p, os.path.basename(p), chunk_size, overlap)
        st.session_state.cards = {}
        st.session_state.flash = f"Indexed {len(folder_pdfs)} paper(s), {total} chunks."
        st.rerun()

    info = idx.papers()
    names = list(info)
    scope = st.multiselect("Search within", names, default=names) if names else []
    papers_filter = scope if scope and len(scope) < len(names) else None

# ------------------------------------------------------------------ header
st.markdown(
    '<div class="hero"><h1>🔬 ScholarLens</h1>'
    "<p>Ask, compare and summarise research papers. Every answer is grounded in the papers "
    "and cited with paper name and page.</p></div>",
    unsafe_allow_html=True,
)

if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

if not names:
    c = st.columns(3)
    steps = [
        ("1. Add papers", "Upload PDFs in the sidebar, or run <code>python download_samples.py</code> "
                          "to fetch 3 famous papers into <code>papers/</code>."),
        ("2. Index", "Click <b>Index uploaded papers</b>. Each page is split into chunks, embedded "
                     "and stored in ChromaDB with paper + page metadata."),
        ("3. Ask", "Add your Gemini API key, then ask questions, compare papers or generate paper cards."),
    ]
    for col, (title, body) in zip(c, steps):
        col.markdown(f'<div class="step"><b>{title}</b><br>{body}</div>', unsafe_allow_html=True)
    st.stop()

s = st.columns(4)
s[0].markdown(stat(len(names), "papers"), unsafe_allow_html=True)
s[1].markdown(stat(sum(v["pages"] for v in info.values()), "pages"), unsafe_allow_html=True)
s[2].markdown(stat(idx.count(), "chunks"), unsafe_allow_html=True)
s[3].markdown(stat(mode, "retrieval mode"), unsafe_allow_html=True)
st.write("")

tab_ask, tab_cmp, tab_cards, tab_lib = st.tabs(["💬 Ask", "⚖️ Compare", "🗂️ Paper cards", "📚 Library"])

# ------------------------------------------------------------------ Ask
with tab_ask:
    if st.session_state.chat:
        if st.button("Clear chat"):
            st.session_state.chat = []
            st.rerun()
    else:
        st.caption("Try one of these:")
        examples = ["What problem does each paper try to solve?",
                    "Which datasets were used for evaluation?",
                    "What are the main limitations mentioned?"]
        for col, ex in zip(st.columns(3), examples):
            if col.button(ex, width="stretch"):
                st.session_state.pending = ex
                st.rerun()

    for m in st.session_state.chat:
        render_msg(m)

    question = st.chat_input("Ask anything about your papers...")
    question = question or st.session_state.pop("pending", None)
    if question:
        history = [(m["role"], m["content"]) for m in st.session_state.chat[-6:]]
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("assistant"):
            with st.spinner("Searching papers and writing the answer..."):
                try:
                    res = answer_question(idx, question, api_key, top_k, papers_filter, mode,
                                          history, temperature)
                    st.session_state.chat.append({"role": "assistant", "content": res["answer"],
                                                  "sources": res["sources"], "latency": res["latency"],
                                                  "standalone": res["standalone"]})
                    st.rerun()
                except Exception as e:
                    st.session_state.chat.pop()  # drop the unanswered question
                    st.error(f"Could not answer: {e}")

# ------------------------------------------------------------------ Compare
with tab_cmp:
    if len(names) < 2:
        st.info("Index at least two papers to use compare mode.")
    else:
        sel = st.multiselect("Papers to compare", names, default=names[:2], key="cmp_papers")
        aspect = st.selectbox("Compare on", list(ASPECTS))
        if st.button("Compare papers", type="primary", disabled=len(sel) < 2):
            with st.spinner("Retrieving evidence from each paper..."):
                try:
                    st.session_state.compare = compare_papers(idx, sel, aspect, api_key, 4, mode, temperature)
                except Exception as e:
                    st.session_state.compare = None
                    st.error(f"Could not compare: {e}")
        res = st.session_state.compare
        if res:
            st.markdown(res["answer"])
            st.caption(f'{res["latency"]:.1f}s')
            with st.expander(f'Evidence used ({len(res["sources"])} chunks)'):
                src_cards(res["sources"])

# ------------------------------------------------------------------ Paper cards
with tab_cards:
    paper = st.selectbox("Paper", names, key="card_paper")
    if st.button("Generate paper card", type="primary"):
        with st.spinner("Reading the paper..."):
            try:
                st.session_state.cards[paper] = paper_card(idx, paper, api_key, mode, temperature)
            except Exception as e:
                st.error(f"Could not build card: {e}")
    card = st.session_state.cards.get(paper)
    if card:
        st.markdown(f"#### {html.escape(paper)}")
        st.markdown(card["answer"])
        with st.expander(f'Evidence used ({len(card["sources"])} chunks)'):
            src_cards(card["sources"])

# ------------------------------------------------------------------ Library
with tab_lib:
    df = pd.DataFrame([{"Paper": k, "Pages": v["pages"], "Chunks": v["chunks"]} for k, v in info.items()])
    st.dataframe(df, width="stretch", hide_index=True)
    d1, d2 = st.columns([3, 1])
    victim = d1.selectbox("Remove a paper", names, key="del_paper")
    d2.write("")
    d2.write("")
    if d2.button("Delete", width="stretch"):
        idx.delete(victim)
        st.session_state.cards.pop(victim, None)
        st.session_state.flash = f"Removed {victim}."
        st.rerun()
