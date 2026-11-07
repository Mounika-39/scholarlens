"""Download three well-known papers from arXiv into ./papers so you can try ScholarLens immediately."""
import os
import urllib.request

PAPERS = {
    "attention.pdf": "https://arxiv.org/pdf/1706.03762",  # Attention Is All You Need
    "bert.pdf": "https://arxiv.org/pdf/1810.04805",       # BERT
    "rag.pdf": "https://arxiv.org/pdf/2005.11401",        # Retrieval-Augmented Generation
}

os.makedirs("papers", exist_ok=True)
for name, url in PAPERS.items():
    path = os.path.join("papers", name)
    if os.path.exists(path) and os.path.getsize(path) > 10_000:
        print(f"already have {name}")
        continue
    print(f"downloading {name} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "ScholarLens-sample-downloader"})
    with urllib.request.urlopen(req, timeout=60) as r, open(path, "wb") as f:
        f.write(r.read())
print("Done. Now run: streamlit run app.py  (then click 'Index ./papers folder')")
