"""Measure retrieval quality: vector vs hybrid search across chunk sizes.

Usage:  python evaluate.py [--papers papers] [--questions eval_questions.json] [--k 4]

A retrieved chunk counts as a HIT when it comes from the expected paper AND contains at least one
of the expected keywords. Metrics:
  Hit@k : share of questions where any of the top-k chunks is a hit
  MRR   : mean of 1/rank of the first hit (rewards putting the right chunk first)
No LLM / API key needed: this evaluates retrieval only.
"""
import argparse
import csv
import glob
import json
import os
import sys

from core import Index

CONFIGS = [(400, 60), (800, 150), (1200, 200)]  # (chunk size, overlap)


def first_hit_rank(hits, q):
    for rank, h in enumerate(hits, start=1):
        if h["paper"] == q["paper"] and any(kw.lower() in h["text"].lower() for kw in q["keywords"]):
            return rank
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default="papers")
    ap.add_argument("--questions", default="eval_questions.json")
    ap.add_argument("--k", type=int, default=4)
    args = ap.parse_args()

    questions = json.load(open(args.questions, encoding="utf-8"))
    pdfs = sorted(glob.glob(os.path.join(args.papers, "*.pdf")))
    if not pdfs:
        sys.exit(f"No PDFs in '{args.papers}'. Run: python download_samples.py")
    have = {os.path.basename(p) for p in pdfs}
    questions = [q for q in questions if q["paper"] in have]
    if not questions:
        sys.exit("None of the questions match the PDFs in the papers folder.")

    rows = []
    for size, overlap in CONFIGS:
        idx = Index(path=None)  # fresh in-memory index per configuration
        chunks = sum(idx.ingest_pdf(p, os.path.basename(p), size, overlap) for p in pdfs)
        for mode in ("vector", "hybrid"):
            hits_n, rr = 0, 0.0
            for q in questions:
                rank = first_hit_rank(idx.search(q["question"], args.k, None, mode), q)
                if rank:
                    hits_n += 1
                    rr += 1.0 / rank
            rows.append({"chunk_size": size, "overlap": overlap, "mode": mode, "chunks": chunks,
                         f"hit@{args.k}": round(hits_n / len(questions), 3),
                         "mrr": round(rr / len(questions), 3)})
            print(f"size={size:<5} mode={mode:<7} hit@{args.k}={rows[-1][f'hit@{args.k}']:.2f} "
                  f"mrr={rows[-1]['mrr']:.2f}")

    os.makedirs("results", exist_ok=True)
    with open("results/eval_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    hk = f"hit@{args.k}"
    lines = [f"| Chunk size | Overlap | Mode | Chunks | {hk} | MRR |", "|---|---|---|---|---|---|"]
    lines += [f"| {r['chunk_size']} | {r['overlap']} | {r['mode']} | {r['chunks']} | {r[hk]:.2f} | {r['mrr']:.2f} |"
              for r in rows]
    open("results/eval_results.md", "w").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\n{len(questions)} questions. Saved results/eval_results.csv and results/eval_results.md")


if __name__ == "__main__":
    main()
