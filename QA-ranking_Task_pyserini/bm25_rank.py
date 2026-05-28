"""BM25 (Pyserini / Lucene) baseline for the QA-ranking task.

This is a drop-in replacement for the LLM listwise reranker in
``QA-ranking_Task/model_eval.py``. Instead of asking an LLM to order the
candidate passages, it ranks each query's 100-passage candidate pool with
classic BM25 retrieval from Pyserini (Anserini/Lucene).

For every row in ``datasets/QACandidate_Pool.csv`` we:
  1. build a tiny in-memory-style Lucene index over that row's 100 passages
     (one Lucene document per passage, docid = 0-based passage index),
  2. run BM25 search with the row's query,
  3. record the resulting passage order.

The output JSON uses the exact same ``RankingResult`` schema as the original
``model_eval.py``/``GPT_eval.py``, so the metrics step is unchanged.

Prerequisites: a JDK (Java 21) on PATH and ``pip install -r requirements.txt``.
Run from anywhere; dataset/output paths default relative to the repo root.
"""

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

try:
    # In-process indexer + searcher: keeps a single JVM for the whole run.
    from pyserini.index.lucene import LuceneIndexer
    from pyserini.search.lucene import LuceneSearcher
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "Could not import Pyserini. Install it with "
        "`pip install -r requirements.txt` and make sure a JDK (Java 21) is on "
        "your PATH. Pyserini is officially supported on Linux/macOS.\n"
        f"Original import error: {exc}"
    )

REPO_ROOT = Path(__file__).resolve().parent.parent
NUM_PASSAGES = 100


@dataclass
class RankingResult:
    """Same shape as QA-ranking_Task/model_eval.py so metrics.py is reusable."""
    query: str
    correct_passage: str
    ranking: str          # human-readable "[i] > [j] > ..." (1-based)
    correct_idx: int      # 1-based ground-truth passage index (from the dataset)
    passages: List[str]
    ranks: List[int]      # 0-based passage indices in ranked order


def _passage(value) -> str:
    """Coerce a passage cell to a non-null string."""
    return "" if pd.isna(value) else str(value)


def bm25_rank(query: str, passages: List[str], k1: float, b: float,
              base_tmp: str) -> List[int]:
    """Rank ``passages`` for ``query`` with BM25, returning 0-based indices.

    Passages with zero query-term overlap are not returned by Lucene; we append
    them, in their original order, after the scored ones so every candidate
    appears in the ranking (mirroring the original reranker's leftover handling).
    """
    index_dir = tempfile.mkdtemp(dir=base_tmp)
    searcher = None
    try:
        indexer = LuceneIndexer(index_dir, threads=1)
        for i, passage in enumerate(passages):
            indexer.add_doc_dict({"id": str(i), "contents": passage})
        indexer.close()

        searcher = LuceneSearcher(index_dir)
        searcher.set_bm25(k1, b)
        hits = searcher.search(query, k=len(passages))
        ranked = [int(hit.docid) for hit in hits]
    finally:
        if searcher is not None and hasattr(searcher, "close"):
            searcher.close()
        shutil.rmtree(index_dir, ignore_errors=True)

    seen = set(ranked)
    ranked.extend(i for i in range(len(passages)) if i not in seen)
    return ranked


def process_query(row: pd.Series, k1: float, b: float,
                  base_tmp: str) -> Optional[RankingResult]:
    try:
        query = str(row["query"])
        correct_idx = int(row["correct_passage_index"])  # 1-based
        passages = [_passage(row[f"passage_{i}"]) for i in range(1, NUM_PASSAGES + 1)]

        ranks = bm25_rank(query, passages, k1, b, base_tmp)
        ranking = " > ".join(f"[{r + 1}]" for r in ranks)

        return RankingResult(
            query=query,
            correct_passage=passages[correct_idx - 1],
            ranking=ranking,
            correct_idx=correct_idx,
            passages=passages,
            ranks=ranks,
        )
    except Exception as exc:  # keep the run going on a bad row
        print(f"Error processing query (id={row.get('id', '?')}): {exc}")
        return None


def save_results(results: List[RankingResult], filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pyserini BM25 ranker for the QA task")
    parser.add_argument(
        "--input",
        default=str(REPO_ROOT / "datasets" / "QACandidate_Pool.csv"),
        help="Candidate-pool CSV (default: datasets/QACandidate_Pool.csv).",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parent / "bm25_output.json"),
        help="Output JSON (RankingResult list). Default: ./bm25_output.json.",
    )
    parser.add_argument("--k1", type=float, default=0.9, help="BM25 k1 (default 0.9).")
    parser.add_argument("--b", type=float, default=0.4, help="BM25 b (default 0.4).")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N queries (handy for a quick smoke test).",
    )
    parser.add_argument(
        "--save-every", type=int, default=50,
        help="Flush results to disk every N processed queries (default 50).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if args.limit is not None:
        df = df.head(args.limit)

    base_tmp = tempfile.mkdtemp(prefix="bm25_qa_")
    results: List[RankingResult] = []
    try:
        for processed, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df)), 1):
            result = process_query(row, args.k1, args.b, base_tmp)
            if result is None:
                continue
            results.append(result)
            if processed % args.save_every == 0:
                save_results(results, args.output)
    finally:
        shutil.rmtree(base_tmp, ignore_errors=True)

    save_results(results, args.output)
    print(f"\nRanked {len(results)} queries -> {args.output}")
    print("Next: python metrics.py --input " + args.output)


if __name__ == "__main__":
    main()
