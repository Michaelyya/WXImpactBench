# QA-ranking Task — Pyserini BM25 baseline

This folder is a BM25 (lexical) version of the [`QA-ranking_Task`](../QA-ranking_Task)
ranking task. The original task uses an LLM to do **listwise reranking** of each
query's candidate pool (`model_eval.py` with HuggingFace models, `GPT_eval.py`
with the OpenAI API). Here, the same pool is ranked with classic **BM25**
retrieval from [Pyserini](https://github.com/castorini/pyserini)
(Anserini / Lucene), giving a reproducible non-LLM baseline.

The evaluation is unchanged: `bm25_rank.py` writes the **same `RankingResult`
JSON schema** as the original reranker, and `metrics.py` reports the same
metrics (NDCG@1/5/10, MRR, Recall@5) via `pytrec_eval`, so the numbers are
directly comparable to the LLM runs.

## How it works

The benchmark gives each query its own 100-passage candidate pool
(`datasets/QACandidate_Pool.csv`: `id, query, correct_passage_index,
passage_1 … passage_100`). The task is to rank *those 100 passages*, so for each
query we:

1. build a small Lucene index over its 100 passages (one document per passage,
   `docid` = 0-based passage index),
2. run BM25 search with the query (`k1=0.9`, `b=0.4` by default — Anserini's
   defaults),
3. order the passages by BM25 score. Passages with no query-term overlap aren't
   returned by Lucene, so they're appended (in original order) after the scored
   ones, exactly like the original reranker handled leftovers.

Indexing per query (rather than over one global corpus) keeps IDF scoped to the
candidate pool, which matches the "rank this pool" framing of the benchmark.

## Prerequisites

- Python 3.10 or 3.11
- **A JDK — Java 21** on your `PATH` (`java -version` should report 21).
  Pyserini ships the Anserini fat-jar but needs a JVM to run it.
- Pyserini is officially supported on **Linux/macOS**; Windows support is
  unofficial. On Windows, prefer WSL or a Linux/Mac machine.

## Setup

```bash
cd QA-ranking_Task_pyserini
pip install -r requirements.txt
```

## Run

1. Produce the BM25 ranking:

   ```bash
   python bm25_rank.py
   # options:
   #   --input    candidate-pool CSV (default: ../datasets/QACandidate_Pool.csv)
   #   --output   ranking JSON       (default: ./bm25_output.json)
   #   --k1 --b   BM25 parameters    (default: 0.9 / 0.4)
   #   --limit N  only the first N queries (quick smoke test)
   ```

2. Evaluate:

   ```bash
   python metrics.py --input bm25_output.json
   ```

   This prints NDCG@1/5/10, MRR and Recall@5 and writes a per-query
   `bm25-ranking-results.csv`.

## Notes

- Tune `--k1` / `--b` to sweep BM25 settings.
- To regenerate the candidate pool or the pseudo-queries, the original
  `Generate_Query.py` / `Generate_Pool.py` in `../QA-ranking_Task` are unchanged
  and still apply.
