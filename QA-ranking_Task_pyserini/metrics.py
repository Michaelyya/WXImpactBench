"""Retrieval metrics for the Pyserini BM25 ranking output.

Trimmed from QA-ranking_Task/metrics.py: same TREC/pytrec_eval pipeline and the
same metric set (NDCG@1/5/10, MRR, Recall@5), but it consumes the BM25 output
JSON produced by bm25_rank.py. The input schema is identical to the original
reranker's output, so the numbers are directly comparable.
"""

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytrec_eval


@dataclass
class RankingResult:
    query: str
    correct_passage: str
    ranking: str
    correct_idx: int
    passages: List[str]
    ranks: List[int]


class Evaluator:
    @staticmethod
    def write_trec_files(results: List[RankingResult]) -> tuple[str, str]:
        run_file = tempfile.NamedTemporaryFile(delete=False).name
        qrels_file = tempfile.NamedTemporaryFile(delete=False).name

        with open(run_file, "w") as f_run, open(qrels_file, "w") as f_qrel:
            for i, result in enumerate(results):
                qid = str(i)
                correct_docid = f"passage_{result.correct_idx}"
                f_qrel.write(f"{qid} 0 {correct_docid} 1\n")

                seen_ranks = set()
                adjusted_ranks = []
                for rank in result.ranks:
                    while rank in seen_ranks:
                        rank += 1
                    seen_ranks.add(rank)
                    adjusted_ranks.append(rank)

                for rank_position, passage_num in enumerate(adjusted_ranks, 1):
                    docid = f"passage_{passage_num + 1}"  # 0-based -> 1-based
                    score = 1.0 / rank_position
                    f_run.write(f"{qid} Q0 {docid} {rank_position} {score:.4f} run\n")

        return qrels_file, run_file

    @staticmethod
    def calculate_metrics(qrels_file: str, run_file: str) -> Dict[str, float]:
        with open(qrels_file) as f_qrel, open(run_file) as f_run:
            qrel = pytrec_eval.parse_qrel(f_qrel)
            run = pytrec_eval.parse_run(f_run)

        evaluator = pytrec_eval.RelevanceEvaluator(
            qrel,
            {"ndcg_cut.1", "ndcg_cut.5", "ndcg_cut.10", "recip_rank", "recall.5"},
        )
        scores = evaluator.evaluate(run)

        metrics = {"NDCG@1": 0.0, "NDCG@5": 0.0, "NDCG@10": 0.0, "MRR": 0.0, "Recall@5": 0.0}
        for query_scores in scores.values():
            metrics["NDCG@1"] += query_scores["ndcg_cut_1"]
            metrics["NDCG@5"] += query_scores["ndcg_cut_5"]
            metrics["NDCG@10"] += query_scores["ndcg_cut_10"]
            metrics["MRR"] += query_scores["recip_rank"]
            metrics["Recall@5"] += query_scores["recall_5"]

        num_queries = len(scores)
        return {k: round(v / num_queries, 4) for k, v in metrics.items()}


def load_results(filename: str) -> List[RankingResult]:
    with open(filename, "r", encoding="utf-8") as f:
        results_data = json.load(f)
    return [
        RankingResult(
            query=data["query"],
            correct_passage=data["correct_passage"],
            ranking=data["ranking"],
            correct_idx=data["correct_idx"],
            passages=data["passages"],
            ranks=data["ranks"],
        )
        for data in results_data
    ]


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Evaluate BM25 ranking output")
    parser.add_argument(
        "--input", default=str(here / "bm25_output.json"),
        help="Ranking JSON produced by bm25_rank.py (default: ./bm25_output.json).",
    )
    parser.add_argument(
        "--csv", default=str(here / "bm25-ranking-results.csv"),
        help="Where to dump the per-query results CSV (default: ./bm25-ranking-results.csv).",
    )
    args = parser.parse_args()

    loaded_results = load_results(args.input)
    qrels_file, run_file = Evaluator.write_trec_files(loaded_results)

    try:
        metrics = Evaluator.calculate_metrics(qrels_file, run_file)
    finally:
        os.unlink(qrels_file)
        os.unlink(run_file)

    print("\nEvaluation Results (BM25):")
    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")

    results_df = pd.DataFrame([vars(r) for r in loaded_results])
    results_df.to_csv(args.csv, index=False)
    print(f"\nPer-query results saved to {args.csv}")


if __name__ == "__main__":
    main()
