import pandas as pd
import random
import copy
from tqdm import tqdm
import time
import json
from openai import OpenAI
import os
import dotenv
import tempfile
import numpy as np
import pytrec_eval
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

dotenv.load_dotenv()

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
    def clean_ranking_response(response: str) -> List[int]:
        return [int(num) for num in ''.join(c if c.isdigit() else ' ' for c in response).split()]
    
    @staticmethod
    def write_trec_files(results: List[RankingResult]) -> tuple[str, str]:
        run_file = tempfile.NamedTemporaryFile(delete=False).name
        qrels_file = tempfile.NamedTemporaryFile(delete=False).name
        
        with open(run_file, 'w') as f_run, open(qrels_file, 'w') as f_qrel:
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
                    docid = f"passage_{passage_num+1}"  # Convert to 1-based passage numbering
                    score = 1.0/rank_position
                    f_run.write(f"{qid} Q0 {docid} {rank_position} {score:.4f} run\n")
                    
        return qrels_file, run_file
    
    @staticmethod
    def calculate_metrics(qrels_file: str, run_file: str) -> Dict[str, float]:
        with open(qrels_file) as f_qrel, open(run_file) as f_run:
            qrel = pytrec_eval.parse_qrel(f_qrel)
            run = pytrec_eval.parse_run(f_run)
        
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrel, 
            {'ndcg_cut.1', 'ndcg_cut.5', 'ndcg_cut.10', 'recip_rank', 'recall.5'}
        )
        scores = evaluator.evaluate(run)
        
        metrics = {
            'NDCG@1': 0.0, 
            'NDCG@5': 0.0, 
            'NDCG@10': 0.0,
            'MRR': 0.0,
            'Recall@5': 0.0
        }
        
        for query_scores in scores.values():
            metrics['NDCG@1'] += query_scores['ndcg_cut_1']
            metrics['NDCG@5'] += query_scores['ndcg_cut_5']
            metrics['NDCG@10'] += query_scores['ndcg_cut_10']
            metrics['MRR'] += query_scores['recip_rank']
            metrics['Recall@5'] += query_scores['recall_5']
        
        num_queries = len(scores)
        return {k: round(v / num_queries, 4) for k, v in metrics.items()}
    

def load_results(filename: str) -> List[RankingResult]:
    with open(filename, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    
    results = []
    for data in results_data:
        result = RankingResult(
            query=data['query'],
            correct_passage=data['correct_passage'],
            ranking=data['ranking'],
            correct_idx=data['correct_idx'],
            passages=data['passages'],
            ranks=data['ranks']
        )
        results.append(result)
    
    return results

def main():
    loaded_results = load_results('./ranking/Qwen_rank.json')
    qrels_file, run_file = Evaluator.write_trec_files(loaded_results)
    
    print("\nQRELS file contents:")
    with open(qrels_file, 'r') as f:
        print(f.read())
    
    print("\nRun file contents:")
    with open(run_file, 'r') as f:
        print(f.read())
    
    metrics = Evaluator.calculate_metrics(qrels_file, run_file)
    
    print("\nEvaluation Results:")
    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")
    
    os.unlink(qrels_file)
    os.unlink(run_file)
    results_df = pd.DataFrame([vars(r) for r in loaded_results])
    results_df.to_csv('reranking_100_passages_test.csv', index=False)

main()