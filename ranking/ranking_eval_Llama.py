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
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
dotenv.load_dotenv()
API_KEY = ""
login(token=API_KEY)
model_name = "meta-llama/Llama-3.1-8B-Instruct"

class GPTReranker:
    def __init__(self, api_key: str, model: str = model_name, window_size: int = 30, overlap: int = 10):
        if window_size <= overlap:
            raise ValueError("Window size must be greater than overlap")
        if overlap < 0:
            raise ValueError("Overlap must be non-negative")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.window_size = window_size
        self.overlap = overlap
        
    def _create_messages(self, query: str, passages: List[str], start_idx: int) -> List[Dict[str, str]]:
        messages = [
            {
                "role": "system",
                "content": """You are an expert that ranks passages based on their relevance to a given query. 
                The most relevant passage should be ranked first. 
                Important: Do not just sort the passage numbers. Evaluate each passage's content for relevance."""
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nI will give you passages numbered from {start_idx+1} to {start_idx+len(passages)}. Rank them by relevance to the query, with the most relevant first."
            }
        ]
        
        for i, passage in enumerate(passages):
            messages.extend([
                {"role": "user", "content": f"[{start_idx+i+1}] {passage}"},
                {"role": "assistant", "content": f"Received passage [{start_idx+i+1}]."}
            ])
            
        messages.append({
            "role": "user",
            "content": "Based on the content of each passage (not just their numbers), rank them from most to least relevant. Format: [most_relevant] > [next] > [next]. No explanation needed."
        })
        
        return messages

    def get_ranking_for_group(self, query: str, passages: List[str], start_idx: int = 0, max_retries: int = 3) -> List[int]:
        messages = self._create_messages(query, passages, start_idx)
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    max_tokens=150,
                    timeout=30
                )
                ranking_str = response.choices[0].message.content.strip()
                raw_ranks = Evaluator.clean_ranking_response(ranking_str)
                global_ranks = []
                for rank in raw_ranks:
                    local_idx = rank - (start_idx + 1)
                    if 0 <= local_idx < len(passages):
                        global_idx = start_idx + local_idx
                        global_ranks.append(global_idx)
                
                return global_ranks
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(5)

    def get_ranking(self, query: str, passages: List[str]) -> str:
        if not passages:
            raise ValueError("No passages provided")
        
        first_group = passages[:40]
        second_group = passages[40:80]
        third_group = passages[80:]
        
        first_ranks = self.get_ranking_for_group(query, first_group, 0)
        print(f"First group top 10: {first_ranks[:10]}")
        second_ranks = self.get_ranking_for_group(query, second_group, 40)
        print(f"Second group top 10: {second_ranks[:10]}")
        third_ranks = self.get_ranking_for_group(query, third_group, 80)
        print(f"Third group top 10: {third_ranks[:10]}")
        
        top_30_indices = []
        if first_ranks:
            top_30_indices.extend(first_ranks[:10])
        if second_ranks:
            top_30_indices.extend(second_ranks[:10])
        if third_ranks:
            top_30_indices.extend(third_ranks[:10])
        top_30_passages = [passages[i] for i in top_30_indices]
        
        final_local_ranks = self.get_ranking_for_group(query, top_30_passages, 0)
        
        final_indices = []
        for rank in final_local_ranks:
            if rank < len(top_30_indices):
                final_indices.append(top_30_indices[rank])
        
        remaining_top = [idx for idx in top_30_indices if idx not in final_indices]
        final_indices.extend(remaining_top)
        
        all_other_indices = [i for i in range(len(passages)) if i not in top_30_indices]
        final_indices.extend(all_other_indices)
        
        ranking_str = " > ".join(f"[{r+1}]" for r in final_indices)
        return ranking_str

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
    
def process_query(row: pd.Series, reranker: GPTReranker) -> Optional[RankingResult]:
    try:
        query = row['query']
        correct_passage_idx = int(row['correct_passage_index'])
        passages = [row[f'passage_{i}'] for i in range(1, 101)]
        
        ranking_response = reranker.get_ranking(query, passages)
        ranks = [i-1 for i in Evaluator.clean_ranking_response(ranking_response)]
        
        return RankingResult(
            query=query,
            correct_passage=passages[correct_passage_idx - 1],
            ranking=ranking_response,
            correct_idx=correct_passage_idx,
            passages=passages,
            ranks=ranks
        )
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return None

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found")
    
    df = pd.read_csv('./ranking/query_and_100_passages.csv')

    reranker = GPTReranker(api_key)

    results = []
    for _, row in tqdm(df.iterrows()):
        if result := process_query(row, reranker):
            print(f"\nQuery: {result.query}")
            print(f"Correct index: {result.correct_idx}")
            print(f"Ranks: {result.ranks[:10]}")  # Show first 10 ranks
            results.append(result)
            time.sleep(1) 

    qrels_file, run_file = Evaluator.write_trec_files(results)
    
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
    results_df = pd.DataFrame([vars(r) for r in results])
    results_df.to_csv('reranking_100_passages_test.csv', index=False)

main()