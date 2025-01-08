import pandas as pd
from tqdm import tqdm
import time
import json
import requests

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

API_KEY = "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"
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"


class GPTReranker:
    def __init__(self, api_key: str, model_name: str = model_name, window_size: int = 30, overlap: int = 10):
        if window_size <= overlap:
            raise ValueError("Window size must be greater than overlap")
        if overlap < 0:
            raise ValueError("Overlap must be non-negative")
            
        # Initialize tokenizer and model
        
        self.window_size = window_size
        self.overlap = overlap
        
    def get_ranking_for_group(self, query: str, passages: List[str], start_idx: int = 0, max_retries: int = 3) -> List[int]:
        API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        client = {"API_URL": API_URL, "headers": headers}

        prompt = self._create_messages(query, passages, start_idx)
        payload = {"inputs": prompt}

        for attempt in range(max_retries):
            try:
                response = requests.post(client["API_URL"], headers=client["headers"], json=payload)
                if response.status_code == 200:
                    ranking_str = response.json()[0]["generated_text"]
                else:
                    print(f"Hugging Face API call failed with status code {response.status_code}")
                    ranking_str = response.json().get("error", "Unknown error")
                
                if ranking_str.startswith(prompt):
                    ranking_str = ranking_str[len(prompt):].strip()
                else:
                    ranking_str = ranking_str.strip()
                
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

    def _create_messages(self, query: str, passages: List[str], start_idx: int) -> str:
        # Format the prompt in Llama's expected format
        prompt = f"""<s>[INST] You are an expert that ranks passages based on their relevance to a given query. 
            The most relevant passage should be ranked first. 
            Important: Do not just sort the passage numbers. Evaluate each passage's content for relevance.

            Query: {query}

            I will give you passages numbered from {start_idx+1} to {start_idx+len(passages)}. Rank them by relevance to the query, with the most relevant first.
        """
        for i, passage in enumerate(passages):
            prompt += f"[{start_idx+i+1}] {passage}\n"
        
        prompt += """
            Based on the content of each passage (not just their numbers), rank them from most to least relevant. 
            Format: [most_relevant] > [next] > [next]. No explanation needed.[/INST]
        """
        
        return prompt

    # def get_ranking_for_group(self, query: str, passages: List[str], start_idx: int = 0, max_retries: int = 3) -> List[int]:
    #     prompt = self._create_messages(query, passages, start_idx)
        
    #     for attempt in range(max_retries):
    #         try:
    #             inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                
    #             with torch.no_grad():
    #                 outputs = self.model.generate(
    #                     **inputs,
    #                     max_new_tokens=150,
    #                     temperature=0.0,
    #                     do_sample=False,
    #                     pad_token_id=self.tokenizer.eos_token_id
    #                 )
    #             ranking_str = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #             ranking_str = ranking_str[len(prompt):]
                
    #             raw_ranks = Evaluator.clean_ranking_response(ranking_str)
    #             global_ranks = []
    #             for rank in raw_ranks:
    #                 local_idx = rank - (start_idx + 1)
    #                 if 0 <= local_idx < len(passages):
    #                     global_idx = start_idx + local_idx
    #                     global_ranks.append(global_idx)
                
    #             return global_ranks
                
    #         except Exception as e:
    #             print(f"Attempt {attempt + 1} failed: {str(e)}")
    #             if attempt == max_retries - 1:
    #                 raise
    #             time.sleep(5)

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

def save_results(results: List[RankingResult], filename: str):
    results_data = []
    for result in results:
        results_data.append({
            'query': result.query,
            'correct_passage': result.correct_passage,
            'ranking': result.ranking,
            'correct_idx': result.correct_idx,
            'passages': result.passages,
            'ranks': result.ranks
        })
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)


def main():
    df = pd.read_csv('./query_and_100_passages.csv')
    reranker = GPTReranker(API_KEY)

    results = []
    for _, row in tqdm(df.iterrows()):
        if result := process_query(row, reranker):
            print(f"\nQuery: {result.query}")
            print(f"Correct index: {result.correct_idx}")
            print(f"Ranks: {result.ranks[:10]}")  # Show first 10 ranks
            results.append(result)
            time.sleep(1) 

            save_results(results, 'mixtral_ranking_results.json')
            time.sleep(1)
    
main()
