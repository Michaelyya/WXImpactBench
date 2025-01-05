import pandas as pd
import random
import copy
from tqdm import tqdm
import time
import json
from openai import OpenAI
import os
import dotenv
import pytrec_eval
import numpy as np

dotenv.load_dotenv()

class OpenAIReranker:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        
    def create_ranking_prompt(self, query, passages, start_idx):
        messages = [
            {
                "role": "system",
                "content": "You are an expert that ranks passages based on their relevance to a given query. The most relevant text should be able to answer the query."
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nI will provide you with passages numbered [{start_idx+1} to {start_idx+len(passages)}]. Rank them based on how well they answer the query."
            },
            {
                "role": "assistant",
                "content": "I'll help rank the passages. Please provide them."
            }
        ]
        
        for i, passage in enumerate(passages):
            messages.append({
                "role": "user",
                "content": f"[{start_idx+i+1}] {passage}"
            })
            messages.append({
                "role": "assistant",
                "content": f"Received passage [{start_idx+i+1}]."
            })
            
        messages.append({
            "role": "user",
            "content": f"Please rank these passages ({len(passages)} total) in order of relevance to the query. Only output the ranking numbers in descending order of relevance, separated by ' > '. For example: [3] > [1] > [2]"
        })
        
        return messages

    def get_ranking(self, query, passages, start_idx, max_retries=3):
        messages = self.create_ranking_prompt(query, passages, start_idx)
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=0,
                    max_tokens=150
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(5)

def clean_ranking_response(response: str):
    numbers = []
    current_num = ""
    for char in response:
        if char.isdigit():
            current_num += char
        elif current_num:
            numbers.append(int(current_num))
            current_num = ""
    if current_num:
        numbers.append(int(current_num))
    return numbers

def calculate_ndcg(ranking, correct_passage_idx, k):
    # Create ideal and actual relevance lists
    ideal = [1] + [0] * (len(ranking)-1)  # One relevant document
    # Create actual relevance list
    actual = [1 if i == correct_passage_idx else 0 for i in ranking]
    dcg = 0
    idcg = 0
    for i in range(min(k, len(ranking))):
        # DCG
        if i < len(actual):
            dcg += actual[i] / np.log2(i + 2)
        # IDCG
        if i < len(ideal):
            idcg += ideal[i] / np.log2(i + 2)
    # NDCG
    ndcg = dcg / idcg if idcg > 0 else 0
    return ndcg

def evaluate_rankings(ranks, correct_idx):
    ndcg_scores = {
        'ndcg@1': calculate_ndcg(ranks, correct_idx, 1),
        'ndcg@5': calculate_ndcg(ranks, correct_idx, 5),
        'ndcg@10': calculate_ndcg(ranks, correct_idx, 10)
    }
    return ndcg_scores


df = pd.read_csv('passages_and_queries.csv')
sample_df = df.head(50)
reranked_results = []
api_key = os.environ.get("OPENAI_API_KEY")

for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
    query = row['Generated_Query']
    correct_passage = row['Text']
    other_passages = df[df['Text'] != correct_passage]['Text'].sample(n=29).tolist()# Get correct passage and 29 random ones 
    passages_to_rank = [correct_passage] + other_passages

    correct_idx = 0
    random.shuffle(passages_to_rank)
    correct_idx = passages_to_rank.index(correct_passage)
    
    try:
        reranker = OpenAIReranker(api_key)
        ranking_response = reranker.get_ranking(query, passages_to_rank, 0)
        print(f"\nQuery: {query}")
        print(f"Ranking response: {ranking_response}")
        
        ranks = clean_ranking_response(ranking_response)
        ranks = [i-1 for i in ranks]  # Convert to 0-based indexing
        
        ndcg_scores = evaluate_rankings(ranks, correct_idx)
        
        result = {
            'query': query,
            'correct_passage': correct_passage,
            'ranking': ranking_response,
            'correct_idx': correct_idx,
            'ndcg_scores': ndcg_scores,
            'passages': passages_to_rank,
            'ranks': ranks
        }
        reranked_results.append(result)
        print(f"NDCG@1:  {ndcg_scores['ndcg@1']:.4f}")
        print(f"NDCG@5:  {ndcg_scores['ndcg@5']:.4f}")
        print(f"NDCG@10: {ndcg_scores['ndcg@10']:.4f}")
        
    except Exception as e:
        print(f"Error processing query {idx}: {str(e)}")
    
    time.sleep(1)

mean_ndcg = {
    'ndcg@1': np.mean([r['ndcg_scores']['ndcg@1'] for r in reranked_results]),
    'ndcg@5': np.mean([r['ndcg_scores']['ndcg@5'] for r in reranked_results]),
    'ndcg@10': np.mean([r['ndcg_scores']['ndcg@10'] for r in reranked_results])
}

print("\nDetailed Results:")
for idx, result in enumerate(reranked_results, 1):
    print(f"\nQuery {idx}:")
    print(f"Query: {result['query']}")
    print(f"Ranking order: {result['ranking']}")
    print(f"Correct passage index: {result['correct_idx']}")
    print(f"NDCG@1:  {result['ndcg_scores']['ndcg@1']:.4f}")
    print(f"NDCG@5:  {result['ndcg_scores']['ndcg@5']:.4f}")
    print(f"NDCG@10: {result['ndcg_scores']['ndcg@10']:.4f}")
    
    print("\nTop 10 passages in ranking order:")
    for i, rank_idx in enumerate(result['ranks'][:10]):
        is_correct = rank_idx == result['correct_idx']
        print(f"Rank {i+1}: {'[CORRECT] ' if is_correct else ''}{result['passages'][rank_idx][:100]}...")

print("\nMean NDCG Scores:")
print(f"Mean NDCG@1:  {mean_ndcg['ndcg@1']:.4f}")
print(f"Mean NDCG@5:  {mean_ndcg['ndcg@5']:.4f}")
print(f"Mean NDCG@10: {mean_ndcg['ndcg@10']:.4f}")

results_df = pd.DataFrame(reranked_results)
results_df.to_csv('reranking_results_gpt_4_50-30.csv', index=False)