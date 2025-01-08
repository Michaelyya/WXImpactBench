import pandas as pd
import time
import os
# import dotenv
import numpy as np
from tqdm import tqdm
from openai import OpenAI
import re

# dotenv.load_dotenv()
model = "gpt-4"

class OpenAIScorer:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
    
    def create_scoring_prompt(self, query, passage):
        messages = [
            {
                "role": "system",
                "content": "You are an expert that evaluates the relevance of passages to a given query. Please score each passage based on its ability to answer the query. The score should be a number between 0 (not relevant at all) and 100 (perfectly answers the query)."
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nPassage: {passage}\n\nPlease provide a score (0-100). Only Answer with a number"
            }
        ]
        return messages

    def get_score(self, query, passage, max_retries=3):
        messages = self.create_scoring_prompt(query, passage)
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=150
                )
                match = re.search(r'\d+', response.choices[0].message.content.strip())
                score = int(match.group()) if match else -1
                # print(score)
                return score
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise e
                time.sleep(5)

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
    return {
        'ndcg@1': calculate_ndcg(ranks, correct_idx, 1),
        'ndcg@5': calculate_ndcg(ranks, correct_idx, 5),
        'ndcg@10': calculate_ndcg(ranks, correct_idx, 10)
    }

df = pd.read_csv('query_and_100_passages.csv')
reranked_results = []
# api_key = os.environ.get("OPENAI_API_KEY")


scorer = OpenAIScorer(api_key)

for row_idx, row in tqdm(df.iterrows(), total=len(df)):
    query = row['query']
    correct_passage_idx = row['correct_passage_index']
    
    try:
        scores = []
        for i in range(1, 101):
            passage = row[f'passage_{i}']
            score = scorer.get_score(query, passage)
            scores.append({'index': i, 'score': score})
            # print(f"Query: {query[:50]}... | Passage: {passage[:50]}... | Score: {score}")

        sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)
        sorted_indices = [score['index'] for score in sorted_scores]

        ndcg_scores = evaluate_rankings(sorted_indices, correct_passage_idx)

        result = {
            'query': query,
            'correct_passage': row[f'passage_{correct_passage_idx}'],
            'scores': scores,
            'ndcg_scores': ndcg_scores,
        }
        reranked_results.append(result)

        print(f"NDCG@1:  {ndcg_scores['ndcg@1']:.4f}")
        print(f"NDCG@5:  {ndcg_scores['ndcg@5']:.4f}")
        print(f"NDCG@10: {ndcg_scores['ndcg@10']:.4f}")

    except Exception as e:
        print(f"Error processing query {row_idx}: {str(e)}")

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
    print(f"Scores: {result['scores']}")
    # print(f"Sorted indices: {result['sorted_indices']}")
    # print(f"Correct passage index: {result['correct_idx']}")
    print(f"NDCG@1:  {result['ndcg_scores']['ndcg@1']:.4f}")
    print(f"NDCG@5:  {result['ndcg_scores']['ndcg@5']:.4f}")
    print(f"NDCG@10: {result['ndcg_scores']['ndcg@10']:.4f}")

print("\nMean NDCG Scores:")
print(f"Mean NDCG@1:  {mean_ndcg['ndcg@1']:.4f}")
print(f"Mean NDCG@5:  {mean_ndcg['ndcg@5']:.4f}")
print(f"Mean NDCG@10: {mean_ndcg['ndcg@10']:.4f}")

# Save results
results_df = pd.DataFrame(reranked_results)
results_df.to_csv(f'score_based_ranking_results_{model}.csv', index=False)
