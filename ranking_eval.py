import pandas as pd
import random
import copy
from tqdm import tqdm
import time
import json
from openai import OpenAI
import os
import dotenv

dotenv.load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

class OpenAIReranker:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        
    def create_ranking_prompt(self, query, passages, start_idx):
        messages = [
            {
                "role": "system",
                "content": "You are an expert that ranks passages based on their relevance to a given query. The most relevant text should be able to answere the query."
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
                    model="gpt-4o",
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

def sliding_window_rerank(query, passages, window_size=5, step_size=2, api_key=None):
    reranker = OpenAIReranker(api_key)
    passages = copy.deepcopy(passages)
    total_passages = len(passages)
    passage_objects = [{'text': p, 'original_idx': i} for i, p in enumerate(passages)]
    
    end_pos = total_passages
    start_pos = max(0, end_pos - window_size)
    
    while start_pos < total_passages:
        window = passage_objects[start_pos:end_pos]
        
        if not window:
            break
        try:
            window_texts = [p['text'] for p in window]
            ranking_response = reranker.get_ranking(query, window_texts, start_pos)
            relative_ranks = clean_ranking_response(ranking_response)
            
            relative_ranks = [x - 1 for x in relative_ranks]  
            
            window_reordered = [] #Reorder it
            for rank in relative_ranks:
                if rank < len(window):
                    window_reordered.append(window[rank])
            
            included_indices = set(relative_ranks)
            for i in range(len(window)):
                if i not in included_indices:
                    window_reordered.append(window[i])
            
            passage_objects[start_pos:end_pos] = window_reordered
            
        except Exception as e:
            print(f"Error processing window {start_pos}:{end_pos}: {str(e)}")
        
        end_pos = end_pos - step_size
        start_pos = max(0, end_pos - window_size)
        time.sleep(1)
    return [p['text'] for p in passage_objects]


df = pd.read_csv('passages_and_queries.csv')
sample_df = df.sample(n=30, random_state=42)

reranked_results = []
api_key=os.environ.get("OPENAI_API_KEY")
for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
    query = row['Generated_Query']
    all_passages = df['Text'].tolist()
    random.shuffle(all_passages)
    passages_to_rank = all_passages[:30]  # Take 30 passages to test
    
    if row['Text'] not in passages_to_rank:
        passages_to_rank[-1] = row['Text']
    
    try:
        reranked_passages = sliding_window_rerank(
            query=query,
            passages=passages_to_rank,
            window_size=5,
            step_size=2,
            api_key=api_key
        )
        
        result = {
            'query': query,
            'correct_passage': row['Text'],
            'correct_passage_rank': reranked_passages.index(row['Text']) + 1,
            'total_passages': len(reranked_passages)
        }
        reranked_results.append(result)
        
    except Exception as e:
        print(f"Error processing query {idx}: {str(e)}")
    
results_df = pd.DataFrame(reranked_results)
results_df.to_csv('reranking_results.csv', index=False)
    
print("\nReranking Results Summary:")
print(f"Total queries processed: {len(results_df)}")
print(f"Average rank of correct passage: {results_df['correct_passage_rank'].mean():.2f}")
print(f"Median rank of correct passage: {results_df['correct_passage_rank'].median()}")
