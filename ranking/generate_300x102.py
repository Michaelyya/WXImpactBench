import pandas as pd
import random

RANDOM_SEED = 1510
random.seed(RANDOM_SEED)

input_file = "passages_and_queries.csv"
output_file = "query_and_100_passages.csv"

data = pd.read_csv(input_file)
output_data = []

for idx, row in data.iterrows():
    correct_passage = row['Text']
    query = row['Generated_Query']

    all_passages = data['Text'].tolist()
    all_passages.remove(correct_passage)
    distractors = random.sample(all_passages, 99)
    
    passages = distractors + [correct_passage]
    random.shuffle(passages)
    
    correct_index = passages.index(correct_passage) + 1
    
    new_row = {
        "id": idx + 1,
        "query": query,
        "correct_passage_index": correct_index
    }
    for i, passage in enumerate(passages):
        new_row[f"passage_{i+1}"] = passage
    
    output_data.append(new_row)


output_df = pd.DataFrame(output_data)
output_df.to_csv(output_file, index=False)
