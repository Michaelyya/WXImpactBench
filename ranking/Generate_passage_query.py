import pandas as pd
from openai import OpenAI
import os
import dotenv
import time
from tqdm import tqdm

dotenv.load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)
def create_prompt(row):
    impacts = []
    if row['Infrastructural Impact'] > 0:
        impacts.append('infrastructure')
    if row['Political Impact'] > 0:
        impacts.append('political')
    if row['Economic Impact'] > 0:
        impacts.append('economic')
    if row['Ecological Impact'] > 0:
        impacts.append('ecological')
    if row['Agricultural Impact'] > 0:
        impacts.append('agricultural')
    if row['Human Health Impact'] > 0:
        impacts.append('human health')
    
    impact_str = ', '.join(impacts) if impacts else 'general'
    
    prompt = f"""Given the following passage about {row['Weather']}, generate a specific question that:
    1. Can be answered using ONLY the information in this passage
    2. Focuses on the {impact_str} impacts mentioned
    3. Is detailed and specific to this exact situation
    4. Requires understanding the passage's unique context
    5. Cannot be answered by other similar passages about {row['Weather']}

    Passage:
    {row['Text']}

    Generate a single, focused question that meets these criteria."""

    return prompt

def generate_query(prompt, max_retries=3):
    """Generate a query using GPT-4 with retry logic."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates specific, focused questions about weather-related passages. Your questions should be answerable using only the information in the given passage."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=400
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error after {max_retries} attempts: {e}")
                return "Error generating query"
            time.sleep(5)


df = pd.read_csv('datasets/context_data/reranking_passage.csv')
df['Generated_Query'] = ''
for idx in tqdm(df.index):
    if df.loc[idx, 'Remove'] == 0:  
        prompt = create_prompt(df.loc[idx])
        query = generate_query(prompt)
        df.loc[idx, 'Generated_Query'] = query
        time.sleep(1) 
output_file = 'reranking_passage_with_queries.csv'
df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")

