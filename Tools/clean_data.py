import re
from openai import OpenAI

import os
from model import configure_model
from openai import OpenAI

client = OpenAI()


client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# model = configure_model()

def clean_text(raw_text):
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', raw_text)
    # Replace incorrect characters and remove non-ASCII characters
    text = re.sub(r"[^a-zA-Z0-9,.!?':; -]", '', text)
    # Replace multiple commas or misplaced punctuations, handle typical scanning errors
    text = re.sub(r",+", ', ', text).strip()
    return text

def summarize_text(clean_text):
    sentences = re.split(r'(?<=[.!?]) +', clean_text)
    summaries = []

    for sentence in sentences:
        if sentence:
            try:
                response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a highly detail-oriented assistant tasked with correcting grammatical errors, fixing punctuation, and improving the overall clarity of text without changing the original meaning."},
                    {"role": "user", "content":  f"Please correct and improve this text: \"{sentence}\""}
                ],
                max_tokens=60)
                summaries.append(response.choices[0].message.content.strip())
            except Exception as e:
                print(f"Error processing sentence [{sentence}]: {str(e)}")
                continue

    return summaries

def process_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_text = file.read()

    cleaned_text = clean_text(raw_text)
    sentence_summaries = summarize_text(cleaned_text)

    return sentence_summaries or []

if __name__ == '__main__':
    file_path = "/Users/yonganyu/Desktop/GEOG_research/blog/cold _English_historical_ML_corpus.txt"
    summaries = process_text(file_path)
    for summary in summaries:
        print(summary)