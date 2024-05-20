import re
from openai import OpenAI

import os
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

# def summarize_text(clean_text):
#     sentences = re.split(r'(?<=[.!?]) +', clean_text)
#     # summaries = []

#     # for sentence in sentences:
#     #     if sentence:
#     #         try:
#     #             response = client.chat.completions.create(
#     #             model="gpt-4",
#     #             messages=[
#     #                 {"role": "system", "content": "You are an assistant tasked with correcting grammatical errors without summarizing or altering the content."},
#     #                 {"role": "user", "content":  f"Please corect the grammar this texts: \"{sentence}\""}
#     #             ],
#     #             temperature=0.1,
#     #             max_tokens=60)
#     #             summaries.append(response.choices[0].message.content.strip())
#     #         except Exception as e:
#     #             print(f"Error processing sentence [{sentence}]: {str(e)}")
#     #             continue
#     # print(summaries)
#     return sentences

def process_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_text = file.read()

    cleaned_text = clean_text(raw_text)

    return cleaned_text

if __name__ == '__main__':
    file_path = "/Users/yonganyu/Desktop/vulnerability-Prediction-GEOG-research-/blog/cold _English_historical_ML_corpus.txt"
    summaries = process_text(file_path)
    for summary in summaries:
        print(summary)