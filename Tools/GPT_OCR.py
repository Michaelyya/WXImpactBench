import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import tiktoken
from dotenv import load_dotenv
import csv
import ast
import re
load_dotenv()


def chunk_text(text, max_tokens=3000):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(text)
    chunks = []
    current_chunk = []
    current_count = 0

    for token in tokens:
        if current_count + 1 > max_tokens:
            chunks.append(enc.decode(current_chunk))
            current_chunk = []
            current_count = 0
        current_chunk.append(token)
        current_count += 1

    if current_chunk:
        chunks.append(enc.decode(current_chunk))

    return chunks

# Function to correct OCR text using GPT-4
def correct_ocr(text):
    response = client.chat.completions.create(model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are an expert in correcting OCR errors. Your task is to fix any OCR mistakes in the given text while preserving the original meaning and formatting as much as possible."},
        {"role": "user", "content": f"Please correct the following OCR text:\n\n{text}"},
        {"role": "assistant", "content": "Please correct the following OCR text:\n\n{text}"}
    ],
    max_tokens=4000,
    n=1,
    stop=None,
    temperature=0.5)
    return response.choices[0].message.content.strip()


def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    chunks = chunk_text(text)
    corrected_chunks = []

    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1} of {len(chunks)}...")
        corrected_chunk = correct_ocr(chunk)
        corrected_chunks.append(corrected_chunk)

    corrected_text = ' '.join(corrected_chunks)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(corrected_text)

    print(f"Corrected text has been written to {output_file}")

input_file = 'blog/processed_output.txt'
output_file = 'GPT_corrected.txt'
process_file(input_file, output_file)