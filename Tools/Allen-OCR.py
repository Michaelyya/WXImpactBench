import csv
import re
import os
import tiktoken
from openai import OpenAI
import math

# Constants for ChatGPT API
CHATGPT_MODEL = 'gpt-4o-mini'  # Assuming this is the model we will use
TOKEN_LIMIT = 60000  # 128k maximum, not sure if it is the sum of input and output or not so set it to 60k.

# Initialize OpenAI API client
client = OpenAI(
    # API key from environment variable
    api_key=os.getenv("OPENAI_API_KEY"))

encoding = tiktoken.encoding_for_model(CHATGPT_MODEL)


def call_chatgpt_api(text_chunk):
    instruction = (
    "You are an expert OCR correction assistant specializing in newspaper text. Your task is to:"
    "1. Correct OCR errors while preserving the original text's meaning, structure, and formatting."
    "2. Maintain proper nouns, dates, numbers, and specialized terms accurately."
    "3. Preserve paragraph breaks and any visible text formatting (e.g., headlines, subheadings)."
    "4. Remove unnecessary characters like extra commas, quotation marks, or periods."
    "5. Ensure hyphenated words split across lines are properly rejoined."
    "6. Preserve any visible article structure (bylines, datelines, section headers)."
    "7. Output the corrected text as a single string, with line breaks (\\n) for paragraphs and formatting."
    "NOTE: Do not provide any explanations, summaries, or additional comments. Output only the corrected text."
    )
    try:
        response = client.chat.completions.create(
            model=CHATGPT_MODEL,
            messages=[
                {"role": "system",
                "content": (f"{instruction}",
                "Your output should strictly follow these guidelines."
        )},
                {"role": "user", "content": f"Correct the following newspaper OCR text:\n\n{text_chunk}"}
            ],
            n=1,
            stop=None,
            temperature=0.2
        )
        # Extract response content
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Error calling ChatGPT API: {e}")


def split_text_to_chunks(text):
    words = text.split()
    num_words = len(words)
    # 如果总词数小于 TOKEN_LIMIT，直接返回一个 chunk
    if num_words <= TOKEN_LIMIT:
        return [text]

    # 计算需要分成的 chunk 数量
    num_chunks = math.ceil(num_words / TOKEN_LIMIT)  # 例如 2.9 倍就分成 3 段
    print(f"Split into {num_chunks} chunks")
    # 计算每个 chunk 的近似大小
    chunk_size = num_words // num_chunks

    chunks = []
    current_chunk = []
    words_in_chunk = 0

    for i, word in enumerate(words):
        current_chunk.append(word)
        words_in_chunk += 1
        # 如果到达了预设的chunk大小或者遇到了句号(".")
        if words_in_chunk >= chunk_size and (word.endswith(".") or i == len(words) - 1):
            num_words -= words_in_chunk
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            words_in_chunk = 0
            # 如果剩余的单词数少于 TOKEN_LIMIT，直接将剩余单词作为最后一个 chunk 添加
            if num_words < TOKEN_LIMIT:
                chunks.append(' '.join(words[i + 1:]))
                break

    # 如果最后一个 chunk 还有剩余单词
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks


def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)

        with open(output_file, 'w', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["Date", "Text"])  # Output header

            for row in reader:
                date, text = row

                # Skip empty or irrelevant OCR text or first row
                if text.strip() == "[]" or not text.strip() or date == "Date":
                    continue

                # Split text into chunks based on token limit
                text.strip("\"").strip("[").strip("]").strip("\'")
                text_chunks = split_text_to_chunks(text)

                try:
                    processed_chunks = []
                    for chunk in text_chunks:
                        fixed_text = call_chatgpt_api(chunk)
                        fixed_text = "[" + fixed_text + "]"
                        fixed_text.strip("\"")
                        processed_chunks.append(fixed_text)

                    # Only write to the file if all chunks are successfully processed
                    writer.writerow([date, ' '.join(processed_chunks)])
                    print(f"Processed text for date: {date}")

                except RuntimeError as e:
                    # Print error and skip the row if any API call fails
                    print(f"ERROR OCCURRED Date {date}, text starts with: {text[:50]}")
                    continue


if __name__ == "__main__":
    input_file = "torrential_English_historical_ML_corpus.csv"  # Your input file in CSV format
    output_file = "output.csv"  # Your output file
    process_file(input_file, output_file)