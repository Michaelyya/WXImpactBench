import csv
from openai import OpenAI
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import sys

# Constants for ChatGPT API
CHATGPT_MODEL = 'gpt-4o-mini'  # Assuming this is the model we will use
TOKEN_LIMIT = 3000  # Adjust token limit
MAX_RETRIES = 5  # Maximum retries if API call fails
CONCURRENT_WORKERS = 20  # Number of concurrent threads for API calls

# Initialize OpenAI API client
client = OpenAI(
    api_key="sk")
write_lock = Lock()  # 写入锁
csv.field_size_limit(1000000)


def call_chatgpt_api(text_chunk):
    print("Call chatgpt")
    instruction = (
    "You are an expert OCR correction assistant specializing in newspaper text. Your task is to:"
    "1. Correct OCR errors while preserving the original text's meaning, structure, and formatting."
    "2. Maintain proper nouns, dates, numbers, and specialized terms accurately."
    "3. Preserve paragraph breaks and any visible text formatting (e.g., headlines, subheadings)."
    "4. Remove unnecessary characters like extra commas, quotation marks, or periods."
    "5. Ensure hyphenated words split across lines are properly rejoined."
    "6. Preserve any visible article structure (bylines, datelines, section headers)."
    "7. Remove all extra spaces and remove all newlines characters."
    "NOTE: Do not provide any explanations, summaries, or additional comments. Output only the corrected text. Do not add any new line character, I need it to be just in one line"
    )

    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model=CHATGPT_MODEL,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": f"Correct the following newspaper OCR text:\n\n{text_chunk}"}
                ],
                n=1,
                stop=None,
                temperature=0.2
            )
            # Extract response content
            return response.choices[0].message.content.strip()
        except Exception as e:
            retries += 1
            print(f"Error calling ChatGPT API (attempt {retries}/{MAX_RETRIES}): {e}")
            time.sleep(1)  # Wait before retrying
            if retries >= MAX_RETRIES:
                raise RuntimeError(f"Max retries exceeded: {e}")
    return None


def split_text_to_chunks(text):
    words = text.split()
    num_words = len(words)
    n = math.ceil(num_words/TOKEN_LIMIT)

    if n >= num_words:
        return [' '.join([word]) for word in words]

    chunk_size = math.ceil(num_words / n)

    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, num_words, chunk_size)]
    return chunks


def process_row(row):
    date, text = row
    if text.strip() == "[]" or not text.strip() or date == "Date":
        return None  # Skip rows without any Text

    text = text.strip("[]\"' ")
    text_chunks = split_text_to_chunks(text)

    try:
        processed_chunks = []
        print(len(text_chunks))

        # Process all chunks it has
        for chunk in text_chunks:
            fixed_text = call_chatgpt_api(chunk)
            processed_chunks.append(fixed_text)

        final_text = ' '.join(processed_chunks)
        return [date, final_text[:1000000]]  # Write back the results

    except RuntimeError as e:
        print(f"ERROR OCCURRED Date {date}, text starts with: {text[:50]}")
        return None


def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)

        with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
            writer.writerow(["Date", "Text"])  # Output header

            with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
                futures = [
                    executor.submit(process_row, row)
                    for row in reader
                ]

                for future in as_completed(futures):
                    result = future.result()
                    if result:  # If there is a result, add lock before the wrrite to synch
                        with write_lock:
                            print(result)
                            writer.writerow(result)
                            print(f"Processed text for date: {result[0]}")


if __name__ == "__main__":
    filenames = ["snow"]
    for filename in filenames:
        # input_file = f"./{filename}_English_historical_corrected_unprocessed.csv"  # Your input file in CSV format
        # output_file = f"./{filename}_English_historical_corrected_processed.csv"  # Your output file

        input_file = f"./{filename}_English_modern_corrected_unprocessed.csv"  # Your input file in CSV format
        output_file = f"./{filename}_English_modern_corrected_processed.csv"  # Your output file

        # input_file = f"./OCRs/{filename}_English_modern_ML_corpus.csv"  # Your input file in CSV format
        # output_file = f"./Corrected_OCRs/{filename}_English_modern_corrected.csv"  # Your output file
        process_file(input_file, output_file)
