import csv
import os
from openai import OpenAI
import math
import argparse
import traceback 

# Constants for ChatGPT API
CHATGPT_MODEL = 'gpt-4o-mini'  # Assuming this is the model we will use
TOKEN_LIMIT = 60000  # 128k maximum, not sure if it is the sum of input and output or not so set it to 60k.

# Initialize OpenAI API client
client = OpenAI(
    # API key from environment variable
    api_key=os.getenv("OPENAI_API_KEY"))

def call_chatgpt_api(text_chunk):
    instruction = (
    "You are an expert OCR correction assistant specializing in historical newspaper text. Your task is to:"
    "1. Correct OCR errors while preserving the original text’s meaning, structure, and formatting"
    "2. Accurately retain proper nouns, dates, numbers, and domain-specific terminology."
    "3. Maintain paragraph breaks, section headers, bylines, and other structural elements."
    "4. Remove extraneous characters (e.g., unnecessary punctuation, OCR artifacts) without altering the content."
    "5. Properly reconstruct hyphenated words that were split across lines."
    "6. Standardize spacing by eliminating extra spaces and ensuring a consistent format."
    "7. Return the corrected text as a single continuous line, with no newline characters."
    "NOTE: Do not include explanations, summaries, or additional comments. Only return the corrected text in the specified format."
    )
    try:
        response = client.chat.completions.create(
            model=CHATGPT_MODEL,
            messages=[
                {"role": "system",
                "content": instruction},
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
    # If the total word count is smaller than TOKEN_LIMIT, return this chunk
    if num_words <= TOKEN_LIMIT:
        return [text]

    # Calculate how many chunks it needs to be splitted into
    num_chunks = math.ceil(num_words / TOKEN_LIMIT)  # Floor it. E.g. 2.9 will be round to 3
    print(f"Split into {num_chunks} chunks")
    chunk_size = num_words // num_chunks

    chunks = []
    current_chunk = []
    words_in_chunk = 0

    for i, word in enumerate(words):
        current_chunk.append(word)
        words_in_chunk += 1
        # If reached the chunk size or period (".")
        if words_in_chunk >= chunk_size and (word.endswith(".") or i == len(words) - 1):
            num_words -= words_in_chunk
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            words_in_chunk = 0
            # If the rest words are less than TOKEN_LIMIT，add the remaining words as the last chunk
            if num_words < TOKEN_LIMIT:
                chunks.append(' '.join(words[i + 1:]))
                break

    # If the last chunk has any remaining texts
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)

        with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
            writer.writerow(["Date", "Text"])  # Output header

            for row in reader:
                date, text = row

                # Skip empty or irrelevant OCR text or first row
                if text.strip() == "[]" or not text.strip() or date == "Date":
                    continue

                # Remove surrounding brackets and quotes
                text = text.strip("[]\"' ")
                text_chunks = split_text_to_chunks(text)

                try:
                    processed_chunks = []
                    for chunk in text_chunks:
                        fixed_text = call_chatgpt_api(chunk)
                        processed_chunks.append(fixed_text)

                    # Join processed chunks and write to the file
                    final_text = ' '.join(processed_chunks)
                    writer.writerow([date, final_text])
                    print(f"Processed text for date: {date}")

                except RuntimeError as e:
                    print(f"ERROR OCCURRED Date {date}, text starts with: {text[:50]}")
                    print(f"ERR MESSAGE: {traceback.format_exc()}")
                    continue

if __name__ == "__main__":
    """
    Call by specifying the location of csv files.

    Usage example: 
    python Final-OCR-Script.py --src-file "test.csv"--dst-file "out.csv"
    >>> Configuration to script: {'src_file': 'test.csv', 'dst_file': 'out.csv'}

    """
    parser = argparse.ArgumentParser(description="OCR_post-correction_args",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--src-file", help="Source file location", default="/Users/yonganyu/Desktop/vulnerability-Prediction-GEOG-research-/blog/corrected/cold_English_modern_ML_corpus.csv")
    parser.add_argument("--dst-file", help="Destination file location", default="/Users/yonganyu/Desktop/vulnerability-Prediction-GEOG-research-/blog/corrected/E_deluge_English_modern_ML_corpus.csv")
    args = parser.parse_args()
    config = vars(args)
    print(f"Configuration to script: {config}")

    input_file = args.src_file  # Your input file in CSV format
    output_file = args.dst_file  # Your output file

    process_file(input_file, output_file)
