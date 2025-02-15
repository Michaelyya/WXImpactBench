import csv
import tiktoken
from llama_index.core.node_parser.text.sentence import SentenceSplitter

# Custom Tokenizer using GPT-4o encoding
class Tokenizer():
    def __init__(self):
        self.gpt4_tokenizer = tiktoken.encoding_for_model('gpt-4o')

    def __call__(self, text):
        return self.gpt4_tokenizer.encode(text)

# Function to split text into chunks
def split_text(text: str, chunk_size=250, chunk_overlap=50):
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, tokenizer=Tokenizer())
    return splitter.split_text(text)

def process_csv(input_csv, output_csv, start_id=25):
    with open(input_csv, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    with open(output_csv, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile, quotechar='"', quoting=csv.QUOTE_ALL)

        writer.writerow(['id', 'sentence'])  # Output header

        current_id = start_id
        for line in lines:
            data = line.strip()
            if data:
                sentences = split_text(data)
                for sentence in sentences:
                    writer.writerow([current_id, sentence])  # Write split sentences
                current_id += 1

# Example usage
input_csv = 'source.csv'
output_csv = 'output-splitted-2.csv'
process_csv(input_csv, output_csv)
