import re
import os
import torch
from transformers import BertTokenizer, BertForMaskedLM
import re
import nltk
from difflib import SequenceMatcher
from transformers import BertTokenizer, BertForMaskedLM
from openai import OpenAI
# Set the environment variable for enchant library path
enchant_prefix = "/opt/homebrew/opt/enchant"
os.environ['ENCHANT_PREFIX'] = enchant_prefix
os.environ['DYLD_LIBRARY_PATH'] = f"{enchant_prefix}/lib:{os.environ.get('DYLD_LIBRARY_PATH', '')}"
os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = f"{enchant_prefix}/lib:{os.environ.get('DYLD_FALLBACK_LIBRARY_PATH', '')}"
from enchant.checker import SpellChecker
def download_nltk_resources():
    """
    Download necessary NLTK resources.
    """
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

def clean_text(raw_text):
    """
    Clean the input text by removing excessive whitespace, incorrect characters, and non-ASCII characters.
    """
    rep = {
        '\n': ' ', '\\': ' ', '\"': '"', '-': ' ', '"': ' " ', 
        '"': ' " ', '"': ' " ', ',':' , ', '.':' . ', '!':' ! ', 
        '?':' ? ', "n't": " not" , "'ll": " will", '*':' * ', 
        '(': ' ( ', ')': ' ) ', "s'": "s '"
    }
    rep = dict((re.escape(k), v) for k, v in rep.items()) 
    pattern = re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], raw_text)
    text = text.strip()
    text = re.sub(r"[^a-zA-Z0-9,.!?':; -]", '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = re.sub(r"n't", ' not', text)
    text = re.sub(r"'ll", ' will', text)
    text = re.sub(r"'re", ' are', text)
    text = re.sub(r"'ve", ' have', text)
    text = re.sub(r"'m", ' am', text)
    text = re.sub(r"'s", ' is', text)
    text = re.sub(r"'d", ' would', text)
    text = re.sub(r"'t", ' not', text)
    text = re.sub(r"'em", ' them', text)
    text = re.sub(r"\'s", ' is', text)
    text = re.sub(r"\'d", ' would', text)
    return text

def process_text(file_path):
    """
    Read the text from the given file path, clean it, and return the cleaned text and suggested words.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_text = file.read()

    text = clean_text(raw_text)
    personslist = get_personslist(text)
    ignorewords = personslist + ["!", ",", ".", "\"", "?", '(', ')', '*', "'"]
    d = SpellChecker("en_US")
    words = text.split()
    incorrectwords = [w for w in words if not d.check(w) and w not in ignorewords]
    # using enchant.checker.SpellChecker, get suggested replacements
    suggestedwords = [d.suggest(w) for w in incorrectwords]
    # replace incorrect words with [MASK]
    for w in incorrectwords:
        text = text.replace(w, '[MASK]')
        raw_text = raw_text.replace(w, '[MASK]')
    return raw_text, suggestedwords

def get_personslist(text):
    personslist=[]
    for sent in nltk.sent_tokenize(text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if isinstance(chunk, nltk.tree.Tree) and chunk.label() == 'PERSON':
                personslist.insert(0, (chunk.leaves()[0][0]))
    return list(set(personslist))

def Bert_Clean_text(text):
    """
    Use BERT to clean the input text.
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    MASKIDS = [i for i, e in enumerate(tokenized_text) if e == '[MASK]']
    
    # Split text into chunks of max length 512
    chunks = [indexed_tokens[i:i+512] for i in range(0, len(indexed_tokens), 512)]
    segments_tensors_list = []
    predictions_list = []

    model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    for chunk in chunks:
        # Create the segments tensors
        segs = [i for i, e in enumerate(chunk) if e == tokenizer.convert_tokens_to_ids('.')]
        segments_ids=[]
        prev=-1
        for k, s in enumerate(segs):
            segments_ids = segments_ids + [k] * (s-prev)
            prev=s
        segments_ids = segments_ids + [len(segs)] * (len(chunk) - len(segments_ids))
        segments_tensors = torch.tensor([segments_ids])
        segments_tensors_list.append(segments_tensors)
        tokens_tensor = torch.tensor([chunk])

        # Predict all tokens
        with torch.no_grad():
            predictions = model(tokens_tensor, segments_tensors)
        predictions_list.append(predictions.logits)  # Ensure you get logits

    return predictions_list, MASKIDS, tokenizer

def predict_word(text_original, predictions_list, MASKIDS, tokenizer, suggestedwords):
    pred_words = []
    for i in range(len(MASKIDS)):
        chunk_idx = MASKIDS[i] // 512
        within_chunk_idx = MASKIDS[i] % 512
        predictions = predictions_list[chunk_idx]
        preds = torch.topk(predictions[0][within_chunk_idx], k=50)
        indices = [int(idx) for idx in preds.indices.tolist()]  # Convert each tensor to an int
        list1 = tokenizer.convert_ids_to_tokens(indices)
        list2 = suggestedwords[i]
        simmax = 0
        predicted_token = ''
        for word1 in list1:
            for word2 in list2:
                s = SequenceMatcher(None, word1, word2).ratio()
                if s is not None and s > simmax:
                    simmax = s
                    predicted_token = word1
        text_original = text_original.replace('[MASK]', predicted_token, 1)
    return text_original
    
if __name__ == '__main__':
    download_nltk_resources()
    file_path = "/Users/yonganyu/Desktop/vulnerability-Prediction-GEOG-research-/blog/thunder_English_modern_ML_corpus.csv"
    cleaned_text, suggestedwords = process_text(file_path)
    predictions, MASKIDS, tokenizer = Bert_Clean_text(cleaned_text)
    final_text = predict_word(cleaned_text, predictions, MASKIDS, tokenizer, suggestedwords)
    print("Final Text:", final_text)