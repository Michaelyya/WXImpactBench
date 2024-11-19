from typing import List
from llama_index.core.node_parser import SentenceSplitter
import os
import openai
import json
import re
import random
from tqdm import tqdm
import time
from datasets import load_dataset
import argparse
import requests
import json
import csv
huggingface_models=["meta-llama/Meta-Llama-3-8B-Instruct","mistralai/Mixtral-8x7B-Instruct-v0.1"]

def split_text(text: str, chunk_size=500, chunk_overlap=50) -> List[str]:
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    sentences = splitter.split_text(text)
    return sentences

def extract_answer(full_text):
    pattern = (
        r"(Infrastructural impact|"
        r"Agricultural impact|"
        r"Ecological impact|"
        r"Economic impact|"
        r"Societal impact|"
        r"Human Health impact|"
        r"Political impact):\s*(true|false)"
    )
    answers = re.findall(pattern, full_text)
    result = {key: value for key, value in answers}
    return result

def call_huggingface(model_name, instruction, prompt=None):
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    client = {"API_URL": API_URL, "headers": headers}
    default_prompt = """
      Given the following text:
      "{instruction}"
      Classify the text for the presence of the following features. For each feature, return only 'true' or 'false' based on whether it is explicitly or implicitly mentioned in the text.

      Respond in the format:
      Infrastructural impact: true/false
      Agricultural impact: true/false
      Ecological impact: true/false
      Economic impact: true/false
      Societal impact: true/false
      Human Health impact: true/false
      Political impact: true/false
    """
    prompt = prompt or default_prompt
    formatted_prompt = prompt.format(instruction=instruction)
    payload = {"inputs": formatted_prompt}
    response = requests.post(client["API_URL"], headers=client["headers"], json=payload)
    if response.status_code == 200:
        result = response.json()[0]["generated_text"]
    else:
        print(f"Hugging Face API call failed with status code {response.status_code}")
        result = response.json().get("error", "Unknown error")

    if result.startswith(formatted_prompt):
        result = result[len(formatted_prompt):].strip()
    else:
        result = result.strip()
    return result, extract_answer(result)

api_choices=["gpt-4", "gpt-4o", "o1-preview",
        "deepseek-chat", "deepseek-coder",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
        "claude-3-opus-20240229",
        "gemini-1.5-flash-8b",
        "claude-3-sonnet-20240229",
        "gemini-002-pro",
        "gemini-002-flash"]

def process_csv_to_json(input_csv, output_json, models=huggingface_models,prompt=None):
    count = 1

    with open(output_json, mode='a', encoding='utf-8') as json_file:
        with open(input_csv, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row.get("Remove") == "1":
                    continue
                original_text = row.get("Text", "")
                date = row.get("Date", "")
                event_type = row.get("Type", "")
                weather = row.get("Weather", "")

                for model_name in models:
                    for text in split_text(original_text, chunk_size=1250, chunk_overlap=10):
                        try:
                            original_response, extracted_response = call_huggingface(model_name, text, prompt=prompt)
                            result = {
                                "Date": date,
                                "Type": event_type,
                                "Weather": weather,
                                "Classification": extracted_response,
                                "Original_response": original_response,
                                "Model_Type": model_name
                            }
                        except Exception as e:
                            print(f"Error processing model {model_name} for text: {text} - {e}")
                            result = {
                                "Date": date,
                                "Type": event_type,
                                "Weather": weather,
                                "Classification": "ERROR",
                                "Original_response": str(e),
                                "Model_Type": model_name
                            }
                        json_file.write(json.dumps(result, ensure_ascii=False, indent=4) + "\n")
                print(f'Finished {count}')
                count += 1

API_KEY = ""
input_csv = "/content/selected_query.csv"  # Replace with your CSV file path
output_json = "output_prompt_withDefinition4.json"  # Replace with your desired JSON file path
prompt_withDefinition="""
      Given the following text:
      "{instruction}"
      Classify the text for the presence of the following features. For each feature, return only 'true' or 'false' based on whether it is explicitly or implicitly mentioned in the text.
      1. **Infrastructural impact**: Analyze the text for any mentions of damage to or disruption of physical infrastructure due to the weather event.
      2. **Agricultural impact**: Examine the text for any indications of weather-related effects on agriculture, forestry, or livestock management
      3. **Ecological impact**: Review the text for any mentions of effects on the natural environment due to the weather event.
      4. **Economic impact**: Analyze the text for indications of economic consequences resulting from the weather event
      5. **Societal impact**: Examine the text for any mentions of how the weather event affects society and daily life.
      6. **Human Health impact**: Review the text for any indications of how the weather event affects human health, both physical and mental.
      7. **Political impact**: Analyze the text for any indications of political consequences or responses to the weather event.

      Respond in the format:
      Infrastructural impact: true/false
      Agricultural impact: true/false
      Ecological impact: true/false
      Economic impact: true/false
      Societal impact: true/false
      Human Health impact: true/false
      Political impact: true/false
    """
process_csv_to_json(input_csv, output_json,prompt=prompt_withDefinition)
print(f"Results written to {output_json}")