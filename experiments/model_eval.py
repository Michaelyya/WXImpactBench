from typing import List
import requests
import re
import random
import time
import json
from openai import OpenAI
import os
import dotenv
import csv
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
dotenv.load_dotenv()
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

API_KEY=""
from huggingface_hub import login
login(token=API_KEY)
huggingface_models=["meta-llama/Meta-Llama-3-8B-Instruct","Qwen/Qwen2.5-7B-Instruct","mistralai/Mixtral-8x7B-Instruct-v0.1"]
model_name = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16, 
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True 
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=slurm_tmpdir,
    device_map={"": 0}, 
    quantization_config=config  
)
model.gradient_checkpointing_enable()

def extract_answer(full_text):
    pattern = (
        r"(Infrastructural|"
        r"Agricultural|"
        r"Ecological|"
        r"Financial|"
        r"Human Health|"
        r"Political):\s*(true|false)"
    )
    answers = re.findall(pattern, full_text)
    # result = {key: value.lower() == "true" for key, value in answers}
    result = {key: 1 if value.lower() == "true" else 0 for key, value in answers}
    return result

def inference(input_text, prompt=None, typ="hf"):
    default_prompt = """
        Given the following historical newspaper text:
        "{input_text}"
        Provide a binary classification (respond ONLY with 'true' or 'false') for each impact category based on explicit mentions in the text. Follow these specific guidelines:
        1. ***Infrastructural Impact***: Classify as 'true' if the text mentions any damage or disruption to physical infrastructure and essential services. This includes structural damage to buildings, roads, or bridges; any disruptions to transportation systems such as railway cancellations or road closures; interruptions to public utilities including power and water supply; any failures in communication networks; or damage to industrial facilities. Consider only explicit mentions of physical damage or service disruptions in your classification.
        2. ***Agricultural Impact***: Classify as 'true' if the text mentions any weather-related effects on farming and livestock management operations. This includes yield variations in crops and animal products; direct damage to crops, timber resources, or livestock; modifications to agricultural practices or schedules; disruptions to food production or supply chains; impacts on farming equipment and resources; or effects on agricultural inputs including soil conditions, water availability for farming, and essential materials such as seedlings, fertilizers, or animal feed.
        3. ***Ecological Impact***: Classify as 'true' if the text mentions any effects on natural environments and ecosystems. This includes alterations to local environments and biodiversity; impacts on wildlife populations and behavior patterns; effects on non-agricultural plant life and vegetation; modifications to natural habitats including water bodies, forests, and wetlands; changes in hydrological systems such as river levels and lake conditions; or impacts on urban plant life.
        4. ***Financial Impact***: Classify as 'true' if the text explicitly mentions economic consequences of weather events. This includes direct monetary losses; business disruptions or closures requiring financial intervention; market price fluctuations or demand changes for specific goods; impacts on tourism and local economic activities; or insurance claims or economic relief measures. Focus only on explicit mentions of financial losses or fluctuations.
        5. ***Human Health Impact***: Classify as 'true' if the text mentions physical or mental health effects of weather events on populations. This includes direct injuries or fatalities (including cases where zero or more casualties are explicitly mentioned); elevated risks of weather-related or secondary illnesses; mental health consequences such as stress or anxiety; impacts on healthcare service accessibility; or long-term health implications.
        6. ***Political Impact***: Classify as 'true' if the text mentions governmental and policy responses to weather events. This includes government decision-making and policy modifications in response to events; changes in public opinion or political discourse; effects on electoral processes or outcomes; international relations and aid responses; or debates surrounding disaster preparedness and response capabilities.
        Note: 
        - Return 'false' for any impact category that is either not present in the text or not related to weather events
        - Base classifications on explicit mentions in the text
        - Focus on direct impacts rather than implications
        - Consider immediate and direct effects

        Answer only once in the following format:
        Infrastructural: true/false
        Agricultural: true/false
        Ecological: true/false
        Financial: true/false
        Human Health: true/false
        Political: true/false
    """
    prompt = prompt or default_prompt
    formatted_prompt = prompt.format(input_text=input_text)
    if typ=="gpt":
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an assistant specialized in analyzing historical weather event impacts from historical newspaper."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                top_p=0,
                frequency_penalty=0,
                presence_penalty=0
            )
            print(response.choices[0].message.content)
            result = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error: {e}")
    else:
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs,max_new_tokens=120)
        result=tokenizer.decode(outputs[0], skip_special_tokens=True)
        result=result[len(prompt):].strip()

    return result, extract_answer(result)

def process_csv(input_csv, output_csv, prompt=None,typ="hf"):
    count = 1
    with open(output_csv, mode='w', encoding='utf-8', newline='') as csv_file:
        fieldnames = [
            "ID", "Date", "Type", "Model_Type", "Infrastructural impact", 
            "Agricultural impact", "Ecological impact", "Financial impact", 
            "Human health impact", "Political impact"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        with open(input_csv, mode='r', encoding='utf-8') as input_file:
            csv_reader = csv.DictReader(input_file)
            for row in csv_reader:
                original_text = row.get("Text", "")
                date = row.get("Date", "")
                type_row = row.get("Type", "")
                id_row = row.get("ID", "")

                original_response, extracted_response = inference(original_text,typ=typ)

                result = {
                    "ID": id_row,
                    "Date": date,
                    "Type": type_row,
                    "Model_Type": model_name,
                    "Infrastructural impact": extracted_response.get("Infrastructural", ""),
                    "Agricultural impact": extracted_response.get("Agricultural", ""),
                    "Ecological impact": extracted_response.get("Ecological", ""),
                    "Financial impact": extracted_response.get("Financial", ""),
                    "Human Health impact": extracted_response.get("Human Health", ""),
                    "Political impact": extracted_response.get("Political", ""),
                }

                writer.writerow(result)
                    
                print(f'Finished {count}')
                count += 1


input_csv = "model-ability-test/Final350.csv" 
out_csv = "model-ability-test/dataset_ver1_350_output.csv" 
process_csv(input_csv,out_csv)
print(f"Results written to {out_csv}")
