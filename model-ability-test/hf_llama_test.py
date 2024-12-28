from typing import List
import requests
import re
import random
import time
import json
import csv
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
API_KEY=" "
from huggingface_hub import login
login(token=API_KEY)

huggingface_models=["meta-llama/Meta-Llama-3-8B-Instruct","mistralai/Mixtral-8x7B-Instruct-v0.1"]

def extract_answer(full_text):
    pattern = (
        r"(Infrastructural|Agricultural|Ecological|Financial|Health|Political):\s*(true|false)"
    )
    answers = re.findall(pattern, full_text)
    # result = {key: value.lower() == "true" for key, value in answers}
    result = {key: 1 if value.lower() == "true" else 0 for key, value in answers}
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

prompt = f"""
    Given the following historical newspaper text:
    "STORMY WEATHER Heavy gales over the United Kingdom Bourne weather on the Atlantic Disastrous loss of cattle shipments London, February 18 The weather continues very unsettled over the whole of the United Kingdom, and gales are reported at several stations The heavy gale which has raged at Penzance for the past two days has somewhat abated The wind is now blowing strongly from the southwest and the barometer marks 28.70 inches The gale is still blowing at Liverpool, but it has moderated a little London, February 18 The British steamer Canopus, Captain Horsfall, which arrived at Liverpool yesterday from Boston, lost her boats and 247 head of cattle, and sustained other damages in consequence of heavy weather Sports and Pastimes Curling Stuarton, X8, February 18 The curling match between the Truro and Stuarton clubs, which took place here today, resulted in a victory for Stuarton, which places the club in the van as good curlers Quebec, February 18 The Quebec Curling Club Challenge Cup was played for at the rink, St Charles street today, by the Montreal Caledonia Curling Club and the Quebec Curling Club The play was excellent on both sides, Quebec winning by 18 shots FEDERALIST London, February 19 At 2 a.m. the following was the score in the six days' walking match: Brown 328, Hazael 280, and ""Limping"" Day 258, and going splendidly AQUATIC"
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
    Health: true/false
    Political: true/false
"""

start=time.time()
model_name = "meta-llama/Llama-3.1-8B-Instruct"

start=time.time()
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16, 
    device_map="auto"     
)

model.gradient_checkpointing_enable()
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(**inputs,max_new_tokens=120)
output=tokenizer.decode(outputs[0], skip_special_tokens=True)
output=output[len(prompt):].strip()
print(output)
print(extract_answer(output))
print("-"*15)
print(f"time is {time.time()-start}")


# input_csv = "/content/selected_query.csv"  # Replace with your CSV file path
# output_json = "output_prompt_withDefinition4.json"  # Replace with your desired JSON file path
# prompt_withDefinition="""
#       Given the following text:
#       "{instruction}"
#       Classify the text for the presence of the following features. For each feature, return only 'true' or 'false' based on whether it is explicitly or implicitly mentioned in the text.
#       1. **Infrastructural impact**: Analyze the text for any mentions of damage to or disruption of physical infrastructure due to the weather event.
#       2. **Agricultural impact**: Examine the text for any indications of weather-related effects on agriculture, forestry, or livestock management
#       3. **Ecological impact**: Review the text for any mentions of effects on the natural environment due to the weather event.
#       4. **Economic impact**: Analyze the text for indications of economic consequences resulting from the weather event
#       5. **Societal impact**: Examine the text for any mentions of how the weather event affects society and daily life.
#       6. **Human Health impact**: Review the text for any indications of how the weather event affects human health, both physical and mental.
#       7. **Political impact**: Analyze the text for any indications of political consequences or responses to the weather event.

#       Respond in the format:
#       Infrastructural impact: true/false
#       Agricultural impact: true/false
#       Ecological impact: true/false
#       Economic impact: true/false
#       Societal impact: true/false
#       Human Health impact: true/false
#       Political impact: true/false
#     """
# process_csv_to_json(input_csv, output_json,prompt=prompt_withDefinition)
# print(f"Results written to {output_json}")