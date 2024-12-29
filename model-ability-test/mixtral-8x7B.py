import re
import requests
import csv
huggingface_models=["mistralai/Mixtral-8x7B-Instruct-v0.1"]

def extract_answer(full_text):
    pattern = (
        r"(Infrastructural|"
        r"Agricultural|"
        r"Ecological|"
        r"Financial|"
        r"Health|"
        r"Political):\s*(true|false)"
    )
    answers = re.findall(pattern, full_text)
    # result = {key: value.lower() == "true" for key, value in answers}
    result = {key: 1 if value.lower() == "true" else 0 for key, value in answers}
    return result

def call_huggingface(model_name, instruction):
    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {API_KEY}"}
    client = {"API_URL": API_URL, "headers": headers}
    prompt = """
        Given the following historical newspaper text:
        "{instruction}"
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
    
    # print("Result is ", result)
    print("Extractted result is ", extract_answer(result))
    return result, extract_answer(result)


def process_csv(input_csv, output_csv, models=huggingface_models):
    count = 1

    with open(output_csv, mode='w', encoding='utf-8', newline='') as csv_file:
        fieldnames = [
            "ID", "Date", "Type", "Weather", "Model_Type", "Infrastructural impact", 
            "Agricultural impact", "Ecological impact", "Financial impact", 
            "Human health impact", "Political impact", "isError"
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        with open(input_csv, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                # if row.get("Remove") == "1":
                #     continue
                id = row.get("ID", "")
                original_text = row.get("Text", "")
                date = row.get("Date", "")
                event_type = row.get("Type", "")
                weather = row.get("Weather", "")

                for model_name in models:
                    text = original_text
                    try:
                        original_response, extracted_response = call_huggingface(model_name, text)
                        result = {
                            "ID": id,
                            "Date": date,
                            "Type": event_type,
                            "Weather": weather,
                            "Model_Type": model_name,
                            "Infrastructural impact": extracted_response.get("Infrastructural", ""),
                            "Agricultural impact": extracted_response.get("Agricultural", ""),
                            "Ecological impact": extracted_response.get("Ecological", ""),
                            "Financial impact": extracted_response.get("Financial", ""),
                            "Human health impact": extracted_response.get("Health", ""),
                            "Political impact": extracted_response.get("Political", ""),
                            "isError": "false"
                        }
                        # print(result)
                        writer.writerow(result)
                    except Exception as e:
                        print(f"Error processing model {model_name} for text id: {id} - {e}")
                        result = {
                            "ID": id,
                            "Date": date,
                            "Type": event_type,
                            "Weather": weather,
                            "Model_Type": model_name,
                            "Infrastructural impact": '-1',
                            "Agricultural impact": '-1',
                            "Ecological impact": '-1',
                            "Financial impact": '-1',
                            "Human health impact": '-1',
                            "Political impact": '-1',
                            "isError": "true"
                        }
                        writer.writerow(result)
                print(f'Finished {count}, id: {id}')
                count += 1


API_KEY = ""
input_csv = "final_query.csv"  # Replace with your CSV file path
output_csv = "mixtral_1300_output.csv"  # Replace with your desired JSON file path

process_csv(input_csv, output_csv)
print(f"Results written to {output_csv}")