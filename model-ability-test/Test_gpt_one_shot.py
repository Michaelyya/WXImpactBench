from openai import OpenAI
import csv
import os
import dotenv
import re

dotenv.load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

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
    return {key: 1 if value.lower() == "true" else 0 for key, value in answers}

def inference(input_text, model_name):
    prompt = f"""
    Given the following historical newspaper text:
    "{input_text}"

    Analyze the text and provide a binary classification (respond ONLY with 'true' or 'false') for each impact category based on explicit mentions in the text. Follow these specific guidelines:
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

    Remeber only output format as:
    Infrastructural: true/false
    Agricultural: true/false
    Ecological: true/false
    Financial: true/false
    Health: true/false
    Political: true/false
    
    if this is not weather-related event or you cannot find any impact, return all impact category as false

    example:
    text: "Collision and loss of life. Calhoun, March 20. During a heavy wind the other night the oyster pungie Jasper and Industry collided at the mouth of the Wicomico River, Virginia. The Jasper sank. Boats were unable to reach her in the darkness, and in the morning the captain and two men were found lashed to the rigging, one frozen to death. Four others of the crew had dropped off during the night and were drowned. Business failure. New York, March 20. James Pendergast, ship broker, has assigned; liabilities, $75,000. Schuloss Heilbronner, woolens, have also assigned; liabilities, $50,000. Happenings abroad. St. Louis, March 20. Leading jewelers have been notified from New York that an organized band of daring thieves is about to raid Western cities. Claiming fortune. Des Moines, March 20. Reports from this section show that the apple orchards have been killed by the severe winter. One farmer lost two hundred trees."
    output: "Infrastructural: true, Agricultural: false, Ecological: true, Financial: true, Health: true, Political: false"

    ***Rmember only output true or false***
    """

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
        return result, extract_answer(result)
    except Exception as e:
        error_message = f"Error: {e}"
        return error_message, {}

def process_csv(input_csv, output_dir, models):
    for model_name in models:
        model_file_name = model_name.replace("-", "_") + "_oneshot_short.csv"
        output_csv = os.path.join(output_dir, model_file_name)
        headers = [
            "ID", "Date", "Type", "Model_Type",
            "Infrastructural impact", "Agricultural impact",
            "Ecological impact", "Financial impact",
            "Health impact", "Political impact"
        ]

        with open(output_csv, mode='w', encoding='utf-8', newline='') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=headers)
            writer.writeheader()

            with open(input_csv, mode='r', encoding='utf-8') as input_file:
                reader = csv.DictReader(input_file)
                for count, row in enumerate(reader, start=1):
                    original_text = row.get("Text", "")
                    date = row.get("Date", "")
                    type_row = row.get("Type", "")
                    id_row = row.get("ID", "")
                    result = inference(original_text, model_name)

                    if not isinstance(result, tuple) or len(result) != 2:
                        print(f"Skipping row {count} for model {model_name} due to unexpected response format.")
                        continue

                    classification_text, extracted_response = result
                    result = {
                        "ID": id_row,
                        "Date": date,
                        "Type": type_row,
                        "Model_Type": model_name,
                        "Infrastructural impact": extracted_response.get("Infrastructural", 0),
                        "Agricultural impact": extracted_response.get("Agricultural", 0),
                        "Ecological impact": extracted_response.get("Ecological", 0),
                        "Financial impact": extracted_response.get("Financial", 0),
                        "Health impact": extracted_response.get("Health", 0),
                        "Political impact": extracted_response.get("Political", 0),
                    }

                    writer.writerow(result)
                    print(f"Processed row {count} for model {model_name}")


input_csv = "./datasets/context_data/short.csv"
output_dir = "./datasets/context_data/"
models = ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]

process_csv(input_csv, output_dir, models)
print("Done")

# test_result = test_single("your test text here")
# print("Test result:", test_result)
