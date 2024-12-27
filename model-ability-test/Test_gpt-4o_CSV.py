from openai import OpenAI
import json
import csv

# Set your OpenAI API key

def parse_classification_response(response_text):
    # Initialize the result dictionary
    # Change 1/0 when impacts are added
    result = {
        'Infrastructural impact': 0,
        'Agricultural impact': 0,
        'Ecological impact': 0,
        'Economic impact': 0,
        'Societal impact': 0,
        'Human Health impact': 0,
        'Political impact': 0
    }
    for line in response_text.split('\n'):
        if ':' in line:
            category, value = line.split(':')
            category = category.strip()
            value = value.strip().lower()
            # Convert true/false to 1/0
            result[category] = 1 if value == 'true' else 0
    return result

def classify_text(input_text):
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

    Output format:
    Infrastructural: true/false
    Agricultural: true/false
    Ecological: true/false
    Financial: true/false
    Health: true/false
    Political: true/false
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an assistant specialized in analyzing historical weather event impacts from historical newspaper."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def process_csv_to_csv(input_csv, output_csv):
    headers = [
        'Date', 'Type', 'Model_Type',
        'Infrastructural impact', 'Agricultural impact',
        'Ecological impact', 'Economic impact',
        'Societal impact', 'Human Health impact',
        'Political impact'
    ]
    
    with open(output_csv, mode='w', encoding='utf-8', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=headers)
        writer.writeheader()
        
        with open(input_csv, mode='r', encoding='utf-8') as input_file:
            reader = csv.DictReader(input_file)
            
            for row in reader:
                if row.get("Remove") == "1":
                    continue

                # Get classification results
                classification_response = classify_text(row.get("Text", ""))
                classification_dict = parse_classification_response(classification_response)
                
                # Create output row with exact format
                output_row = {
                    'Date': row.get("Date", ""),
                    'Type': row.get("Type", ""),
                    'Model_Type': 'Classification-GPT4',
                    'Infrastructural impact': classification_dict.get('Infrastructural', 0),
                    'Agricultural impact': classification_dict.get('Agricultural', 0),
                    'Ecological impact': classification_dict.get('Ecological', 0),
                    'Economic impact': classification_dict.get('Financial', 0),
                    'Societal impact': classification_dict.get('Societal', 0),
                    'Human Health impact': classification_dict.get('Health', 0),
                    'Political impact': classification_dict.get('Political', 0)
                }
                
                writer.writerow(output_row)

# def test_single(text):
#     classification_response = classify_text(text)
#     return parse_classification_response(classification_response)

input_csv = "blog/corrected/test.csv"
output_csv = "final_structured.csv"

process_csv_to_csv(input_csv, output_csv)
print(f"Results written to {output_csv}")

# test_result = test_single("your test text here")
# print("Test result:", test_result)