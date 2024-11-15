from openai import OpenAI
import json
import csv

# Set your OpenAI API key
client = OpenAI(
    api_key="")
def classify_text(input_text):
    # Construct the prompt
    prompt = f"""
Given the following text:
"{input_text}"

Classify the text for the presence of the following features. For each feature, return only 'true' or 'false' based on whether it is explicitly or implicitly mentioned in the text.

1. **Infrastructural impact**: 
   - Damage to buildings, roads, bridges, or other structures
   - Disruption to transportation systems (e.g., train cancellations, road closures)
   - Impacts on utility services (e.g., power outages, water supply issues, coal, wood supply)
   - Communication network failures
   - Damage to industrial facilities

2. **Agricultural impact**: 
   - Changes in crop or animal yields (positive or negative)
   - Damage to crops, timber, or livestock
   - Alterations in farming practices or schedules
   - Impacts on food production or supply
   - Impacts on farm equipment (e.g., saws, tractors, horses)
   - Effects on soil conditions, water availability, or other material (e.g., seedlings, fertilizer, eggs) for agriculture, forestry, or livestock

3. **Ecological impact**: 
   - Changes in local environment, ecosystem, or biodiversity
   - Impacts on wildlife populations or behavior
   - Effects on plant life or vegetation patterns (non-agricultural)
   - Alterations to natural habitats (e.g., forests, swamps)
   - Changes in water bodies (e.g., river levels, lake conditions)

4. **Economic impact**: 
   - Direct financial losses (e.g., property damage, crop destruction)
   - Business disruptions or closures
   - Changes in market prices or demand for certain goods
   - Impacts on tourism or local economies
   - Insurance claims or economic relief measures

5. **Societal impact**: 
   - Changes in daily routines or behaviors
   - Effects on housing (e.g., loss of home, room, apartment)
   - Effects on transportation (e.g., disruption to bus services, carriages)
   - Impacts on community activities or gatherings
   - Effects on education (e.g., school closures)
   - Changes in social interactions or community support
   - Alterations to cultural or religious practices (e.g., shutdown of church services, impacts on nuns)

6. **Human Health impact**: 
   - Direct injuries or fatalities caused by the weather
   - Increased risk of illnesses (e.g., heat-related illnesses, water-borne diseases)
   - Mental health effects (e.g., stress, anxiety, depression)
   - Impacts on healthcare services or access (e.g., impacts on ambulances)
   - Reference to long-term health consequences

7. **Political impact**: 
   - Government decisions or policy changes in response to the event (e.g., about flood responses)
   - Shifts in public opinion or political discourse
   - Impacts on election processes or outcomes (e.g., inability to clear snow, ice jams)
   - International relations or aid responses
   - Debates about disaster preparedness

Respond in the format:
Infrastructural impact: true/false
Agricultural impact: true/false
Ecological impact: true/false
Economic impact: true/false
Societal impact: true/false
Human Health impact: true/false
Political impact: true/false
"""

    # Call GPT API
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an assistant specialized in text classification."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"


def process_csv_to_json(input_csv, output_json):
    count = 1
    results = []

    with open(input_csv, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            if row.get("Remove") == "1":
                continue  # Skip rows where Remove column is 1

            text = row.get("Text", "")
            date = row.get("Date", "")
            event_type = row.get("Type", "")
            weather = row.get("Weather", "")

            # Classify the text
            classification = classify_text(text)

            # Append the result
            results.append({
                "Date": date,
                "Type": event_type,
                "Weather": weather,
                "Classification": classification
            })
            print(f'Finished {count}')
            count = count + 1

    # Write to JSON file
    with open(output_json, mode='w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)


# Example usage
input_csv = "selected_query.csv"  # Replace with your CSV file path
output_json = "output.json"  # Replace with your desired JSON file path

process_csv_to_json(input_csv, output_json)
print(f"Results written to {output_json}")
