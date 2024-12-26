from openai import OpenAI
import json
import csv

def classify_text(input_text):
    # Construct the prompt
    prompt = f"""
    Given the following text:
    "{input_text}"

    Classify the text for the presence of the following features. For each feature, return only 'true' or 'false' based on whether it is explicitly or implicitly mentioned in the text.

    1. frastructural impact: Analyze the text for any mentions of damage to or disruption of physical infrastructure due to the weather event.
    Damage to buildings, roads, bridges, or other structures 
    Disruption to transportation systems (e.g., train cancellations, road closures)
    Impacts on utility services (e.g., power outages, water supply issues, coal, wood supply)
    Communication network failures
    Damage to industrial facilities
    Note: if not weather-related event, return false

    2. Agricultural impact: Agricultural impact: Examine the text for any indications of weather-related effects on agriculture, forestry, or livestock management (e.g., cows, chickens, pigs).
    Changes in crop or animal yields (positive or negative)
    Damage to crops, timber or livestock
    Alterations in farming practices or schedules
    Impacts on food production or supply
    Impacts on farm equipment (e.g., saws, tractors, horses)
    Effects on soil conditions, water availability or other material (e.g., seedlings, fertilizer, eggs) for agriculture, forestry, or livestock
    Note: if not weather-related event, return false

    3. Ecological impact: Review the text for any mentions of effects on the natural environment due to the weather event.
    Changes in local environment, ecosystem or biodiversity
    Impacts on wildlife populations or behavior
    Effects on plant life or vegetation patterns (non-agricultural)
    Alterations to natural habitats (e.g., forests, swamps)
    Changes in water bodies (e.g., river levels, lake conditions)
    Note: if not weather-related event, return false

    4. Economic impact: Analyze the text for indications of economic consequences resulting from the weather event
    Direct financial losses (e.g., property damage, crop destruction)
    Business disruptions or closures
    Changes in market prices or demand for certain goods
    Impacts on tourism or local economies
    Insurance claims or economic relief measures
    Note: if not weather-related event, return false

    5. Societal impact: Examine the text for any mentions of how the weather event affects society and daily life.
    Changes in daily routines or behaviors
    Effects on housing (e.g., loss of home, room, apartment)
    Effects on transportation (e.g., disruption to bus services, carriages)
    Impacts on community activities or gatherings
    Effects on education (e.g., school closures)
    Changes in social interactions or community support
    Alterations to cultural or religious practice (e.g., shutdown of church services, impacts on nuns)
    Note: if not weather-related event, return false

    6.Human Health impact: Review the text for any indications of how the weather event affects human health, both physical and mental.
    Direct injuries or fatalities caused by the weather
    Increased risk of illnesses (e.g., heat-related illnesses, water-borne diseases)
    Mental health effects (e.g., stress, anxiety, depression)
    Impacts on healthcare services or access (e.g., impacts on ambulances)
    Reference to long-term health consequences
    Note: if not weather-related event, return false

    7. Political impact: Analyze the text for any indications of political consequences or responses to the weather event.
    Government decisions or policy changes in response to the event (e.g., about flood responses)
    Shifts in public opinion or political discourse
    Impacts on election processes or outcomes (e.g., inability to clear snow, ice jams)
    International relations or aid responses
    Debates about disaster preparedness
    Note: if not weather-related event, return false

    Note: Return 'false' for any impact category that is either not present in the text or not related to weather events.

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
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an assistant specialized in text classification."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

def process_csv_to_csv(input_csv, output_csv):
    count = 1
    
    # Open the output CSV file
    with open(output_csv, mode='w', encoding='utf-8', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        # Write the header
        csv_writer.writerow(['Date', 'Type', 'Classification'])
        
        # Read input CSV
        with open(input_csv, mode='r', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            
            for row in csv_reader:
                if row.get("Remove") == "1":
                    continue  # Skip rows where Remove column is 1

                text = row.get("Text", "")
                date = row.get("Date", "")
                event_type = row.get("Type", "")
                
                # Get classification
                classification = classify_text(text)
                
                # Write to CSV
                csv_writer.writerow([date, event_type, classification])
                
                print(f'Finished {count}')
                count = count + 1
                
def process_csv_to_json(input_csv, output_json):
    count = 1
    results = []

    with open(input_csv, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            if row.get("Remove") == "1":
                continue 
            text = row.get("Text", "")
            date = row.get("Date", "")
            event_type = row.get("Type", "")
            weather = row.get("Weather", "")

            classification = classify_text(text)

            results.append({
                "Date": date,
                "Type": event_type,
                "Weather": weather,
                "Classification": classification
            })
            print(f'Finished {count}')
            count = count + 1
            if count >=5:
                break

    with open(output_json, mode='w', encoding='utf-8') as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=4)                

# Example usage
input_csv = "blog/corrected/Final Query selected - selected_query.csv"  # Replace with your CSV file path
output_csv = "final.csv"  # Replace with your desired CSV file path

process_csv_to_csv(input_csv, output_csv)
print(f"Results written to {output_csv}")
print(classify_text("asked if the Board thought the unpaid capital could be collected. The Cashier replied that under the Act MONTREAL, FRIDAY, JANUARY 16, 1880, the stock of all those who did not pay up their calls could be confiscated. Mr. Baebsac asked on what basis was the valuation of the real estate made. The Cashier said the property owned by the Bank was on St Joseph and Seigneur streets, and gave a return of 5 per cent. The valuation was its actual worth. Mr. Babeiac entered into a lengthy charge against the management, contending that the Bank had not sufficient capital to do a profitable business, and that the shareholders had not been sufficiently consulted. A general meeting, he argued, should have been called after the Paquet defalcation, and also when the fusion with the Jacques Cartier Bank was under consideration. Had the latter been accomplished the result, he contended, would have been ruinous to the bank. Hon. Mr. Thibaudac replied to the charges made by Mr. Barbeau. The argument of the latter in relation to the circulation and reserve fund of the bank was absurd. As to calling a general meeting of the shareholders after the defalcation of Mr. Paquet, that would destroy the credit of the bank, for the shareholders under the excitement of the moment when rumors were afloat detrimental to various monetary institutions, would have caused precipitate action. As far as the fusion with the Jacques Cartier Bank was concerned, no measure to that end could have been passed without the consent of the shareholders and without an Act of Parliament to authorize it. It was not customary in the consideration of such questions for the directors to call a meeting until they had the scheme fully digested. Mr. Bxiqcx stated that if the fusion with the Jacques Cartier Bank was not proposed to the shareholders, it was because the directors of the Hochelaga Bank did not consider the terms offered by the Jacques Cartier Bank satisfactory. As to Mr. Barbeau's complaint that the defalcation of Mr. Paquet was owing to insufficient precautions having been taken by the Jacques Cartier Bank, he might say that the Jacques Cartier Bank's system in this respect had been the same as those then followed by other banks, and that the City and District Savings Bank had adopted further precautions in relation to its officers since Mr. Paquet's defalcation had been discovered. The Hochelaga Bank had also taken greater precautions and measures since that incident. Aid. Labebgi said that he had entered this meeting with views opposed to the management of the Hochelaga Bank, but since Mr. Barbeau's speech, he had altered his views and considered that the directors had done all in their power to provide against losses. He was, therefore, fully satisfied with the conduct of the directors and considered they should be re-elected. After some further discussion, the President made a speech which fairly carried away the audience and enlisted their sympathies and support on behalf of the directors. He sketched a history of the Bank, which was started in 1873"))