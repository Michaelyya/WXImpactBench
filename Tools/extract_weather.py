import pandas as pd
from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from dotenv import load_dotenv
load_dotenv
import time


def extract_weather_content(text):
    prompt = """Please extract all weather-related content from the following text, following these specific guidelines:
        1. Preserve all mentions of:
        - Weather events (storms, floods, snowfall, etc.)
        - Weather impacts (damage to property, transportation disruptions, casualties)
        - Weather conditions (temperature changes, wind conditions)
        - Weather-related disruptions to business or daily life
        - everything you think it is realted to a weather event

        2. Maintain the original context and temporal information (dates, locations, sequence of events)

        3. Keep the original language and phrasing for weather-related content

        4. Remove:
        - Advertisements
        - Unrelated business news
        - Sports events (unless directly impacted by weather)
        - Market reports (unless directly weather-impacted)
        - Social events and announcements
        - General news unrelated to weather

        5. Do not:
        - Summarize or paraphrase the weather content
        - Change dates or locations
        - Alter any weather-related measurements or statistics
        - Modify descriptions of weather impacts

        Text to process:
        {text}

        Return only the extracted weather-related content while maintaining original phrasing and context."""

    try:
        response = client.chat.completions.create(model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts weather-related content while maintaining original context and language."},
            {"role": "user", "content": prompt.format(text=text)}
        ],
        temperature=0.1)
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error processing text: {e}")
        return ""

df = pd.read_csv('Selected Query - 工作表2.csv', usecols=['Date', 'Type', 'Weather', 'cleaned_text'])

output_df = pd.DataFrame({
    'Date': df['Date'],
    'Type': df['Type'],
    'Weather': df['Weather']
})
cleaned_contents = []
for text in df['cleaned_text']:
    processed_text = extract_weather_content(text)
    print(processed_text)
    cleaned_contents.append(processed_text)
    time.sleep(1)

output_df['cleaned_weather_content'] = cleaned_contents
output_df.to_csv('weather_events_cleaned__.csv', index=False)