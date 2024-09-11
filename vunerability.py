import os
import json
from pathlib import Path
from Tools.clean_data import process_text
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
from transformers import pipeline
import time

# model = configure_model()
client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY")
)




def split_txtChunks(txt: str, chunk_size: int) -> list:
    return [txt[i:i+chunk_size] for i in range(0, len(txt), chunk_size)]


def generate_prompt(topics, text_chunk):
    instructions = """
    Please read the following text carefully. It contains historical accounts and may discuss various events. 
    Your task is to identify and summarize any mentions of extreme or disruptive weather events and their impacts on society.
    Focus on two key areas:
    1. Societal vulnerabilities: Detail how the community or infrastructure was vulnerable due to weather events. Include specific examples such as properties damaged, people affected, and the immediate consequences.
    2. Resilience and responses: Describe the actions taken by individuals, communities, or governments to respond to these events. Highlight strategies for managing or mitigating the impact, including emergency responses and long-term planning.
    It's important that your response only includes information explicitly related to weather events. If no such information is found, please indicate so.

    Additionally, indicate whether the text discusses:
    - Infrastructural impact
    - Agricultural impact
    - Ecological impact
    - Economic impact
    Each should be marked as "True" or "False" based on the presence of relevant information.
    """
    
    example_output = {
        "Vulnerabilities": "e.g., The flood resulted in over 200 homes being destroyed, leaving many without shelter.",
        "Resilience and Responses": "e.g., The local government deployed emergency response teams and set up temporary shelters for the displaced residents.",
        "Infrastructural impact": "False",
        "Agricultural impact": "True",
        "Ecological impact": "True",
        "Economic impact": "True",
        "societal impact": "True",
        "human health impact": "True"
    }
    formatted_example = json.dumps(example_output, indent=4)

    prompt = f"""
    {instructions}
    If there are no relevant info found regarding the topic, return "" for that topic.
    Example of expected output based on the above definitions:
    {formatted_example}

    --- Begin Text ---
    {text_chunk}
    --- End Text ---
    """

    return prompt


def make_request(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Please follow the instructions provided and generate a response."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=2000  # Adjust tokens as needed to ensure complete responses
    )
    return response

def filter_weather_related_chunks(texts, keywords):
    """
    Filters and returns text chunks that contain specified weather-related keywords.

    Args:
    texts (list of str): The list of text chunks to be filtered.
    keywords (list of str): Weather-related keywords used for filtering the texts.

    Returns:
    list of str: A list of text chunks that contain one or more of the specified keywords.
    """
    
    filtered_texts = []

    for text in texts:
    
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in keywords):
            # If a keyword is found, append the text chunk to the filtered list
            filtered_texts.append(text)
    return filtered_texts

def setup_summarizer(model_name="facebook/bart-large-cnn"):
    # Load the summarization pipeline with your chosen model
    summarizer = pipeline("summarization", model=model_name) #If you do not pick a model, it will pick up for you automatically according to the description
    return summarizer


def analyze_topics(file: str, topics: list) -> list:
    new_file = process_text(file)
    texts = split_txtChunks(new_file, 3000)
    # filtered_texts = filter_weather_related_chunks(texts, keywords)
    # print(filtered_texts)
    topic_info = []
    summarizer = setup_summarizer()

    for chunk in texts:
        try:
            summary = summarizer(chunk, max_length=250, min_length=150, do_sample=False)
            processed_summary = summary[0]['summary_text']
        except Exception as e:
            print(f"Error summarizing text: {e}")
            processed_summary = ""
        print("------------------------------------------------------------------------")
        print(processed_summary)
        prompt = generate_prompt(topics, processed_summary)
        try:
            response = make_request(prompt)
            # Adjust the following line to match how the OpenAI API returns the response
            if response.choices[0].message.content.strip():
                content = response.choices[0].message.content.strip()
                topic_info.append(json.loads(content))
            else:
                print("Received an empty response for the chunk:", chunk)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}, in response: {response}")
        except Exception as e:
            print(f"An error occurred: {e}")

    return {"info": topic_info}
 
if __name__ =="__main__":
    file = "blog/snow_English_historical_ML_corpus.txt"
    topics = [
        "weather",
    ]
    print(analyze_topics(file,topics))

