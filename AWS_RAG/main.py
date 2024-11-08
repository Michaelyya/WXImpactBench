import os
from langchain.agents import AgentType
import numpy as np
import json
# from langchain_experimental.agents import create_csv_agent
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings
# from langchain_community.vectorstores import Pinecone as PineconeStore 
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec
# from Agent.csvAgent import create_csv_agent
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from Tools.clean_data import process_text
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import time



# api_key = os.environ.get("PINECONE_API_KEY")
# pc = Pinecone(
#         api_key=os.environ.get("PINECONE_API_KEY")
#     )

# model = configure_model()
client = OpenAI(
  api_key=os.environ.get("OPENAI_API_KEY")
)


# index_name = "geography"
# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name, 
#         dimension=512,  # Adjust the dimension to match your embeddings
#         metric='euclidean',  # Choose the metric that suits your use case
#         spec=ServerlessSpec(
#             cloud='aws',  # or 'gcp' depending on your preference or requirements
#             region='us-west-2'  # Choose the region closest to you or your users
#         )
#     )



def split_txtChunks(txt: str, chunk_size: int) -> list:
    return [txt[i:i+chunk_size] for i in range(0, len(txt), chunk_size)]


# def search(query, texts, text_embeddings):
#     query_embedding = OpenAIEmbeddings.embed_query(query)
#     similarities = [np.dot(query_embedding, text_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(text_emb)) for text_emb in text_embeddings]
#     max_index = np.argmax(similarities)
#     return texts[max_index]


def generate_prompt(topics, text_chunk):
    instructions = """
    Please read the following text carefully. It contains historical accounts and may discuss various events. 
    Your task is to identify and summarize any mentions of extreme or disruptive weather events and their impacts on society.
    Focus on two key areas:
    1. Societal vulnerabilities: Detail how the community or infrastructure was vulnerable due to weather events. Include specific examples such as properties damaged, people affected, and the immediate consequences.
    2. Resilience and responses: Describe the actions taken by individuals, communities, or governments to respond to these events. Highlight strategies for managing or mitigating the impact, including emergency responses and long-term planning.
    
    It's important that your response only includes information explicitly related to weather events. If no such information is found, please indicate so.
    """
    
    example_output = {
        "Vulnerabilities": "e.g., The flood resulted in over 200 homes being destroyed, leaving many without shelter.",
        "Resilience and Responses": "e.g., The local government deployed emergency response teams and set up temporary shelters for the displaced residents."
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


def analyze_topics(file: str, topics: list) -> list:
    new_file = process_text(file)
    texts = split_txtChunks(new_file, 2000)
    # filtered_texts = filter_weather_related_chunks(texts, keywords)
    # print(filtered_texts)
    topic_info = []

    for chunk in texts:
        prompt = generate_prompt(topics, chunk)
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
    file = "blog/cold _English_historical_ML_corpus.txt"
    # weather_keywords = ['weather', 'storm', 'rain', 'floods','snow', 'temperature', 'climate']
    topics = [
        "weather",
    ]
    print(analyze_topics(file,topics))

    # embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    # text_embeddings = [embeddings.embed_query(text) for text in texts]
    # query = "Please give me some summarization about the extreme weather information"
    # result = search (query, texts, text_embeddings)


    # docsearch = PineconeStore.from_documents(text, embeddings, index_name=index_name)
    # qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())



    # csv_agent = create_csv_agent(
    #     llm=ChatOpenAI(temperature=0, model="gpt-4"),
    #     path="/Users/yonganyu/Desktop/GEOG_research/cold _English_historical_ML_corpus.csv",
    #     verbose=True,
    #     agent_Type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # )
    # csv_agent.run("how many rows in this csv file")

