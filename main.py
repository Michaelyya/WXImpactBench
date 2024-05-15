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
from Agent.csvAgent import create_csv_agent
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from Tools.model import configure_model
from google.api_core import exceptions
from Tools.clean_data import process_text



# api_key = os.environ.get("PINECONE_API_KEY")
# pc = Pinecone(
#         api_key=os.environ.get("PINECONE_API_KEY")
#     )

model = configure_model()
# client = OpenAI(
#   api_key=os.environ.get("OPENAI_API_KEY")
# )


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


def search(query, texts, text_embeddings):
    query_embedding = OpenAIEmbeddings.embed_query(query)
    similarities = [np.dot(query_embedding, text_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(text_emb)) for text_emb in text_embeddings]
    max_index = np.argmax(similarities)
    return texts[max_index]


def generate_prompt(topics, text_chunk):
    example_json = {
        topic: f"Relevant information about {topic.lower()}." for topic in topics
    }

    formatted_json_example = json.dumps(example_json, indent=4)

    prompt = f"""
    Please analyze the following text and categorize the information according to these topics: {', '.join(topics)}.
    If there are no relevant info found regarding the topic, return "" for that topic.
    For each topic, format the information into a JSON object as shown in this example:
    {formatted_json_example}
    Don't include the JSON header.

    Text to analyze:
    {text_chunk}
    """
    return prompt

def make_request(prompt):
    return model.generate_content(prompt)

def analyze_topics(file:str, topics: list) -> list:
    new_file = process_text(file)
    texts = split_txtChunks(new_file, 1000)
    topic_info = []
    for chunk in texts:
        prompt = generate_prompt(topics, chunk)
        try:
            response = make_request(prompt)
            if response.text.strip():  
                topic_info.append(json.loads(response.text))
        except exceptions.InternalServerError as error:
            print(f"Failed to process due to server error: {error}")
    return {"info": topic_info}

if __name__ =="__main__":
    file = "blog/cold _English_historical_ML_corpus.txt"
    topics = [
        "weather",
        "food",
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

