from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from Tools.clean_data import obtain_text

def build_index():
    # Set embeddings
    embd = OpenAIEmbeddings()

    # Docs to index
    texts = obtain_text("/Users/yonganyu/Desktop/vulnerability-Prediction-GEOG-research-/blog/cold _English_historical_ML_corpus.txt")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20,
    )
    docs = text_splitter.split_text(texts)

    # Index
    index = Chroma.from_texts(docs, collection_name = "Geog_research", embedding=embd)
    retriver = index.as_retriever()
    return index, retriver


from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

# Define your LLM
def build_llm(model: Literal["gpt-3.5-turbo", "gpt-4"] = "gpt-3.5-turbo"):
    