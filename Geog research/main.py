import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeStore 
from langchain.chains import RetrievalQA
from pinecone import Pinecone, ServerlessSpec

api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )

index_name = "geography"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name, 
        dimension=512,  # Adjust the dimension to match your embeddings
        metric='euclidean',  # Choose the metric that suits your use case
        spec=ServerlessSpec(
            cloud='aws',  # or 'gcp' depending on your preference or requirements
            region='us-west-2'  # Choose the region closest to you or your users
        )
    )

if __name__=="__main__":
    print("hello")
    loader = TextLoader("/Users/yonganyu/Desktop/NLP study/Embedding project/blog/The Man who Invented the Computer.txt")
    document = loader.load()

    TextSplitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    text = TextSplitter.split_documents(document)
    print(len(text))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = PineconeStore.from_documents(text, embeddings, index_name=index_name)

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

    query = "Summarize the person who invented the computer"
    result = qa ({"query":query})
    print(result)