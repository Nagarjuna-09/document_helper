from dotenv import load_dotenv

load_dotenv()

import os

from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


INDEX_NAME = "langchain-doc-index"


# **** load documents ***********
def ingest_docs():
    loader = ReadTheDocsLoader(
        path="langchain-docs/",
        encoding = "utf-8"
    )

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")
    return raw_documents


def text_splitter(raw_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators = ["\n\n", "\n", " ", ""])
    chunks_of_documents = text_splitter.split_documents(raw_docs)
    print(f"Created {len(chunks_of_documents)} chunks/docs")

    # --------------
    # Each doc_chunk is like:
    # {
    # congif=<class cjhdgsjg>,
    # metadata={'source': langchain-docs/langchain.readthedocs.io/latest/index.html},
    # pagecontent="the langchain documentation contains ..."
    # }
    # Note: If you try to remove langchain-docs and include https:// before source it becomes the source url, which can be shared to the user.

    # Editing the source column to include https:// for every doc
    for chunk in chunks_of_documents:
        new_url = chunk.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        chunk.metadata.update({"source": new_url})

    print(f"Going to insert {len(chunks_of_documents)} chunks/docs in Pinecone")
    return chunks_of_documents

def load_into_pinecone(chunks, embedding_to_use):
    PineconeVectorStore.from_documents(
        chunks, embedding = embedding_to_use, index_name=os.environ["INDEX_NAME"]
    )
    print("Loading chunks to vectorstore complete")

# Reading documents
raw_docs = ingest_docs()

#Splitting it
chunks_of_documents = text_splitter(raw_docs)

# Converting each chunk into embedding using openai embedding and laoding them into Pinecone
embedding_to_use = OpenAIEmbeddings()

# loading into pinecone
load_into_pinecone(chunks_of_documents, embedding_to_use)