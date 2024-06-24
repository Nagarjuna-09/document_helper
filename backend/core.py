# for retrieval chain (retrieving embeddings from Pinecone)

import os
from langchain import hub
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

#***** Writing chain to combine user prompt + retrieved docs/embeddings + instructions to send to llm to get response*********
# creating retrieval chain

load_dotenv()

# This method of retrieval is same as when we built RAG with Pinecone and FAISS in previous projects
# def run_llm_without_memory(query: str):
#     llm = ChatOpenAI()
#     embedding_method = OpenAIEmbeddings()
#     retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
#     vectorstore = PineconeVectorStore(
#         index_name=os.environ["INDEX_NAME"], embedding=embedding_method
#     )
#
#     combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
#
#     retrieval_chain = create_retrieval_chain(
#         retriever=vectorstore.as_retriever(),  # vector db to use
#         combine_docs_chain=combine_docs_chain  # docs to use
#     )
#
#     result = retrieval_chain.invoke(input={"input": query})
#
#     print(result)
#     print(result["answer"])
#
# run_llm_without_memory(query="What is langchain?")


##### You can also create retrieval chain like below with memory of previous coversation

from typing import Any, Dict, List
from langchain.chains import ConversationalRetrievalChain
from langchain_pinecone import PineconeVectorStore
INDEX_NAME = "langchain-doc-index"

def run_llm_with_memory(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = OpenAIEmbeddings()
    docsearch = PineconeVectorStore(embedding=embeddings, index_name=INDEX_NAME)

    chat = ChatOpenAI(
        verbose=True,
        temperature=0,
    )

    # This augments with prompt with the chat history
    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )
    return qa.invoke({"question": query, "chat_history": chat_history})

# res = run_llm_with_memory(query="How to read from pdf files in langchain?")
#
# print(res["answer"])