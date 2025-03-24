from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import streamlit as st
import time
from langchain_text_splitters import CharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain_unstructured import UnstructuredLoader
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler
import os
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


llm = ChatOpenAI(
    temperature=0.5,
    streaming=True,
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
     Answer your questions using the following context.
     If you don't know the answer, just say you don't know. DO NOT make anything up.
     
     Context: {context}
     """,
        ),
        ("human", "{question}"),
    ]
)



def embedded_file(file):
    file_content = file.read()
    file_name = os.path.basename(file.name)
    print("file_name : ", file_name)
    file_path = f"./.cache/files/{file_name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    model_name = "jhgan/ko-sbert-sts"
    embedding_name = model_name.split("/")[-1]  # "ko-sbert-sts"
    embedding_cache_dir = f"./.cache/embeddings/{file_name}_{embedding_name}"
    # faiss_index_file = os.path.join(embedding_cache_dir, "index.faiss")
    # faiss_store_file = os.path.join(embedding_cache_dir, "index.pkl")

    # if os.path.exists(faiss_index_file) and os.path.exists(faiss_store_file):
    #     vectorstore = FAISS.load_local(embedding_cache_dir, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    #     return vectorstore.as_retriever()

    cache_dir = LocalFileStore(embedding_cache_dir)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    


    vectorstore.save_local(embedding_cache_dir)
    

    return vectorstore.as_retriever()


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)



file = ".cache/files/test.pdf"
with open(file, "rb") as f:
    retriever = embedded_file(f)
print("______", retriever)
chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
)
# question = "can you explain what fcm(free cantilever method) is in korean?"
question = "F.C.M(free cantilever method) 에 대해 설명해줘"
# retriever에서 질문과 관련된 문서 가져오기
relevant_docs = retriever.get_relevant_documents(question)

# 가져온 문서의 내용을 출력
for i, doc in enumerate(relevant_docs):
    print(f"Document {i+1}:\n{doc.page_content}\n")
response = chain.invoke(question)

print("답변 : ",response)



