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
load_dotenv()

class ChatCallbackHandler(BaseCallbackHandler):

    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
        
    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")
        
    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

llm = ChatOpenAI(
    temperature=0.5,
    streaming=True,
    callbacks=[ChatCallbackHandler()]
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
     Answer your questions using Only the following context.
     If you don't know the answer, just say you don't know. DO NOT make anything up.
     Your response should always be in Korean.
     
     Context: {context}
     """,
        ),
        ("human", "{question}"),
    ]
)



def embedded_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    embedding_cache_dir = f"./.cache/embeddings/{file.name}"
    faiss_index_file = os.path.join(embedding_cache_dir, "index.faiss")
    faiss_store_file = os.path.join(embedding_cache_dir, "index.pkl")

    if os.path.exists(faiss_index_file) and os.path.exists(faiss_store_file):
        vectorstore = FAISS.load_local(embedding_cache_dir, OpenAIEmbeddings())
        return vectorstore.as_retriever()

    cache_dir = LocalFileStore(embedding_cache_dir)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)

    vectorstore.save_local(embedding_cache_dir)

    return vectorstore.as_retriever()


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)



file = "fullstackGPT/files/F.C.M 교량공사 안전보건작업 지침.pdf"
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

response = chain.invoke("FCM공법에 대해 설명해봐")
       
print(response)
