from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import streamlit as st
import time
from langchain_text_splitters import CharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain_unstructured import UnstructuredLoader
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
st.set_page_config(page_title="DocumentGPT", page_icon="📃")

st.title("DocumentGPT")

st.markdown(
    """
    Welcome!
    
    Use this chatbot to ask questions to an AI about your files!
    """
)

file = st.file_uploader(
    "Upload a .txt .pdf or .docx file",
    type=["pdf", "txt", "docx"],
)

def embedded_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
        
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredLoader("./files/chapter_one.txt")
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return retriever

if file:
    retriever = embedded_file(file)
    s = retriever.invoke("winston")
    s