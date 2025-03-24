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
from langchain_huggingface import HuggingFaceEmbeddings
import os
load_dotenv()

st.set_page_config(page_title="PrivateGPT", page_icon="📃")

# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

st.title("PrivateGPT")

st.markdown(
    """
    Welcome!
    
    Use this chatbot to ask questions to an AI about your files!
    
    Upload your files on the side bar
    """
)
with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

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
     Answer your questions using the following context.
     If you don't know the answer, just say you don't know. DO NOT make anything up.
     
     Context: {context}
     """,
        ),
        ("human", "{question}"),
    ]
)


# file check decorator in streamlit
@st.cache_resource(show_spinner="Embedding File ...")
def embedded_file(file):
    file_content = file.read()
    file_path = f"./.cache/private_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    model_name = "jhgan/ko-sbert-sts"
    
    cache_dir = LocalFileStore(f"./.cache/{model_name}_embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)

def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})
    
def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)


if file:
    retriever = embedded_file(file)
    print("______", retriever)
    send_message("I'm ready! Ask me something", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        # ✅ context 디버깅용
        relevant_docs = retriever.get_relevant_documents(message)
        context_str = format_docs(relevant_docs)

        with st.expander("🔍 디버깅: LLM에게 전달된 context 보기"):
            st.markdown(context_str[:3000] + ("..." if len(context_str) > 3000 else ""))

        print("🔍 [DEBUG] Context 전달 내용:\n", context_str[:1000])  # 콘솔 출력도 같이
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            response = chain.invoke(message)
        

else:
    st.session_state["messages"] = []

