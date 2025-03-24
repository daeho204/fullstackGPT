from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.cache import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub
import os
from langchain_huggingface import HuggingFaceEmbeddings

# 웹에서 문서 로드
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# 캐시 저장소 설정
cache_dir = LocalFileStore(".cache/embeddings/")
embeddings = OpenAIEmbeddings()
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

# 벡터스토어 (FAISS) 저장 및 검색
vectorstore = FAISS.from_documents(documents=all_splits, embedding=cached_embeddings)
retriever = vectorstore.as_retriever()

# # OpenAI 임베딩 사용 (retriever_1)
# embeddings_1 = OpenAIEmbeddings()
# vectorstore_1 = FAISS.from_documents(documents=all_splits, embedding=embeddings_1)
# retriever_1 = vectorstore_1.as_retriever()

# # BGE 임베딩 모델 사용 (retriever_2)
# embeddings_2 = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
# vectorstore_2 = FAISS.from_documents(documents=all_splits, embedding=embeddings_2)
# retriever_2 = vectorstore_2.as_retriever()

retriever_1 = vectorstore.as_retriever(search_kwargs={"filter": {"category": "AI"}})  
retriever_2 = vectorstore.as_retriever(search_kwargs={"filter": {"category": "Robotics"}})


# retriever_1 = vectorstore.as_retriever(search_kwargs={"k": 3})  # 상위 3개 문서 반환
# retriever_2 = vectorstore.as_retriever(search_kwargs={"k": 5})  # 상위 5개 문서 반환

# 문서 포맷팅 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# LangGraph State
class QAState:
    question: str
    context: str
    response: str

# LangGraph init
graph = StateGraph(QAState)

# Question node
graph.add_node("question_input", RunnablePassthrough())

# Search node
graph.add_node("retriever", retriever)

# Format_doc node
graph.add_node("format_docs", RunnableLambda(format_docs))

# prompt node
prompt = hub.pull("rlm/rag-prompt")
graph.add_node("prompt", prompt)

# LLM node
llm = ChatOpenAI()
graph.add_node("llm", llm)

# Workflow
graph.set_entry_point("question_input")  # Begin
graph.add_edge("question_input", "retriever")  # Question -> retriever(search)
graph.add_edge("retriever", "format_docs")  # retriever -> format_doc
graph.add_edge("format_docs", "prompt")  # format_doc -> prompt
graph.add_edge("prompt", "llm")  # prompt -> LLM
graph.add_edge("llm", END)  # LLM → response

# LangGraph Execute
app = graph.compile()

response = app.invoke({"question": "What are autonomous agents?"})
print(response)
