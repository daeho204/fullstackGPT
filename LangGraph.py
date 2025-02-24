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

# 5️⃣ LLM 실행 노드
llm = ChatOpenAI()
graph.add_node("llm", llm)

# 6️⃣ 그래프의 흐름 정의
graph.set_entry_point("question_input")  # 시작 지점
graph.add_edge("question_input", "retriever")  # 질문 → 검색
graph.add_edge("retriever", "format_docs")  # 검색 → 문서 포맷팅
graph.add_edge("format_docs", "prompt")  # 문서 포맷팅 → 프롬프트 구성
graph.add_edge("prompt", "llm")  # 프롬프트 → LLM 실행
graph.add_edge("llm", END)  # LLM → 최종 출력

# LangGraph 실행기 생성
app = graph.compile()

# 실행 (예제 질문)
response = app.invoke({"question": "What are autonomous agents?"})
print(response)
