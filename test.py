import os
from dotenv import load_dotenv
# from langchain.prompts import load_prompt
# from langchain_openai import ChatOpenAI
# from langchain.prompts import PromptTemplate, ChatPromptTemplate
# from langchain.callbacks import StreamingStdOutCallbackHandler

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.storage import LocalFileStore
from langchain.embeddings.cache import CacheBackedEmbeddings

load_dotenv()

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# Text Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Cache
cache_dir = LocalFileStore(".cache/embeddings/")
embeddings = OpenAIEmbeddings()
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

# Embedding -> Store(retriever)
vectorstore = FAISS.from_documents(
    documents=all_splits, 
    embedding=cached_embeddings
)

# LLM
llm = ChatOpenAI()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
qa_chain = (
    {
        "context": vectorstore.as_retriever() | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

qa_chain.invoke("What are autonomous agents?")

# loaded_prompt = load_prompt("./prompt.yaml")
# model = ChatOpenAI(
#     temperature=0.1,
#     streaming=True,
#     callbacks=[StreamingStdOutCallbackHandler()]
# )

# intro_prompt = PromptTemplate.from_template(
#     "You are a person from {country}. When called, please response nicely in your mother language. Since you are from suburban, you have accent. If you have accent, please do not hesitate to use it"
# )
# next_prompt = ChatPromptTemplate.from_messages([
#     "system", "you are famous cook in {country}. give me a short list of your finest cuisine."
#     ]
# )

# chain = loaded_prompt | model
# another_chain = next_prompt | model
# combined_chain = {"country": intro_prompt} | another_chain
# country = input("Please enter a country: ")
# result = combined_chain.invoke({
#     "country" : country
# })
# print(result)

# import fitz  # PyMuPDF

# def extract_styled_text_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
    
#     bold_texts = []
#     blue_texts = []

#     for page_num in range(len(doc)):
#         page = doc.load_page(page_num)
#         blocks = page.get_text("dict")["blocks"]
        
#         for block in blocks:
#             if "lines" in block:
#                 for line in block["lines"]:
#                     for span in line["spans"]:
#                         font_name = span.get("font", "").lower()
#                         font_flags = span["flags"]
#                         color = span["color"]

#                         # 1️⃣ Bold 텍스트 추출 (font에 bold 포함 or font_flags & 2)
#                         if "bold" in font_name or (font_flags & 2):
#                             bold_texts.append(span["text"])

#                         # 2️⃣ 파란색(RGB 값) 텍스트 추출 (Blue: 255)
#                         if color == 255:  # RGB(0,0,255) == 255
#                             blue_texts.append(span["text"])
    
#     return bold_texts, blue_texts

# # ✅ 실행
# pdf_path = "./files/bold_list.pdf"
# bold_list, blue_list = extract_styled_text_from_pdf(pdf_path)

# print("\n💪 Bold 텍스트 리스트:")
# print(bold_list)

# print("\n🔵 파란색 텍스트 리스트:")
# print(blue_list)

# import fitz  # PyMuPDF
# import re

# def extract_styled_text_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
    
#     bold_text_fragments = []  # Bold 조각 모음
#     blue_texts = []

#     for page_num in range(len(doc)):
#         page = doc.load_page(page_num)
#         blocks = page.get_text("dict")["blocks"]
        
#         for block in blocks:
#             if "lines" in block:
#                 for line in block["lines"]:
#                     for span in line["spans"]:
#                         font_name = span.get("font", "").lower()
#                         font_flags = span["flags"]
#                         color = span["color"]

#                         # 1️⃣ Bold 텍스트 추출 (font에 bold 포함 or font_flags & 2)
#                         if "bold" in font_name or (font_flags & 2):
#                             bold_text_fragments.append(span["text"])

#                         # 2️⃣ 파란색(RGB 값) 텍스트 추출 (Blue: 255)
#                         if color == 255:
#                             blue_texts.append(span["text"])

#     # 🧩 Bold 텍스트를 문장 단위로 합치기
#     bold_text = " ".join(bold_text_fragments)  # 조각 이어붙이기
#     bold_sentences = re.split(r'(?<!\b[A-Z][a-z])(?<!\bMr)(?<!\bDr)(?<!\bMs)(?<!\bJr)(?<!\bSr)\.\s+', bold_text)
#     bold_sentences = [s.strip() + '.' for s in bold_sentences if s.strip()]  # 문장 끝 마침표 추가

#     # 🔥 5문장 이하라면 한 인덱스에 묶어서 반환
#     if len(bold_sentences) <= 5:
#         bold_sentences = [" ".join(bold_sentences)]

#     return bold_sentences, blue_texts

# # ✅ 실행
# pdf_path = "./files/bold_list.pdf"
# bold_list, blue_list = extract_styled_text_from_pdf(pdf_path)

# print("\n💪 Bold 문장 리스트:")
# for i, sentence in enumerate(bold_list):
#     print(f"{i}: {sentence}")

# print("\n🔵 파란색 텍스트 리스트:")
# print(blue_list)
