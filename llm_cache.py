import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from langchain_community.cache import SQLiteCache
from langchain.callbacks import StreamingStdOutCallbackHandler

load_dotenv()

# InMemoryCache는 재실행 시 기록이 지워져서 다시 질문하는 과정을 거쳐야한다
# set_llm_cache(InMemoryCache())

# sqlite를 사용한 캐싱
set_llm_cache(SQLiteCache("cache.db"))

model = ChatOpenAI(
    temperature=0.1
)
start_time = time.time()
response_1  = model.predict("explain why tesla stock is rapidly changing")
end_time = time.time()
first_d = end_time - start_time
print(f"⏱️ 첫 번째 실행 (캐싱 전): {first_d:.4f}초")
print(f"📝 응답 내용: {response_1}\n")

# 📌 2️⃣ 캐싱 후 두 번째 실행 시간 측정
start_time = time.time()
response_2 = model.predict("explain why tesla stock is rapidly changing")
end_time = time.time()
second_run_duration = end_time - start_time
print(f"⚡ 두 번째 실행 (캐싱 후): {second_run_duration:.4f}초")
print(f"📝 응답 내용: {response_2}\n")
reduction = first_d - second_run_duration
percentage = (reduction / first_d) * 100 if first_d > 0 else 0
print(f"🚀 캐싱으로 인한 속도 향상: {reduction:.4f}초 (약 {percentage:.2f}%)")