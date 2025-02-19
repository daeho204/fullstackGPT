from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.llms.loading import load_llm
from langchain.callbacks import get_openai_callback

load_dotenv()

# model = load_llm("파일위치/파일명.파일형식")

# model = ChatOpenAI(
#     temperature=0.1
# )

model = OpenAI(
    temperature=0.1,
    max_tokens=200,
       
)
# 사용량 확인
# with get_openai_callback() as usage:
#     model.predict("what is the recipe for soju?")
# print(usage)