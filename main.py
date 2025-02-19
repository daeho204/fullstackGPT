import os
from dotenv import load_dotenv

# from langchain.llms import OpenAI # Deprecated
# from langchain.chat_models import ChatOpenAI # Deprecated
from langchain_openai import OpenAI, ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.sequential import SequentialChain
from langchain.chains.llm import LLMChain

class CommaOutputParser(BaseOutputParser):
    def parse(self, text):
        items = text.strip().split(",")
        return list(map(str.strip, items))
    
load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY")

# Prompt 생성
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a list generating machine. 
                   Your answers will always be in a comma separated list format. 
                   You will return maximum of {max_items} answers"""),
        ("ai", "you will answer all the questions very politely in Korean"),
        ("human", "{question}"),
    ]
)
# Model 생성
model = ChatOpenAI(
    temperature=0.1, # 정확도
    max_tokens=150,  # 응답길이
    top_p=0.5,       # 다양성
    stop=["Thank you", "End of response"]
    )

# Chain 생성
# Langchain의 가장 기본적인 구조 template | chatmodel | parser
chain = prompt | model | CommaOutputParser()

# Chain에 값 전달
result = chain.invoke({
    "max_items": 7,
    "question": "provide me the most famous pokemons",
})
print(result)
