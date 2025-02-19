from dotenv import load_dotenv
from langchain.memory import (
    ConversationBufferMemory, 
    ConversationBufferWindowMemory,
    ConversationSummaryMemory,
    ConversationSummaryBufferMemory
)
from langchain_openai import ChatOpenAI

model = ChatOpenAI(temperature=0.1)
# text completion (텍스트 자동완성할 때 효율적)
# return_message = 챗모델 형태
# convMemory = ConversationBufferMemory(return_messages=True) 
# convMemory.save_context({"input": "Hi"}, {"output": "How are you?"})
# convMemory.load_memory_variables({})

# ConversationBufferWindowMemory() 
# 특정 시점까지의 메모리만 저장 ex) 5개까지만 저장한다고 설정하면 가장 오래된 기록이 순차적으로 지워짐
# 단점: 단기간의 대화만 기억이 가능
# convWindowMemory = ConversationBufferWindowMemory(
#     return_messages=True,
#     k=4,
# )

def add_message(memory, input, output):
    memory.save_context({"input": input}, {"output": output})
    
def get_history(memory):
    return memory.load_memory_variables({})

# ConversationSummaryMemory()
# 저장해야할 양이 늘어날 수록 요약해서 저장하면 메모리 세이브가 가능
# 초반에는 많은 정보를 저장해야해서 필요 토큰수가 많지만 시간이 지날 수록 누적된 정보에 따라 토큰 수 감소
# convSumMemory = ConversationSummaryMemory(llm=model)

# ConversationSummaryBufferMemory()
# BufferWindowMemory와 SummaryMemory의 장점을 합친 메모리
# 장기적인 대화에서 모델이 맥락은 이해해야하지만 전체 히스토리를 전부 사용하기에는 비효율적일 때 사용
convSumBufMemory = ConversationSummaryBufferMemory(
    llm=model,
    max_token_limit=150,
)
