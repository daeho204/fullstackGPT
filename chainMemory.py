from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough


load_dotenv()

model = ChatOpenAI(temperature=0.1)

memory = ConversationSummaryBufferMemory(
    llm=model, 
    max_token_limit=100, 
    memory_key="history", 
    return_messages=True,
)

# template = """
#     You are a helpful AI talking to a human
#     {chat_history}
#     if you think that {chat_history} includes personal information. Say "This is personal Information".
#     Human: {question}
#     You:
# """

# chain = LLMChain(
#     llm=model,
#     memory=memory,
#     prompt=PromptTemplate.from_template(template),
#     verbose=True #chain log 확인
# )

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful AI talking to a human"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# chain = LLMChain(
#     llm=model,
#     memory=memory,
#     prompt=prompt,
#     verbose=True #chain log 확인
# )


def load_memory(input):
    print(input)
    return memory.load_memory_variables({})["history"]


chain = RunnablePassthrough.assign(history=load_memory) | prompt | model


def invoke_chain(question):
    result = chain.invoke({"question": question})
    memory.save_context(
        {"input": question},
        {"output": result.content},
    )
    print(result)


invoke_chain("My name is Louis")
invoke_chain("what is my name?")
# chain.predict(question="Hello my name is Louis and I am living in Seoul, Korea")
# result = chain.predict(question="What is my name?")
# location = chain.predict(question="Where am I living? give me a short guide of this location")

# load_memory = memory.load_memory_variables({})
# print(load_memory)
# print(result)
# print(location)
