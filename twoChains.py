import os
from dotenv import load_dotenv
from langchain_openai import OpenAI, ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage, BaseOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains.sequential import SequentialChain
from langchain.chains.llm import LLMChain
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts.few_shot import FewShotPromptTemplate

load_dotenv()

model = ChatOpenAI(
    temperature=0.1, 
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

prompt_1 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a world-class international chef. 
     You will create easy recipes for any type of cuisine with easy to find ingredients""",
        ),
        ("human", "I like to cook {food}."),
    ]
)
chain_1 = prompt_1 | model
prompt_2 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a world-class international vegetarian chef. 
     You will create easy recipes for any type of cuisine with easy to find ingredients.
     When providing a recipe, you will replace every ingredient to fit the vegetarian ingredient.
     If there are no alternative ingredients, say you do not have the replacement for it.""",
        ),
        ("human", "I like to cook {recipe}."),
    ]
)
chain_2 = prompt_2 | model
combined_chain = {"recipe": chain_1} | chain_2

result = combined_chain.invoke({"food": "Korean"})

print(result)
