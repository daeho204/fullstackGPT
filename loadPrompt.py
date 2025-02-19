from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import load_prompt
from langchain.prompts.pipeline import PipelinePromptTemplate
from langchain.prompts import PromptTemplate

load_dotenv()

yaml_prompt = load_prompt("./prompt.yaml")
json_prompt = load_prompt("./prompt.json")

model = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# yaml/json prompt 로드
yaml = yaml_prompt.format(country="Korea")
json = json_prompt.format(country="Germany")
# print(yaml)
# print(json)


intro = PromptTemplate.from_template(
    """
    You are a role playing assistant.
    And you are impersonating a {character}
"""
)

example = PromptTemplate.from_template(
    """
    This is an example of how you talk:

    Human: {example_question}
    You: {example_answer}
"""
)

start = PromptTemplate.from_template(
    """
    Start now!

    Human: {question}
    You:
"""
)

final = PromptTemplate.from_template(
    """
    {intro}
                                     
    {example}
                              
    {start}
"""
)

prompts = [
    ("intro", intro),
    ("example", example),
    ("start", start),
]


full_prompt = PipelinePromptTemplate(
    final_prompt=final,
    pipeline_prompts=prompts,
)


chain = full_prompt | model

chain.invoke(
    {
        "character": "Pirate",
        "example_question": "What is your location?",
        "example_answer": "Arrrrg! That is a secret!! Arg arg!!",
        "question": "What is your fav food?",
    }
)
