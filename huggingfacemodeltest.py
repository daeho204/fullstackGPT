import pandas as pd

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

train = pd.read_csv('C:/Users/DaeHoKim/Desktop/Dacon/open/train.csv', encoding = 'utf-8-sig')
test = pd.read_csv('C:/Users/DaeHoKim/Desktop/Dacon/open/test.csv', encoding = 'utf-8-sig')
# 데이터 전처리
train['공사종류(대분류)'] = train['공사종류'].str.split(' / ').str[0]
train['공사종류(중분류)'] = train['공사종류'].str.split(' / ').str[1]
train['공종(대분류)'] = train['공종'].str.split(' > ').str[0]
train['공종(중분류)'] = train['공종'].str.split(' > ').str[1]
train['사고객체(대분류)'] = train['사고객체'].str.split(' > ').str[0]
train['사고객체(중분류)'] = train['사고객체'].str.split(' > ').str[1]

test['공사종류(대분류)'] = test['공사종류'].str.split(' / ').str[0]
test['공사종류(중분류)'] = test['공사종류'].str.split(' / ').str[1]
test['공종(대분류)'] = test['공종'].str.split(' > ').str[0]
test['공종(중분류)'] = test['공종'].str.split(' > ').str[1]
test['사고객체(대분류)'] = test['사고객체'].str.split(' > ').str[0]
test['사고객체(중분류)'] = test['사고객체'].str.split(' > ').str[1]

# 훈련 데이터 통합 생성
combined_training_data = train.apply(
    lambda row: {
        "question": (
            f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
            f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
            f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
            f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다. "
            f"재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
        ),
        "answer": row["재발방지대책 및 향후조치계획"]
    },
    axis=1
)

# 테스트 데이터 통합 생성
combined_test_data = test.apply(
    lambda row: {
        "question": (
            f"공사종류 대분류 '{row['공사종류(대분류)']}', 중분류 '{row['공사종류(중분류)']}' 공사 중 "
            f"공종 대분류 '{row['공종(대분류)']}', 중분류 '{row['공종(중분류)']}' 작업에서 "
            f"사고객체 '{row['사고객체(대분류)']}'(중분류: '{row['사고객체(중분류)']}')와 관련된 사고가 발생했습니다. "
            f"작업 프로세스는 '{row['작업프로세스']}'이며, 사고 원인은 '{row['사고원인']}'입니다. "
            f"재발 방지 대책 및 향후 조치 계획은 무엇인가요?"
        )
    },
    axis=1
)

# DataFrame으로 변환
combined_test_data = pd.DataFrame(list(combined_test_data))

# DataFrame으로 변환
combined_training_data = pd.DataFrame(list(combined_training_data))

train_questions_prevention = combined_training_data['question'].tolist()
train_answers_prevention = combined_training_data['answer'].tolist()

train_documents = [
    f"Q: {q1}\nA: {a1}" 
    for q1, a1 in zip(train_questions_prevention, train_answers_prevention)
]

model_id = "kfkas/Llama-2-ko-7b-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32 if not torch.cuda.is_available() else torch.float16,
    device_map="auto" if torch.cuda.is_available() else {"": "cpu"}
)

# Train 데이터 준비
train_questions_prevention = combined_training_data['question'].tolist()
train_answers_prevention = combined_training_data['answer'].tolist()

train_documents = [
    f"Q: {q1}\nA: {a1}" 
    for q1, a1 in zip(train_questions_prevention, train_answers_prevention)
]

embedding_model_name = "jhgan/ko-sbert-nli"  # 임베딩 모델 선택
embedding = HuggingFaceEmbeddings(model_name=embedding_model_name)

# 벡터 스토어에 문서 추가
vector_store = FAISS.from_texts(train_documents, embedding)

# Retriever 정의
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True,  # sampling 활성화
    temperature=0.1,
    return_full_text=False,
    max_new_tokens=64,
)

prompt_template = """
### 지침: 당신은 건설 안전 전문가입니다.
질문에 대한 답변을 핵심 내용만 요약하여 간략하게 작성하세요.
- 서론, 배경 설명 또는 추가 설명을 절대 포함하지 마세요.
- 다음과 같은 조치를 취할 것을 제안합니다: 와 같은 내용을 포함하지 마세요.

{context}

### 질문:
{question}

[/INST]

"""

llm = HuggingFacePipeline(pipeline=text_generation_pipeline, device=0 if torch.cuda.is_available() else -1)

# 커스텀 프롬프트 생성
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)


# RAG 체인 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,  
    chain_type="stuff",  # 단순 컨텍스트 결합 방식 사용
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}  # 커스텀 프롬프트 적용
)

res = qa_chain.invoke("습도가 높으면 건설현장에서 사고날 확률이 높을까?")
print(res)