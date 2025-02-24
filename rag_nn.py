import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings


df = pd.read_excel("law_data.xlsx")


pdf_paths = ["law_doc1.pdf", "law_doc2.pdf", "law_doc3.pdf"]
all_docs = []
for path in pdf_paths:
    loader = PyPDFLoader(path)
    all_docs.extend(loader.load())

vectorstore = FAISS.from_documents(all_docs, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class LawDataset(Dataset):
    def __init__(self, dataframe):
        self.questions = dataframe["question"].tolist()
        self.answers = dataframe["answer"].tolist()

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = tokenizer(self.questions[idx], padding="max_length", truncation=True, return_tensors="pt")
        answer = tokenizer(self.answers[idx], padding="max_length", truncation=True, return_tensors="pt")
        return question, answer

dataset = LawDataset(df)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


class LawLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LawLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        output = self.fc(hn[-1])
        return output

INPUT_DIM = 768
HIDDEN_DIM = 512
OUTPUT_DIM = 768
model = LawLSTM(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
EPOCHS = 10
for epoch in range(EPOCHS):
    for question, answer in dataloader:
        question_input = question["input_ids"].squeeze(1).float()
        answer_target = answer["input_ids"].squeeze(1).float()
        optimizer.zero_grad()
        output = model(question_input)
        loss = criterion(output, answer_target)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

def get_law_answer_dl(question_text):
    model.eval()
    with torch.no_grad():
        question_tensor = tokenizer(question_text, return_tensors="pt")["input_ids"].float()
        output = model(question_tensor)
        answer_text = tokenizer.decode(output.argmax(dim=-1).tolist(), skip_special_tokens=True)
        return answer_text

llm = ChatOpenAI(model="gpt-4")

def get_law_answer_rag(question_text, retriever):
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    response = qa_chain.invoke(question_text)
    return response

model_bert = SentenceTransformer("jhgan/ko-sbert-nli")

def compare_answers(dl_answer, rag_answer):
    dl_embedding = model_bert.encode(dl_answer)
    rag_embedding = model_bert.encode(rag_answer)
    cosine_similarity = np.dot(dl_embedding, rag_embedding) / (np.linalg.norm(dl_embedding) * np.linalg.norm(rag_embedding))
    reference = [rag_answer.split()]
    candidate = dl_answer.split()
    bleu_score = sentence_bleu(reference, candidate)
    return cosine_similarity, bleu_score

def hybrid_answer(question, retriever):
    dl_answer = get_law_answer_dl(question)
    rag_answer = get_law_answer_rag(question, retriever)
    cos_sim, bleu = compare_answers(dl_answer, rag_answer)
    if cos_sim < 0.7 or bleu < 0.5:
        return f"RAG 기반 답변: {rag_answer}"
    else:
        return f"딥러닝 모델 답변: {dl_answer}"

question = "부동산 계약 시 주의할 점은?"
final_response = hybrid_answer(question, retriever)
print(final_response)
