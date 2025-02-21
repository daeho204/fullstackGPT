from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()
embedder = OpenAIEmbeddings()

vector = embedder.embed_query("hi")
print(len(vector))