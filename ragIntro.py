from reprlib import recursive_repr
from langchain_openai import ChatOpenAI
# from langchain_community.document_loaders import UnstructuredFileLoader, PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter


splitter = RecursiveCharacterTextSplitter(
    # chunk_size=600,
    # chunk_overlap=100,
    # separators=["\n"],
)
# splitter = CharacterTextSplitter(
#     separator="\n"
# )
print(splitter)
loader = UnstructuredLoader("./files/Chapter_One.txt")
doc = loader.load()

# splitted = splitter.split_documents(doc)
splitted = loader.load_and_split(text_splitter=splitter)
print(splitted)
print(len(splitted))


