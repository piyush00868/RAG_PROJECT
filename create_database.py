# load pdf
# split into chunks
# create the embeddings
# store in chrome vector store

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

loader = PyPDFLoader("recursive_split/designsystem.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size= 1000,
    chunk_overlap= 200
)

chunks = splitter.split_documents(docs)

embeddings_model = MistralAIEmbeddings(model="mistral-embed")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory="chroma_db"
)

