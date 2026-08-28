from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

docs = [
    Document(page_content=" Python is a high-level programming language known for itssimplicity and readability. It is widely used in web development,data science, artificial intelligence, automation, and machine learning.", metadata={"source": "python_book"}),
    Document(page_content="Machine learning is a branch of artificial intelligence that allows computers to learn patterns from data and make predictions or decisions without being explicitly programmed for every task.", metadata={"source": "machine_learning_book"}),
    Document(page_content=" Retrieval-Augmented Generation, or RAG, combines information retrieval with a large language model. Relevant documents are retrieved from a knowledge base and provided to the LLM as context before generating an answer..", metadata={"source": "rag_book"}),
]

embeddings_model = MistralAIEmbeddings(model="mistral-embed")

vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings_model,
    persist_directory="chroma_db"
)

result = vector_store.similarity_search("What is python used for?", k=2)
for r in result:
    print(r.page_content)
    print("Source:", r.metadata["source"])
retriever = vector_store.as_retriever()
docs = retriever.invoke("explain RAG")
for d in docs:
    print(d.page_content)