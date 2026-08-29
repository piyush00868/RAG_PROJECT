from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

splitter = TokenTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)

data = PyPDFLoader("documents/notes.pdf")

docs = data.load()

chunks = splitter.split_documents(docs)

print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk.page_content[:200])

template = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an AI assistant that summarizes the content of the document."
    ),
    (
        "human",
        "Summarize the following document:\n\n{data}"
    )
])

model = ChatMistralAI(model="mistral-small-latest")

document_text = "\n\n".join(
    chunk.page_content for chunk in chunks
)

prompt = template.format_prompt(data=document_text)

response = model.invoke(prompt)

print("\nSummary:")
print(response.content)