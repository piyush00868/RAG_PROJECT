from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader

splitter = TokenTextSplitter(
    chunk_size= 1000,
    chunk_overlap= 0
)

data = PyPDFLoader("token_split/notes.pdf")

docs = data.load()

chunks = splitter.split_documents(docs)

print("Total chunks:", len(chunks))

for i, doc in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(doc.page_content)

