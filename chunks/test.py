from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

splitter = CharacterTextSplitter(
    chunk_size= 10,
    chunk_overlap= 1,
     separator= "",
    
    )

data = TextLoader("chunks/notes.txt")

docs = data.load()

split_docs = splitter.split_documents(docs)

print("Total chunks:", len(split_docs))

for i, doc in enumerate(split_docs):
    print(f"\n--- Chunk {i + 1} ---")
    print(doc.page_content)