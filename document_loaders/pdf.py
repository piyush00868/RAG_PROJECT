from langchain_community.document_loaders import PyPDFLoader

data = PyPDFLoader("document_loaders/notes.pdf")

docs = data.load()

print(docs[0].page_content)