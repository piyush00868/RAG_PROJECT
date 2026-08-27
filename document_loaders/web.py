from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://example.com")

docs = loader.load()

all_text = "\n".join(doc.page_content for doc in docs)

print(all_text)