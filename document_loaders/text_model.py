from dotenv import load_dotenv

from  langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader

load_dotenv()

data = TextLoader("documents/notes2.txt")
docs = data.load()

template = ChatPromptTemplate.from_messages([
    ('system', "You are a helpful assistant that sumarizes the content of the document."),
    ('human', "Summarize the following document: {data}")])

model = ChatMistralAI(model="mistral-small-latest")

prompt = template.format_prompt(data=docs[0].page_content)

response = model.invoke(prompt)

print(response.content)