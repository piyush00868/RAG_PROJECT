from dotenv import load_dotenv
load_dotenv()

from  langchain_mistralai import ChatMistralAI

model = ChatMistralAI(model="mistral-small-latest")

response = model.invoke("What is python and use cases of it?")

print(response.text)