from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4o-mini', temperature=0,completion_tokens=20) 
response = model.invoke("joke on coackroach janta party")
print(response.content)