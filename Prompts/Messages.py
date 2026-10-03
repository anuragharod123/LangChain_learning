from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini")

#this is a simple list
messages = [
    SystemMessage(content='you are a coding professor'),
    HumanMessage(content='what is java(concise ans)')
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))
print(messages)