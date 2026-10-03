from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()
ourllm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task= "text-generation"
)
model = ChatHuggingFace(llm = ourllm)


chathistory = [
    SystemMessage(content="You're a cricket expert (need to response in max 30 words only)")
]

while True:
    user_input = input("You : ")
    if(user_input=="exit") : break
    chathistory.append(HumanMessage(content=user_input))
    result = model.invoke(chathistory)
    chathistory.append(AIMessage(content=result.content))
    print("AI: " , result.content)


print(chathistory)
