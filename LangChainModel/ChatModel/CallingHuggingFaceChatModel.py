from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()
ourllm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task= "text-generation"
)
model = ChatHuggingFace(llm = ourllm)
response = model.invoke("who are you??")
print(response.content)