from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=50,
)

llm = HuggingFacePipeline(pipeline=pipe)

model = ChatHuggingFace(llm=llm)

response = model.invoke("Write a short joke about a cockroach")

print(response.content)