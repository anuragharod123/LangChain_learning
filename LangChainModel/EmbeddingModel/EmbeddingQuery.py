from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",dimensions=5)
response1 = embeddings.embed_query("I am Anurag")
response2 = embeddings.embed_query("I'm Anurag")

print(response1)
print(response2)