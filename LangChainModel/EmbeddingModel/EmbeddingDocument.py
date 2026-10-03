from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

#can do for pdf as well just need some extra classes to load pdf and its pages
document = [
    "Hi Anurag",
    "Hi Yash",
    "Hi Aadi"
            ]

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001', output_dimensionality=10)
response = embeddings.embed_documents(document)
print(response)
