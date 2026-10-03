from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite') 
response = model.invoke("joke on coackroach janta party", max_output_tokens=40)
print(response.text)