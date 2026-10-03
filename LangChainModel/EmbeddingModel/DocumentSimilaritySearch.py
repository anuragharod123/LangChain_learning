from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as nm


#can do for pdf as well just need some extra classes to load pdf and its pages
document = [
    "Virat Kohli is Aggressive Batsman",
    "Virat Kohli is from delhi",
    "Virat Kohli Father died in his early 30's"
]
query = "virat kohli father is alive?"

load_dotenv()
embeddings = GoogleGenerativeAIEmbeddings(model='gemini-embedding-001', output_dimensionality=10)
doc_embeddings = embeddings.embed_documents(document)
query_embeddings = embeddings.embed_query(query)

similarity_score = cosine_similarity([query_embeddings],doc_embeddings)
print(similarity_score)

best_index = nm.argmax(similarity_score[0]) 
#takes 2d list in input so givins index 0 (similarity_score[0]) so basically  
#it will have list of score and then argmax will give index of higest value


print(document[best_index])
