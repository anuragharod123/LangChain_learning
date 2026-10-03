from dotenv import load_dotenv

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

load_dotenv()

# 1. Load PDF 
loader = PyPDFLoader("RAG/document.pdf")
documents = loader.load()

# 2. Split document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)

# 3. Create Embedding Model
embedder = OpenAIEmbeddings(model= "text-embedding-3-small")

# 4. Store embeddings in vector database
if os.path.exists("RAG/faiss_index"):
    # Load existing vector database

    print("Loading existing vector database...")
    vector_db = FAISS.load_local(
        "RAG/faiss_index",
        embedder,
        allow_dangerous_deserialization=True
    )
else:
    # Create vector database for the first time
    print("Creating new vector database...")
    vector_db = FAISS.from_documents(
        chunks,
        embedder
    )
    # Save it locally
    vector_db.save_local("RAG/faiss_index")



# 5. Create retriever
retriever = vector_db.as_retriever(
    search_kwargs={"k": 3}
)

# 6. LLM
model = ChatOpenAI(model="gpt-4o-mini")

while True:
    # 7. Ask question
    question = input("Ask: ")

    # 8. Retrieve relevant chunks
    relevant_docs = retriever.invoke(question)

    # 9. Create context
    context = ""
    for doc in relevant_docs:
        context = context + doc.page_content + "\n\n"

    # 10. Send context + question to LLM
    prompt = f"""
    Answer the question using provided context.
    Context:
    {context}

    Question:
    {question}
    """
    if(question.lower() == "exit"):
        break
    response = model.invoke(prompt)

    print("\nAnswer:")
    print(response.content) 