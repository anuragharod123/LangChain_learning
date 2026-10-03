from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template='Generate 5 interesting facts about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=1, max_tokens=150)
parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({'topic':'cricket'})

print(result)


#Used to visualize the chain graphically it requires this "pip install grandalf" 
#chain.get_graph().print_ascii()