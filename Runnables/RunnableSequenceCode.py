from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence


load_dotenv()

prompt = PromptTemplate(
    template='{topic} ke dwara diye gaye vishesh updesh',
    input_variables=['topic']
)
model = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=1, max_tokens=150)
parser = StrOutputParser()

chain = RunnableSequence(prompt, model, parser)
result = chain.invoke({'topic':'pujya shree Premanand ji maharaj'})

print(result)