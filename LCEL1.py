import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system","你是一个专业的{role}"),
    ("human","{question}")
])

llm = init_chat_model(
    "deepseek-v4-flash",
    temperature=0.8,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)

chain = prompt | llm | StrOutputParser()

result = chain.stream({
    "role": "up主",
    "question": "什么是爆款流量？"
})

for chunk in result:
    print(chunk,end="",flush=True)