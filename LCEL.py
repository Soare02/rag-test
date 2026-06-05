import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek  # 替换这里
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 加载环境变量
load_dotenv()

# 1. 定义提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的 {role}，请用中文简洁地回答。"),
    ("human", "{question}"),
])

# 2. 初始化 DeepSeek LLM（核心替换）
llm = ChatDeepSeek(
    model="deepseek-v4-flash",  # 官方模型名
    temperature=0.5,
    api_key=os.getenv("DEEPSEEK_API_KEY")  # 从.env读取
)

# 3. LCEL 链（完全不变）
chain = prompt | llm | StrOutputParser()

# 4. 调用（完全不变）
result = chain.stream({
    "role": "Python 工程师",
    "question": "什么是 LangChain LCEL？"
})

for chunk in result:
    print(chunk,end="",flush=True)