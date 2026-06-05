import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from tools import TOOLS
load_dotenv()

llm = init_chat_model(
    model="google/gemma-4-e4b",
    base_url="http://localhost:1234/v1",
    temperature=0.8,
    api_key="lm-studio",
    model_provider="openai", 
)

agent = create_agent(
    model = llm,
    tools= TOOLS,
    system_prompt="你是一个帮助助手",
)

response = agent.invoke(
    {
        "messages": [
            SystemMessage("调用工具查询天气和时间"),
            HumanMessage("你好，我是soare"),
            AIMessage("你好soare"),
            HumanMessage("现在时间是多少？北京的天气如何"),
        ]
    },
    config={"recursion_limit": 11},
    # stream_mode="messages"
)

# for chunk,matedata in response:
#     if chunk.content:
#         print(chunk.content, end="", flush=True)
for message in  response['messages']:
    message.pretty_print()
