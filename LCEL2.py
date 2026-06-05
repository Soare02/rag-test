from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from tools import TOOLS

llm = init_chat_model(
    model="google/gemma-4-e4b",
    base_url="http://localhost:1234/v1",
    temperature=0.8,
    api_key="111",
    model_provider="openai",
)

agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt="你是一个帮助助手",
)

