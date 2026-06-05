from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from tools import TOOLS

llm = init_chat_model(
    model="google/gemma-4-e4b",
    base_url="http://localhost:1234/v1",
    temperature=0.8,
    api_key="lm-studio",
    model_provider="openai",
)

agent = create_agent(
    model=llm,
    tools=TOOLS,
    system_prompt="你是一个帮助助手",
)

message = HumanMessage([
    {"type": "text", "text": "描述下面图片内容"},
    {"type": "image", "url": "https://cdn.pixabay.com/photo/2015/04/23/22/00/tree-736885__480.jpg"},
])

stream = agent.stream(
    {"messages":[message]},
    stream_mode="messages"
)

for chunk,metadata in stream: 
    if chunk.content: 
        print(chunk.content, end="", flush=True)