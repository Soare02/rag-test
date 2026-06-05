import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from tools import TOOLS
load_dotenv()

llm = init_chat_model(
    model="google/gemma-4-e4b",
    base_url="http://localhost:1234/v1",
    temperature=0.8,
    api_key="lm-studio",
    model_provider="openai", 
)
llm_deepseek = init_chat_model(
    model="deepseek-v4-flash",
    base_url="https://api.deepseek.com",
    temperature=0.8,
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    model_provider="openai", 
    model_kwargs={"extra_body": {"thinking": {"type": "disabled"}}}
)
SYSTEMPROMPT = """
你是专业**多作品混合圣地巡礼规划师**。
输入：带**地理坐标、地点名、作品名、集数时间戳**的地标列表 + 指定巡礼天数。
任务：基于坐标做空间分析，联网补全现实地理/交通信息，输出**顺路不绕路、交通成本低、可直接落地**的多日巡礼方案。

## 信息采集要求
1. 补全每个地标：精确地址、所属区域、最近车站；测算点位间步行/公交距离耗时，同区域自动分组。
2. 整理当地JR/地铁/巴士线路、运营时间、推荐交通卡。
3. 补充沿途餐饮、休息、卫生间；热门点位最佳拍摄时段、角度、人流避坑、开放预约信息。
4. 标注每个地标作品名场面、可还原打卡细节。
5. 对每个地标调用 get_anime_scene 查询原作场景的剧情氛围（时间/天气/光线/季节），用于还原式时间安排。

## 规划规则
1. 按**地理区域优先**分配每日行程，杜绝跨区折返。
2. 单日步行控制 6–10km，动线按「车站→打卡→午餐→下午打卡→返程车站」闭环排布。
3. 多作品同区域可混合排布，以顺路为第一优先级。
4. 经典名场面地标优先排布。
5. **时间还原优先**：根据场景查询结果中的氛围信息（黄昏/夜晚/清晨/雨天等），将该地标安排在一天中对应的时段访问。同一区域多个地标若场景时段不同，按时间线顺序串联。

## 固定输出格式（严格遵守）
```markdown
# 多作品混合圣地巡礼路线规划（X日版）
## 整体规划说明
- 地标总数：X个
- 涉及作品：xxx
- 出行方式：JR/地铁+步行，单日平均步行约Xkm
- 核心区域：xxx
- 推荐交通卡：xxx

---
## 第N天：XX区域巡礼（共X个地点）
### 路线顺序：起点站 → 地标1 → 地标2 → … → 返程站
### 当日步行：约Xkm | 总耗时：约X小时

1. 【地标：XX】
- 作品出处：《XX》EPX XX:XX
- 坐标：纬度/经度
- 现实地址：xxx
- 交通衔接：xxx
- 打卡建议：xxx
- 原作场景时段：xxx（EPX XX:XX）
- 推荐到访时段：xxx
- 预估停留：X分钟

### 当日餐饮休息建议
午餐/休息/晚餐顺路点位推荐

---
## 整体出行小贴士
交通卡、乘车APP、错峰打卡、名场面还原、当地礼仪、紧急电话极简要点即可，无需冗余描述。
```
"""

agent = create_agent(
    model=llm_deepseek,
    tools=TOOLS,
    system_prompt=SYSTEMPROMPT,
)

result = agent.stream(
    {"messages": [{"role": "user", "content": "巡礼天数：3天\n需要访问的地标（共4个）：\n1. 地点名称：京都音乐厅\n作品名称：吹响吧！上低音号\n出现集数：EP1\n时间戳：0:25\n坐标：35.0503, 135.7664\n\n2. 地点名称：宇治桥\n作品名称：吹响吧！上低音号\n出现集数：EP3\n时间戳：5:20\n坐标：34.8903, 135.8002\n\n3. 地点名称：车站前广场\n作品名称：吹响吧！上低音号\n出现集数：EP2\n时间戳：1:30\n坐标：34.8841, 135.7996\n\n4. 地点名称：大吉山展望台\n作品名称：吹响吧！上低音号\n出现集数：EP8\n时间戳：11:45\n坐标：34.8728, 135.8125"}]},
    config={"recursion_limit": 11} ,
    stream_mode="messages"
)

for chunk,metadata in result: 
    if chunk.content: 
        print(chunk.content, end="", flush=True)