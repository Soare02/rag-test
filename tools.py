import os
from dotenv import load_dotenv
import requests
from langchain.tools import tool
from tavily import TavilyClient
load_dotenv()

@tool
def get_weather(city: str) -> str:
    """
    通过调用 wttr.in API 查询真实的天气信息。
    """
    # API端点，我们请求JSON格式的数据
    url = f"https://wttr.in/{city}?format=j1"
    
    try:
        # 发起网络请求
        response = requests.get(url)
        # 检查响应状态码是否为200 (成功)
        response.raise_for_status() 
        # 解析返回的JSON数据
        data = response.json()
        
        # 提取当前天气状况
        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_C']
        
        # 格式化成自然语言返回
        return f"{city}当前天气:{weather_desc}，气温{temp_c}摄氏度"
        
    except requests.exceptions.RequestException as e:
        # 处理网络错误
        return f"错误:查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        # 处理数据解析错误
        return f"错误:解析天气数据失败，可能是城市名称无效 - {e}"

@tool
def get_time()-> str:
    """获取当前时间"""
    return f"当前时间为：5:20:00"

@tool
def get_yg()->str:
    """用户询问妖狗是谁时获取信息"""
    return f"妖狗是一个王者荣耀选手，最喜欢玩韩信"

@tool
def get_attraction(city: str, weather: str) -> str:
    """
    根据城市和天气，使用Tavily Search API搜索并返回优化后的景点推荐。
    """
    # 1. 从环境变量中读取API密钥
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "错误:未配置TAVILY_API_KEY环境变量。"

    # 2. 初始化Tavily客户端
    tavily = TavilyClient(api_key=api_key)
    
    # 3. 构造一个精确的查询
    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"
    
    try:
        # 4. 调用API，include_answer=True会返回一个综合性的回答
        response = tavily.search(query=query, search_depth="basic", include_answer=True)
        
        # 5. Tavily返回的结果已经非常干净，可以直接使用
        # response['answer'] 是一个基于所有搜索结果的总结性回答
        if response.get("answer"):
            return response["answer"]
        
        # 如果没有综合性回答，则格式化原始结果
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")
        
        if not formatted_results:
             return "抱歉，没有找到相关的旅游景点推荐。"

        return "根据搜索，为您找到以下信息:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"错误:执行Tavily搜索时出现问题 - {e}"
    
@tool
def get_anime_scene(anime_title: str, episode: str, location_name: str, timestamp: str = "") -> str:
    """
    查询动漫中某个特定场景的剧情细节、时间氛围和情节内容。
    模型可根据返回的场景信息（如黄昏、夜晚、雨天等氛围）调整巡礼路线的时间安排。

    Args:
        anime_title: 作品名称，如"吹响吧！上低音号"
        episode: 集数，如"EP1"、"第3集"
        location_name: 地点名称，如"宇治桥"、"大吉山展望台"
        timestamp: 时间戳，如"0:25"、"11:45"
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "错误:未配置TAVILY_API_KEY环境变量。"

    parts = [f"动画《{anime_title}》", episode, location_name]
    if timestamp:
        parts.append(f"第{timestamp}左右的剧情")
    parts.append("场景描写 时间 氛围 情节")
    query = " ".join(parts)

    tavily = TavilyClient(api_key=api_key)

    try:
        response = tavily.search(query=query, search_depth="basic", include_answer=True)

        if response.get("answer"):
            return response["answer"]

        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")

        if not formatted_results:
            return f"未找到《{anime_title}》{episode}中{location_name}的相关场景信息。"

        return "为您找到以下场景信息:\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"错误:搜索场景信息时出现问题 - {e}"


TOOLS = [get_weather, get_attraction, get_anime_scene]