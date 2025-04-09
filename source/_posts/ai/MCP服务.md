---
title: MCP服务
date: 2025-04-09 02:23:56
updated: 2025-04-09 02:23:56
tags:
  - AI
  - MCP
comments: false
categories: AI
thumbnail: https://images.unsplash.com/photo-1682687220923-c58b9a4592ae?crop=entropy&cs=srgb&fm=jpg&ixid=M3w2NDU1OTF8MHwxfHJhbmRvbXx8fHx8fHx8fDE3NDQxNzk4MzV8&ixlib=rb-4.0.3&q=85&w=1920&h=1080
published: false
---
# MCP服务

## 1. 简介

模型上下文协议（MCP）是一个创新的开源协议，它重新定义了大语言模型（LLM）与外部世界的互动方式。MCP 提供了一种标准化方法，使任意大语言模型能够轻松连接各种数据源和工具，实现信息的无缝访问和处理。MCP 就像是 AI 应用程序的 USB-C 接口，为 AI 模型提供了一种标准化的方式来连接不同的数据源和工具。
## 1.1 MCP架构

### 1.1.1 服务架构

![[../../images/MCP服务架构.svg]]

### 1.1.2 Agent架构

![[../../images/Agent架构.svg]]

### 1.1.3 简单客户端

通过启动本地的一个客户端来实现循环对话调用大模型

```python
import asyncio  
from mcp import ClientSession  
from openai import OpenAI  
from contextlib import AsyncExitStack  
  
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  
MODEL_NAME = "qwen-max"  
KEY = "" 
  
class MCPClient:  
  
    def __init__(self):  
        """初始化MCP客户端"""  
        self.session = None  
        self.exit_stack = AsyncExitStack()  
        self.openai_api_key = KEY  
        self.openai_api_base = BASE_URL  
        self.model = MODEL_NAME  
  
        if not self.openai_api_key:  
            raise ValueError("请设置您的OpenAI API密钥")  
  
        self.client = OpenAI(  
            api_key=self.openai_api_key,  
            base_url=self.openai_api_base,  
        )  
  
    # 处理对话请求  
    async def process_query(self, query: str) -> str:  
        messages = [{  
            "role": "system",  
            "content": "你是一个智能助手，帮助用户回答问题",  
        }, {  
            "role": "user",  
            "content": query,  
        }]  
        try:  
            response = await asyncio.get_event_loop().run_in_executor(  
                None,  
                lambda: self.client.chat.completions.create(  
                    model=MODEL_NAME,  
                    messages=messages  
                )  
            )  
            return response.choices[0].message.content  
        except Exception as e:  
            return f"调用OpenAI API错误：{str(e)}"  
  
    async def connect_to_mock_server(self):  
        """模拟 MCP 服务器的连接 """        print("Connecting to mock MCP server...")  
  
    async def chat_loop(self):  
        """运行交互式聊天循环"""  
        print("\n MCP 客户端已启动！输入 ‘quit’ 退出")  
  
        while True:  
            try:  
                user_input = input("请输入您的问题：").strip()  
                if user_input.lower() == "quit":  
                    print("退出交互式聊天")  
                    break  
                response = await self.process_query(user_input)  
                print(f"大模型：{response}")  
            except Exception as e:  
                print(f"发生错误：{str(e)}")  
  
    async def cleanup(self):  
        """清理资源"""  
        print("Cleaning up resources...")  
        await self.exit_stack.aclose()  
  
  
async def main():  
    mcp_client = MCPClient()  
  
    try:  
        await mcp_client.connect_to_mock_server()  
        await mcp_client.chat_loop()  
    finally:  
        await mcp_client.cleanup()  
  
  
if __name__ == '__main__':  
    asyncio.run(main())
```

## 1.2 MCP服务器通讯机制

**Model Context Protocol(MCP)** 是一种由Anthropic开源的协议，旨在将大型语言模型直接连接至数据源，实现无缝集成。根据 MCP 的规范，当前支持两种传输方式:
- 标准输入输出(stdio)：打开文件流的方式进行传输（同一个服务器，不需要通过端口监听）
- 基于HTTP的服务器推送事件(SSE)：在不同的机器分布式部署

而近期，开发者在MCP 的 GitHub 仓库中提交了一项提案，建议采用 **"可流式传输的HTTP（Streamable Http）"** 来替代现有的 HTTP+SSE方案。此举旨在解决当前远程 MCP 传输方式的关键限制，同时保留其优势。HTTP和SSE(服务器推送事件)在数据传输方式上存在明显区别

- 通信方式
	- HTTP：采用请求-响应模式，客户端发送请求，服务器返回响应，每次请求都是独立的 
	- SSE：允许服务器通过单个持久的HTTP连接，持续向客户端推送数据连
- 连接特性
	- HTTP：每次请求通常建立新的连接，虽然在HTTP/1.1中引入了持久连是短连接
	- SSE：适用于需要服务器主动向客户端推送数据的场景，如实时通知、股票行情更新等

注意：SSE仅支持服务器向客户端的单向通信，而websocket则是双向通信


|     传输方式      | 是否需要同时启动服务器 | 是否支持远程连接 |     使用场景      |
| :-----------: | :---------: | :------: | :-----------: |
| stdio（标准输入输出） |      ✔      |    ❌     | 本地通信，低延迟，高速交互 |
|  http（网络api）  |      ❌      |    ✔     |  分布式架构，远程通信   |
## 1.3 简单天气查询服务端

通过 **\@mcp.tool()** 来标注服务端提供的工具有哪些

```python
from typing import Any  
from mcp.server.fastmcp import FastMCP  
mcp = FastMCP("WeatherServer")  
  
async def fetch_weather(city: str) -> dict[str, Any] | None:  
    """  
    获取天气的api  
    :param city: 城市名称（需要使用英文，如Beijing）  
    :return: 天气数据字典；若出错返回包含 error信息的字典  
    """    return {  
        "城市": f"{city}\n",  
        "温度": "25.5°C\n",  
        "湿度": "25%\n",  
        "风俗": "12 m/s\n",  
        "天气": "晴\n",  
    } 
  
@mcp.tool()  
async def get_weather(city: str) -> dict[str, Any] | None:  
    """  
    获取天气  
    :param city: 城市名称（需要使用英文，如Beijing）  
    :return: 天气数据字典；若出错返回包含 error信息的字典  
    """    return await fetch_weather(city)  
  
if __name__ == '__main__':  
    # 使用标准 I/O 方式运行MCP服务器  
    mcp.run(transport='stdio')
```