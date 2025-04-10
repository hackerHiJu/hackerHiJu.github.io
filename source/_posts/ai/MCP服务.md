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
## 2 Python MCP

### 2.1 服务架构

![[../../images/MCP服务架构.svg]]

### 2.2 Agent架构

![[../../images/Agent架构.svg]]

### 2.3 简单客户端

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

### 2.4 MCP服务器通讯机制

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
### 2.5 简单天气查询服务端

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

### 2.6 天气查询客户端

```python
import asyncio  
import json  
from typing import Optional  
  
from mcp import ClientSession, StdioServerParameters  
from mcp.client.stdio import stdio_client  
from openai import OpenAI  
from contextlib import AsyncExitStack  
  
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"  
MODEL_NAME = "qwen-max"  
KEY = "xxx"  
  
  
class MCPClient:  
  
    def __init__(self):  
        """初始化MCP客户端"""  
        self.openai_api_key = KEY  
        self.openai_api_base = BASE_URL  
        self.model = MODEL_NAME  
  
        if not self.openai_api_key:  
            raise ValueError("请设置您的OpenAI API密钥")  
  
        self.client = OpenAI(  
            api_key=self.openai_api_key,  
            base_url=self.openai_api_base,  
        )  
        self.session: Optional[ClientSession] = None  
        self.exit_stack = AsyncExitStack()  
  
    # 处理对话请求  
    async def process_query(self, query: str) -> str:  
        messages = [{  
            "role": "system",  
            "content": "你是一个智能助手，帮助用户回答问题",  
        }, {  
            "role": "user",  
            "content": query,  
        }]  
  
        # 获取到工具列表  
        response = await self.session.list_tools()  
        available_tools = [  
            {  
                "type": "function",  
                "function": {  
                    "name": tool.name,  
                    "description": tool.description,  
                    "input_schema": tool.inputSchema,  
                }  
            }  
            for tool in response.tools]  
  
        try:  
            response = await asyncio.get_event_loop().run_in_executor(  
                None,  
                lambda: self.client.chat.completions.create(  
                    model=MODEL_NAME,  
                    messages=messages,  
                    tools=available_tools,  
                )  
            )  
            content = response.choices[0]  
            if content.finish_reason == "tool_calls":  
                # 如果使用的是工具，解析工具  
                tool_call = content.message.tool_calls[0]  
                tool_name = tool_call.function.name  
                tool_args = json.loads(tool_call.function.arguments)  
  
                # 执行工具  
                result = await self.session.call_tool(tool_name, tool_args)  
                print(f"\n\n工具调用:[{tool_name}]，参数:[{tool_args}]")  
  
                # 将工具返回结果存入message中，model_dump()克隆一下消息  
                messages.append(content.message.model_dump())  
                messages.append({  
                    "role": "tool",  
                    "content": result.content[0].text,  
                    "tool_call_id": tool_call.id,  
                })  
  
                response = self.client.chat.completions.create(  
                    model=MODEL_NAME,  
                    messages=messages,  
                )  
  
                return response.choices[0].message.content  
  
            # 正常返回  
            return content.message.content  
        except Exception as e:  
            return f"调用OpenAI API错误：{str(e)}"  
  
    async def connect_to_server(self, server_script_path: str):  
        """连接到 MCP 服务器的连接 """        is_python = server_script_path.endswith(".py")  
        is_js = server_script_path.endswith(".js")  
        if not (is_python or is_js):  
            raise ValueError("服务器脚本路径必须以 .py 或 .js 结尾")  
  
        command = "python" if is_python else "node"  
  
        server_params = StdioServerParameters(  
            command=command,  
            args=[server_script_path],  
            env=None  
        )  
  
        stdio_transport = await self.exit_stack.enter_async_context(  
            stdio_client(server_params)  
        )  
        read, write = stdio_transport  
        self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))  
        # 初始化会话  
        await self.session.initialize()  
        # 列出工具  
        response = await self.session.list_tools()  
        tools = response.tools  
        print("\n已经连接到服务器，支持以下工具：", [tools.name for tools in tools])  
  
    async def chat_loop(self):  
        """运行交互式聊天循环"""  
        print("\n MCP客户端已启动！输入 ‘quit’ 退出")  
  
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
        await mcp_client.connect_to_server("./mcp_server.py")  
        await mcp_client.chat_loop()  
    finally:  
        await mcp_client.cleanup()  
  
  
if __name__ == '__main__':  
    asyncio.run(main())
```

### 2.7 MCP概念

- Tools：服务器暴露可执行功能，供LLM调用以与外部系统交互
- Resources：服务器暴露数据和内容，供客户端读取并作为LLM上下文
- Prompts：服务器定义可复用的提示模板，引导LLM交互
- Sampling：让服务器借助客户端向LLM发起完成请求，实现复杂的智能行为
- Roots：客户端给服务器指定的一些地址，用来高速服务器该关注哪些资源和去哪里找这些资源

#### 2.7.1 Tools

服务器所支持的工具能力，使用提供的装饰器就可以定义对应的工具

```python
@mcp.tool()  
async def get_weather(city: str) -> dict[str, Any] | None:  
    """  
    获取天气  
    :param city: 城市名称（需要使用英文，如Beijing）  
    :return: 天气数据字典；若出错返回包含 error信息的字典  
    """    return await fetch_weather(city)
```

服务端连接通了session会话就可以通过对应的代码来进行查询支持哪些工具

```python
session.list_tools()
```

#### 2.7.2 Resources
类似于服务端定义了一个api接口用于查询数据，可以给大模型提供上下文

```python
@mcp.resource(uri="echo://hello")  
def resource() -> str:  
    """Echo a message as a resource"""  
    return f"Resource echo: hello"  
  
  
@mcp.resource(uri="echo://{message}/{age}")  
def message(message: str, age: int) -> str:  
    """Echo a message as a resource"""  
    return f"你好，{message}，{age}"
```

服务端查询时，如果使用了 {message} 作为占位符会解析为 **resource_templates** 使用 **list_resources** 只能获取到普通的资源
```python
# 查询资源  
resources = await self.session.list_resources()  
print("\n已经连接到服务器，支持以下资源：", [resources.name for resources in resources.resources])  
  
# 查询资源  
templates = await self.session.list_resource_templates()  
print("\n已经连接到服务器，支持以下模板资源：", [resources.name for resources in templates.resourceTemplates])  
  
resource_result = await self.session.read_resource("echo://hello")  
for content in resource_result.contents:
	# 对返回的字符串进行编码
    print(f"读取资源内容：{unquote(content.text)}")  
  
resource_result = await self.session.read_resource("echo://张三/18")  
for content in resource_result.contents:  
    print(f"读取资源内容：{unquote(content.text)}")
```

#### 2.7.3 Prompt
提示词，用于在服务端定义好自己的提示词来进行复用

```python
@mcp.prompt()  
def review_code(code: str) -> str:  
    return f"Please review this code:\n\n{code}"  
  
@mcp.prompt()  
def debug_error(error: str) -> list[base.Message]:  
    return [  
        base.UserMessage("I'm seeing this error:"),  
        base.UserMessage(error),  
        base.AssistantMessage("I'll help debug that. What have you tried so far?"),  
    ]
```

```python
# 查询提示词  
prompt_result = await self.session.list_prompts()  
print("\n已经连接到服务器，支持以下提示词：", [prompt.name for prompt in prompt_result.prompts])  
  
get_prompt = await self.session.get_prompt("review_code", { "code": "hello world"})  
for message in get_prompt.messages:  
    print(f"提示词内容：{message}")
```

#### 2.7.4 Images
MCP提供的一个Image类，可以自动处理图像数据

```python
from mcp.server.fastmcp import FastMCP, Image

@mcp.tool()  
def create_thumbnail(image_url: str) -> Image:  
    """Create a thumbnail from an image"""  
    img = PILImage.open(image_url)  
    img.thumbnail((100, 100))  
    return Image(data=img.tobytes(), format="jpg")
```

```python
# 调用图片工具  
image = await self.session.call_tool("create_thumbnail", {"image_url": "/Users/Documents/图片/WechatIMG47.jpg"})  
print("读取图片资源：", image)
```

#### 2.7.5 Context
Context 对象为您的工具和资源提供对 MCP 功能的访问权限，在服务端的工具中可以进行使用并且相互之间进行调用

```python
from mcp.server.fastmcp import FastMCP, Context

@mcp.tool()  
async def test(city: str, ctx: Context) -> str:  
    """  
    获取天气  
    :param city: 城市名称（需要使用英文，如Beijing）  
    :return: 天气描述  
    """    get_weather_city = await ctx.read_resource(f"echo://{city}/25")  
    result: str = ""  
    for content in get_weather_city:  
        result += unquote(content.content)  
    return result
```

```python
# 调用天气工具使用Context对象  
weather = await self.session.call_tool(name="test", arguments={"city": "北京"})  
print("天气信息：", weather)
```

## Spring MCP
