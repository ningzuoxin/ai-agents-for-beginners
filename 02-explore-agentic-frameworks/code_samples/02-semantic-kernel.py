"""
Lesson 02 - Semantic Kernel 框架示例 (纯 Python 版)

核心主题: 演示如何使用 Semantic Kernel 作为 AI Agent 框架的替代方案。

为什么需要 Semantic Kernel？与 MAF 相比有什么优势？
  1. Plugin 架构 — 工具以类(Plugin)组织，@kernel_function 装饰类方法，更符合 OOP 设计，
     适合复杂场景下将相关功能内聚在一起。
  2. 多语言支持 — 同时提供 C#、Python、Java SDK，方便 .NET 团队和 Java 团队接入。
  3. 企业级集成 — 内置大量 Azure 服务连接器（AI Search、Cosmos DB 等），
     开箱即用的记忆(memory)和向量存储集成。
  4. Process 框架 — 内置工作流编排能力，适合构建复杂的多步骤业务流程。
  5. 细粒度控制 — invoke_stream() 返回原始流事件，可精确区分函数调用/函数结果/文本内容，
     适合需要对中间过程做审计或日志的场景。

Semantic Kernel 核心概念:
  - Plugin: 用 @kernel_function 装饰的类方法，组织为类（区别于 MAF 的 @tool 独立函数）
  - ChatCompletionAgent: 使用 ChatCompletion 服务驱动对话
  - ChatHistoryAgentThread: 维护多轮对话的线程上下文
  - invoke_stream(): 流式调用，需手动区分 FunctionCall / FunctionResult / Text 事件

Plugin 架构设计原则:
  - 按领域聚合，而非一个 function 一个 Plugin：
    一个 Plugin 类 = 一组内聚的 function（如 TravelPlugin 包含 search_flights + search_hotels，
    WeatherPlugin 包含 get_forecast + get_alerts），遵循 OOP 内聚原则。
  - 组合的单位是 Plugin 类，而非单个 function：
    注册时通过 plugins=[TravelPlugin(), WeatherPlugin()] 按需装配，
    LLM 可见所有 Plugin 中的全部 @kernel_function。
  - 自由组合、高复用性：不同场景只需切换 Plugin 组合即可，无需修改个别 function 的注册逻辑。
  - 对比 MAF：MAF 的 @tool 是独立函数注册，适合轻量场景；SK 的 Plugin 更适合
    功能模块化、业务逻辑复杂的生产级场景。

说明: Semantic Kernel 是独立框架，本示例不涉及 MAF API 废弃问题。
"""


import os
import asyncio
import random
import json
from typing import Annotated

from dotenv import load_dotenv
from openai import AsyncOpenAI
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import (
    FunctionCallContent,
    FunctionResultContent,
    StreamingTextContent,
)
from semantic_kernel.functions import kernel_function

load_dotenv()


# ============================================================
# Plugin: 将工具组织为类，用 @kernel_function 装饰（区别于 MAF 的 @tool）
# ============================================================
class DestinationsPlugin:
    """A List of Random Destinations for a vacation."""

    def __init__(self):
        self.destinations = [
            "Barcelona, Spain",
            "Paris, France",
            "Berlin, Germany",
            "Tokyo, Japan",
            "Sydney, Australia",
            "New York, USA",
            "Cairo, Egypt",
            "Cape Town, South Africa",
            "Rio de Janeiro, Brazil",
            "Bali, Indonesia",
        ]
        self.last_destination = None

    @kernel_function(description="Provides a random vacation destination.")
    def get_random_destination(
        self,
    ) -> Annotated[str, "Returns a random vacation destination."]:
        available = self.destinations.copy()
        if self.last_destination and len(available) > 1:
            available.remove(self.last_destination)
        destination = random.choice(available)
        self.last_destination = destination
        return destination


async def main():
    # 创建 OpenAI 异步客户端和 Semantic Kernel 服务
    client = AsyncOpenAI(
        api_key=os.getenv("GITHUB_TOKEN"),
        base_url=os.getenv("GITHUB_ENDPOINT", "https://models.inference.ai.azure.com"),
    )
    chat_service = OpenAIChatCompletion(
        ai_model_id=os.getenv("GITHUB_MODEL_ID", "gpt-4o-mini"),
        async_client=client,
    )

    # 创建 Agent —— 注意: plugins 而非 tools
    agent = ChatCompletionAgent(
        service=chat_service,
        plugins=[DestinationsPlugin()],
        name="TravelAgent",
        instructions=(
            "You are a helpful AI Agent that can help plan vacations "
            "for customers at random destinations"
        ),
    )

    user_inputs = [
        "Plan me a day trip.",
        "I don't like that destination. Plan me another vacation.",
    ]

    thread = None
    for user_input in user_inputs:
        print("=" * 60)
        print(f"User: {user_input}")
        print("-" * 40)

        full_response: list[str] = []
        function_calls: list[str] = []
        current_fn_name = None
        arg_buffer = ""

        # 流式调用，需手动区分 FunctionCall / FunctionResult / Text
        async for response in agent.invoke_stream(messages=user_input, thread=thread):
            thread = response.thread
            for item in response.items:
                if isinstance(item, FunctionCallContent):
                    if item.function_name:
                        current_fn_name = item.function_name
                    if isinstance(item.arguments, str):
                        arg_buffer += item.arguments

                elif isinstance(item, FunctionResultContent):
                    if current_fn_name:
                        function_calls.append(
                            f"  [Function Call] {current_fn_name}({arg_buffer.strip()})"
                        )
                        current_fn_name = None
                        arg_buffer = ""
                    function_calls.append(f"  [Function Result] {item.result}")

                elif isinstance(item, StreamingTextContent) and item.text:
                    full_response.append(item.text)

        # 输出函数调用详情
        if function_calls:
            print("\n".join(function_calls))
            print("-" * 40)

        # 输出 Agent 文本回复
        print(f"Agent: {''.join(full_response)}\n")


if __name__ == "__main__":
    asyncio.run(main())
