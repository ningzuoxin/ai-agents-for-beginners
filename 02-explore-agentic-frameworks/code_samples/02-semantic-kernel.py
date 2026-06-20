"""
Lesson 02 - Semantic Kernel 框架示例 (纯 Python 版)

核心主题: 演示如何使用 Semantic Kernel 作为 AI Agent 框架的替代方案。

历史关系: Semantic Kernel (SK) 于 2023 年发布，是微软最早的 AI 编排 SDK；
  Microsoft Agent Framework (MAF) 于 2025-10 公开预览、2026-04 GA，
  定位为 SK + AutoGen 的企业级统一继任者（见 SK PyPI 官方说明）。
  本示例保留 SK 写法，用于对比两种框架的设计差异，并非"SK 比 MAF 更优"。

SK 与 MAF 的设计差异（各有取舍，非优劣）:
  1. Plugin 架构 vs @tool 函数 — SK 用类(Plugin) + @kernel_function 组织工具，
     适合 OOP 内聚；MAF 用 @tool 独立函数，更轻量 Pythonic。这是继任者做出的简化取舍。
  2. 多语言覆盖 — SK 提供 C#/Python/Java 三语言 SDK；
     MAF 目前仅 .NET + Python（暂无 Java），是 SK 仍有覆盖优势的一点。
  3. 企业集成方式不同 — SK 内置 Azure 连接器(AI Search/Cosmos DB)、记忆与向量存储；
     MAF 改为通过 Azure AI Foundry Agent Service V2 间接集成，抽象层级更高。
  4. Process 框架 vs 多 Agent 编排 — SK 的 Process 是显式工作流抽象；
     MAF 继承 AutoGen 的多 Agent 动态编排，思路不同。
  5. 流式控制粒度 — SK 的 invoke_stream() 在 auto-function-calling 模式下会过滤掉
     流中的 FunctionCall/FunctionResult，仅 yield 最终文本；但提供 on_intermediate_message
     回调获取中间步骤，可做审计。MAF 的 run() 已封装这些细节更简洁，但可控性弱。各有利弊。

Semantic Kernel 核心概念:
  - Plugin: 用 @kernel_function 装饰的类方法，组织为类（区别于 MAF 的 @tool 独立函数）
  - ChatCompletionAgent: 使用 ChatCompletion 服务驱动对话
  - ChatHistoryAgentThread: 维护多轮对话的线程上下文
  - invoke_stream(): 流式调用；SK 1.37 在 auto-function-calling 下仅 yield 最终文本，
    函数调用/结果需通过 on_intermediate_message 回调获取

Plugin 架构设计原则:
  - 按领域聚合，而非一个 function 一个 Plugin：
    一个 Plugin 类 = 一组内聚的 function（如 TravelPlugin 包含 search_flights + search_hotels，
    WeatherPlugin 包含 get_forecast + get_alerts），遵循 OOP 内聚原则。
  - 组合的单位是 Plugin 类，而非单个 function：
    注册时通过 plugins=[TravelPlugin(), WeatherPlugin()] 按需装配，
    LLM 可见所有 Plugin 中的全部 @kernel_function。
  - 自由组合、高复用性：不同场景只需切换 Plugin 组合即可，无需修改个别 function 的注册逻辑。
  - 与 MAF 对比: MAF（SK 的继任者）用 @tool 独立函数注册，更轻量；
    SK 的 Plugin 类更适合功能模块化场景，是两种设计取向，非优劣之分。

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
        print("get_random_destination function called...")
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

        # SK 1.37 的 invoke_stream() 会过滤掉流中的 FunctionCallContent / FunctionResultContent，
        # 只有最终文本会 yield。函数调用/结果通过 on_intermediate_message 回调传出。
        async def on_intermediate_message(message):
            for item in message.items:
                if isinstance(item, FunctionCallContent):
                    args = item.arguments.strip() if isinstance(item.arguments, str) else ""
                    function_calls.append(f"  [Function Call] {item.function_name}({args})")
                elif isinstance(item, FunctionResultContent):
                    function_calls.append(f"  [Function Result] {item.result}")

        # 流式调用，流中只会收到 StreamingTextContent；函数调用走 on_intermediate_message
        async for response in agent.invoke_stream(
            messages=user_input,
            thread=thread,
            on_intermediate_message=on_intermediate_message,
        ):
            thread = response.thread
            for item in response.items:
                if isinstance(item, StreamingTextContent) and item.text:
                    full_response.append(item.text)

        # 输出函数调用详情
        if function_calls:
            print("\n".join(function_calls))
            print("-" * 40)

        # 输出 Agent 文本回复
        print(f"Agent: {''.join(full_response)}\n")


if __name__ == "__main__":
    asyncio.run(main())
