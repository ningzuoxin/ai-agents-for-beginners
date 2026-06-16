"""
Lesson 01 - Introduction to AI Agents (纯 Python 版)
====================================================

原教程 Notebook: 01-python-agent-framework.ipynb
基于: Microsoft AI Agents for Beginners 课程

⚠️ 重要说明 ⚠️
--------------
原教程 Notebook 使用了以下已废弃的 API（agent-framework >= 1.8.0 已移除）:
  - AzureAIProjectAgentProvider  ← 已移除
  - await provider.create_agent(...) ← 不再支持

本文件是参照原教程逻辑、使用当前最新 API 重写的纯 Python 版本:
  - 使用 OpenAIChatCompletionClient + GitHub Models（免费）替代 Azure AI Foundry
    （注意: 是 ChatCompletionClient，不是 ChatClient，后者走 Responses API 不被 GitHub Models 支持）
  - 使用 provider.as_agent(...) 替代 await provider.create_agent(...)
  - 使用 asyncio.run() 运行异步代码（.py 文件不能直接 await）

参考: https://github.com/microsoft/ai-agents-for-beginners/issues/572

使用方法:
  1. 在项目根目录创建 .env 文件，内容如下:
     GITHUB_TOKEN=<your-github-token>
     GITHUB_ENDPOINT=https://models.inference.ai.azure.com
     GITHUB_MODEL_ID=gpt-4o-mini

  2. 运行: python 01-python-agent-framework.py
"""

import logging
import os
import asyncio
from typing import Annotated

from dotenv import load_dotenv

# 加载 .env 中的环境变量
load_dotenv()

# ============================================================
# Step 1: 创建 Provider（连接 AI 模型服务）
# ============================================================
# 【旧 API - 已废弃】原教程使用 AzureAIProjectAgentProvider:
#   from agent_framework.azure import AzureAIProjectAgentProvider
#   provider = AzureAIProjectAgentProvider(credential=AzureCliCredential())
#
# 【新 API】改用 OpenAIChatCompletionClient + GitHub Models:
#   注意: OpenAIChatClient 使用的是 OpenAI Responses API (/v1/responses)，
#   GitHub Models 不支持该接口，必须使用 OpenAIChatCompletionClient
#   （标准 Chat Completions API /v1/chat/completions）
from agent_framework.openai import OpenAIChatCompletionClient

provider = OpenAIChatCompletionClient(
    model=os.getenv("GITHUB_MODEL_ID"),
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url=os.getenv("GITHUB_ENDPOINT"),
)

# ============================================================
# Step 2: 定义 Tool（给 Agent 的工具函数）
# ============================================================
# 使用 @tool 装饰器，agent 会在需要时自动调用这些函数
from agent_framework import tool

@tool(approval_mode="never_require")
def get_destinations() -> list[str]:
    """Get a list of popular vacation destinations."""
    return [
        "Barcelona",
        "Paris",
        "Berlin",
        "Tokyo",
        "Sydney",
        "New York City",
        "Cairo",
        "Cape Town",
        "Rio de Janeiro",
        "Bali",
    ]


# ============================================================
# Step 3: 创建并运行 Agent（同步方式）
# ============================================================
async def run_agent():
    """
    创建 Travel Agent 并以非流式方式运行。
    """
    # 【旧 API - 已废弃】原教程写法:
    #   agent = await provider.create_agent(
    #       tools=[get_destinations],
    #       name="TravelAgent",
    #       instructions="...",
    #   )
    #
    # 【新 API】使用 as_agent()（不需要 await）:
    agent = provider.as_agent(
        tools=[get_destinations],
        name="TravelAgent",
        instructions=(
            "You are a helpful travel agent. Help users find their perfect vacation "
            "destination based on their preferences. Use the get_destinations tool "
            "to see available destinations."
        ),
    )

    print("=" * 60)
    print("🤖 Travel Agent (非流式)")
    print("=" * 60)

    response = await agent.run(
        "I'm looking for a warm beach destination. What do you recommend?"
    )
    print(response)
    print()


# ============================================================
# Step 4: 流式响应（Streaming）
# ============================================================
async def run_agent_streaming():
    """
    以流式方式运行 Agent，逐 token 输出。
    流式输出体验更好，适合聊天界面。
    """
    agent = provider.as_agent(
        tools=[get_destinations],
        name="TravelAgent",
        instructions=(
            "You are a helpful travel agent. Help users find their perfect vacation "
            "destination based on their preferences. Use the get_destinations tool "
            "to see available destinations."
        ),
    )

    print("=" * 60)
    print("🤖 Travel Agent (流式输出)")
    print("=" * 60)

    async for chunk in agent.run(
        "Tell me about Tokyo as a travel destination", stream=True
    ):
        print(chunk, end="", flush=True)
    print()


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    # 在 .py 文件中，需要 asyncio.run() 来执行异步函数
    # asyncio.run(run_agent())
    # print("\n" + "=" * 60 + "\n")
    asyncio.run(run_agent_streaming())
