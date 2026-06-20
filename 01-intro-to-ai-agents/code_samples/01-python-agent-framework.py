"""
Lesson 01 - Introduction to AI Agents (纯 Python 版)

核心主题: 创建一个带工具(tool)的 AI Agent，演示非流式和流式两种运行方式。

API 变更说明（agent-framework >= 1.8.0）:
  - AzureAIProjectAgentProvider 已移除 → 改用 OpenAIChatCompletionClient + GitHub Models
  - await provider.create_agent(...) 已废弃 → 改用 provider.as_agent(...)
"""

import os
import asyncio
from typing import Annotated

from dotenv import load_dotenv
from agent_framework import tool
from agent_framework.openai import OpenAIChatCompletionClient

load_dotenv()

client = OpenAIChatCompletionClient(
    model=os.getenv("GITHUB_MODEL_ID"),
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url=os.getenv("GITHUB_ENDPOINT"),
)


@tool(approval_mode="never_require")
def get_destinations() -> list[str]:
    """Get a list of popular vacation destinations."""
    print("get_destinations function called...")
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


async def run_agent():
    """非流式: 一次性返回完整回复"""
    agent = client.as_agent(
        tools=[get_destinations],
        name="TravelAgent",
        instructions=(
            "You are a helpful travel agent. Help users find their perfect vacation "
            "destination based on their preferences. Use the get_destinations tool "
            "to see available destinations."
        ),
    )
    response = await agent.run(
        "I'm looking for a warm beach destination. What do you recommend?"
    )
    print("=" * 60)
    print("非流式输出:")
    print("=" * 60)
    print(response)


async def run_agent_streaming():
    """流式: 逐 token 输出，适合聊天界面"""
    agent = client.as_agent(
        tools=[get_destinations],
        name="TravelAgent",
        instructions=(
            "You are a helpful travel agent. Help users find their perfect vacation "
            "destination based on their preferences. Use the get_destinations tool "
            "to see available destinations."
        ),
    )
    print("=" * 60)
    print("流式输出:")
    print("=" * 60)
    async for chunk in agent.run(
        "Tell me about Tokyo as a travel destination", stream=True # 这个问题不会调用工具
        # "I'm looking for a warm beach destination. What do you recommend?", stream=True
    ):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    # asyncio.run(run_agent())
    asyncio.run(run_agent_streaming())
