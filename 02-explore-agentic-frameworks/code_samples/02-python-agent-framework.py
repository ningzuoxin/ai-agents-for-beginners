"""
Lesson 02 - Exploring Microsoft Agent Framework (纯 Python 版)

核心主题: 演示 Agent Framework 四层架构 (Client→Agent→Tools→Session)，
         重点是带参数的工具和多轮对话 Session 机制。

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
def check_destination_availability(
    # Annotated 给类型附加元数据，框架会将其作为参数描述传给 LLM
    destination: Annotated[str, "The destination to check availability for"],
) -> str:
    """Check if a vacation destination is currently available for booking."""
    available = {
        "Barcelona": True,
        "Tokyo": True,
        "Cape Town": False,
        "Vancouver": True,
        "Dubai": False,
    }
    is_available = available.get(destination, False)
    return f"{destination} is {'available' if is_available else 'not available'} for booking."


async def run_multi_turn_conversation():
    """多轮对话: Agent 通过 Session 记住上下文"""
    agent = client.as_agent(
        name="TravelAvailabilityAgent",
        instructions=(
            "You are a travel booking agent. Help users check destination availability "
            "and make recommendations. Always check availability before recommending "
            "a destination."
        ),
        tools=[check_destination_availability],
    )
    session = agent.create_session()

    # Turn 1
    print("=" * 60)
    print("Turn 1 — 询问可用目的地")
    print("=" * 60)
    response = await agent.run(
        "Which destinations do you have available?",
        session=session,
    )
    print(f"Agent: {response}\n")

    # Turn 2 — Agent 利用 Session 记住之前的对话
    print("=" * 60)
    print("Turn 2 — 追问（Agent 利用上下文记忆）")
    print("=" * 60)
    response = await agent.run(
        "I'd like to go somewhere warm. What's available?",
        session=session,
    )
    print(f"Agent: {response}\n")

    # Turn 3
    print("=" * 60)
    print("Turn 3 — 继续追问")
    print("=" * 60)
    response = await agent.run(
        "What about Cape Town? Is that available?",
        session=session,
    )
    print(f"Agent: {response}\n")


if __name__ == "__main__":
    asyncio.run(run_multi_turn_conversation())
