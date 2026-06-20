"""
Lesson 03 - Agentic Design Patterns (纯 Python 版)

核心主题: 演示三种基础 Agentic 设计模式，均以"旅游目的地推荐"场景串联:
  1. 清晰的 Agent 指令 (Clear Agent Instructions) — 明确 persona / 职责 / 约束
  2. 使用 Pydantic 模型的结构化输出 (Structured Output) — 定义 schema → 解析 → 校验
  3. 单一职责 Agent (Single Responsibility Agents) — 多个聚焦 Agent 接力协作

API 变更说明（agent-framework >= 1.8.0）:
  - AzureAIProjectAgentProvider 已移除 → 改用 OpenAIChatCompletionClient + GitHub Models
  - await provider.create_agent(...) 已废弃 → 改用 client.as_agent(...)
  - as_agent() 不支持原生 response_format（已安装版本无该能力）→
    本版通过"指令约束输出 JSON + Pydantic model_validate 校验"实现结构化输出，
    这也是框架未原生支持 schema 时的通用做法，更能体现该知识点的工程价值。
    (注: 原笔记本中 Pydantic 模型仅定义未使用，本版补充了真正的解析与校验逻辑。)
"""

import os
import re
import sys
import json
import asyncio
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from agent_framework import tool
from agent_framework.openai import OpenAIChatCompletionClient

# Windows 控制台默认 GBK 编码，无法输出 ✓/✗ 等 Unicode 字符，强制 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

client = OpenAIChatCompletionClient(
    model=os.getenv("GITHUB_MODEL_ID"),
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url=os.getenv("GITHUB_ENDPOINT"),
)


# ============================================================
# 共用工具: 查询目的地详情（Pattern 2 / Pattern 3 都会用到）
# ============================================================
@tool(approval_mode="never_require")
def get_destination_details(
    destination: Annotated[str, "The destination to look up"],
) -> str:
    """Get details about a vacation destination."""
    details = {
        "Barcelona": "Available. Best: May-Jun. Beach, architecture, nightlife. ~$2000/week",
        "Tokyo": "Available. Best: Mar-Apr. Culture, food, technology. ~$2500/week",
        "Cape Town": "Not available. Best: Nov-Mar. Nature, wine, adventure. ~$1800/week",
    }
    print(f"  [tool] get_destination_details({destination!r}) called...")
    return details.get(destination, f"{destination}: No information available.")


# ============================================================
# Pattern 2 用的结构化输出 schema (Pydantic 模型)
# ============================================================
class DestinationRecommendation(BaseModel):
    destination: str
    available: bool
    best_season: str
    highlights: list[str]
    estimated_budget_usd: int


class TravelRecommendations(BaseModel):
    recommendations: list[DestinationRecommendation]
    personalized_note: str


def _extract_json(text: str) -> str:
    """从 Agent 回复中提取 JSON（兼容 markdown 代码块与前言文字）。"""
    text = text.strip()
    # 去除 ```json ... ``` 代码块围栏
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # 取第一个 { 到最后一个 } 之间的内容，忽略前后多余文字
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start : end + 1]
    return text


# ============================================================
# Pattern 1: 清晰的 Agent 指令
# ============================================================
async def pattern1_clear_instructions():
    """好指令需定义: Who(persona) / What(职责) / How(约束与风格)。"""
    print("=" * 60)
    print("Pattern 1: Clear Agent Instructions")
    print("=" * 60)

    agent = client.as_agent(
        name="TravelConcierge",
        instructions=(
            "You are a luxury travel concierge named Alex. Your role is to:\n"
            "1. Understand the traveler's preferences (budget, climate, activities)\n"
            "2. Check destination availability before making recommendations\n"
            "3. Provide detailed, personalized travel suggestions\n"
            "4. Always mention visa requirements and best travel seasons\n"
            "Be warm, professional, and enthusiastic about travel."
        ),
    )

    response = await agent.run(
        "I'd love a week-long vacation somewhere with great food and history. "
        "Budget around $2500."
    )
    print(response.text)
    print()


# ============================================================
# Pattern 2: 结构化输出 (Pydantic 模型)
# ============================================================
async def pattern2_structured_output():
    """让 Agent 返回可被程序解析、校验的结构化数据，而非自由文本。"""
    print("=" * 60)
    print("Pattern 2: Structured Output with Pydantic Models")
    print("=" * 60)

    structured_agent = client.as_agent(
        name="StructuredTravelExpert",
        instructions=(
            "You are a travel expert. Recommend destinations based on traveler "
            "preferences. Use the get_destination_details tool to ground your answer, "
            "and ONLY recommend destinations you can verify with the tool "
            "(it knows about: Barcelona, Tokyo, Cape Town).\n\n"
            "IMPORTANT: Respond with ONLY valid JSON (no markdown, no code fences, "
            "no extra commentary) matching exactly this schema:\n"
            "{\n"
            '  "recommendations": [\n'
            '    {"destination": str, "available": bool, "best_season": str, '
            '"highlights": [str], "estimated_budget_usd": int}\n'
            "  ],\n"
            '  "personalized_note": str\n'
            "}\n"
            "Provide exactly 3 recommendations."
        ),
        tools=[get_destination_details],
    )

    response = await structured_agent.run(
        "Recommend 3 destinations for a culture-loving traveler with a $2500 budget"
    )

    # 将 Agent 的文本回复解析为结构化数据并用 Pydantic 校验
    raw = _extract_json(response.text)
    try:
        result = TravelRecommendations.model_validate(json.loads(raw))
        print("✓ 结构化解析成功，Pydantic 校验通过:\n")
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    except (json.JSONDecodeError, ValidationError) as e:
        # 解析失败时回退展示原始回复，便于排查
        print("✗ 解析/校验失败，以下是 Agent 原始回复:")
        print(response.text)
        print(f"\n错误: {e}")
    print()


# ============================================================
# Pattern 3: 单一职责 Agent
# ============================================================
async def pattern3_single_responsibility():
    """复杂任务拆分给多个聚焦 Agent，各司其职、接力协作（关注点分离）。"""
    print("=" * 60)
    print("Pattern 3: Single Responsibility Agents")
    print("=" * 60)

    # Agent A: 只负责目的地研究
    destination_agent = client.as_agent(
        name="DestinationExpert",
        tools=[get_destination_details],
        instructions=(
            "You are a destination research specialist. Your only job is to:\n"
            "1. Evaluate destinations based on traveler preferences\n"
            "2. Check availability using the provided tool\n"
            "3. Return a short ranked list with pros/cons\n"
            "Do NOT discuss flights, hotels, or logistics — another agent handles that."
        ),
    )

    # Agent B: 只负责物流规划
    logistics_agent = client.as_agent(
        name="LogisticsPlanner",
        instructions=(
            "You are a travel logistics planner. Your only job is to:\n"
            "1. Create a day-by-day itinerary for the chosen destination\n"
            "2. Suggest flight and hotel options within the stated budget\n"
            "3. Note visa requirements and travel insurance recommendations\n"
            "Do NOT recommend destinations — another agent handles that."
        ),
    )

    # Step 1: 目的地专家挑选最佳选项
    dest_response = await destination_agent.run(
        "I want a week of culture and food for under $2500. Where should I go?"
    )
    print("=== Destination Expert ===")
    print(dest_response.text)

    # Step 2: 物流规划师基于推荐制定行程
    logistics_response = await logistics_agent.run(
        f"Plan a week-long trip based on this recommendation:\n{dest_response.text}"
    )
    print("\n=== Logistics Planner ===")
    print(logistics_response.text)
    print()


async def main():
    await pattern1_clear_instructions()
    await pattern2_structured_output()
    await pattern3_single_responsibility()


if __name__ == "__main__":
    asyncio.run(main())
