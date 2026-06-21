"""
Lesson 04 - Tool Use Design Pattern (纯 Python 版)

核心主题: 演示"工具使用 (Tool Use)"这一 Agentic 设计模式，以"旅行预订 Agent"场景串联:
  1. 用 @tool 装饰器定义工具 — docstring 即工具描述，类型注解 (含 Annotated) 即工具 schema
  2. 组合多个工具 — Agent 依需自动调用、串行协作，回答复杂问题
  3. 结构化输出 — 用 Pydantic 模型约束 + 校验 Agent 返回，便于下游程序消费
  4. 工具审批控制 — approval_mode 决定工具调用是否需要人工确认，敏感操作"人在回路"

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
# 工具定义: @tool 装饰器把普通函数变成 Agent 可调用的工具
#
# 要点:
#   - 函数 docstring  → 模型看到的"工具描述"
#   - 参数类型注解    → 工具的参数 schema (Annotated 的第二项是参数说明)
#   - approval_mode   → 控制该工具调用是否需要人工审批
# ============================================================
@tool(approval_mode="never_require")
def get_destinations() -> list[str]:
    """Get available vacation destinations."""
    print("  [tool] get_destinations() called...")
    return ["Barcelona", "Paris", "Berlin", "Tokyo", "Sydney", "New York City"]


@tool(approval_mode="never_require")
def check_availability(
    destination: Annotated[str, "The destination to check"],
) -> str:
    """Check booking availability for a destination."""
    print(f"  [tool] check_availability({destination!r}) called...")
    availability = {
        "Barcelona": "Available - 3 spots left",
        "Paris": "Available",
        "Berlin": "Sold out",
        "Tokyo": "Available - 1 spot left",
        "Sydney": "Available",
        "New York City": "Available",
    }
    return availability.get(destination, "Unknown destination")


@tool(approval_mode="never_require")
def get_flight_info(
    origin: Annotated[str, "Origin airport code"],
    destination: Annotated[str, "Destination airport code"],
) -> str:
    """Get flight information between two cities."""
    print(f"  [tool] get_flight_info({origin!r}, {destination!r}) called...")
    flights = {
        "LHR-BCN": "BA 2042, Departs 08:30, Arrives 11:45, $350",
        "LHR-CDG": "AF 1081, Departs 09:15, Arrives 11:30, $280",
        "LHR-NRT": "JL 044, Departs 11:00, Arrives 07:00+1, $890",
    }
    return flights.get(
        f"{origin}-{destination}",
        f"No direct flights from {origin} to {destination}",
    )


# 敏感操作工具: 预订航班 —— 需要"人在回路"确认后才执行
@tool(approval_mode="always_require")
def book_flight(
    origin: Annotated[str, "Origin airport code"],
    destination: Annotated[str, "Destination airport code"],
    passenger_name: Annotated[str, "Full name of the passenger"],
) -> str:
    """Book a flight for a passenger. Requires approval before executing."""
    print(
        f"  [tool] book_flight({origin!r}, {destination!r}, "
        f"{passenger_name!r}) called..."
    )
    return (
        f"Flight booked from {origin} to {destination} "
        f"for {passenger_name}. Confirmation #TRV-2024-"
        f"{hash(passenger_name) % 10000:04d}"
    )


# ============================================================
# 结构化输出 schema (Pydantic 模型) —— 约束 Agent 返回可被程序解析的数据
# ============================================================
class BookingRecommendation(BaseModel):
    destination: str
    available: bool
    flight_details: str
    estimated_cost: int


class TravelPlan(BaseModel):
    recommendations: list[BookingRecommendation]


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
# Demo 1: 组合多个工具 —— Agent 自动串行调用工具回答复杂问题
# ============================================================
async def demo_multiple_tools():
    """Agent 持有多个工具，依需自动调用 (如先查目的地、再逐一查可用性)。"""
    print("=" * 60)
    print("Demo 1: Agent with Multiple Tools (自动串行调用)")
    print("=" * 60)

    travel_tools = [get_destinations, check_availability, get_flight_info]

    agent = client.as_agent(
        name="TravelToolAgent",
        instructions=(
            "You are a travel agent. Use the available tools to answer questions "
            "about destinations, availability, and flights. Always call the tools "
            "to get real data rather than guessing."
        ),
        tools=travel_tools,
    )

    response = await agent.run(
        "What destinations do you have? Which ones are still available?"
    )
    print("\n--- Agent 回复 ---")
    print(response.text)
    print()


# ============================================================
# Demo 2: 结构化输出 —— Pydantic 模型约束 + 校验
# ============================================================
async def demo_structured_output():
    """让 Agent 返回可被程序解析、校验的结构化数据，而非自由文本。"""
    print("=" * 60)
    print("Demo 2: Structured Output with Tools (Pydantic 校验)")
    print("=" * 60)

    structured_agent = client.as_agent(
        name="StructuredTravelAgent",
        instructions=(
            "You are a travel agent. Use the available tools to find destinations, "
            "check availability, and get flight info.\n\n"
            "The traveler wants to fly from London Heathrow (airport code LHR) to "
            "somewhere warm in Europe. Use check_availability to verify which "
            "European destinations are available, and get_flight_info to look up "
            "flights from LHR (e.g. LHR->BCN for Barcelona, LHR->CDG for Paris).\n\n"
            "IMPORTANT: Respond with ONLY valid JSON (no markdown, no code fences, "
            "no extra commentary) matching exactly this schema:\n"
            "{\n"
            '  "recommendations": [\n'
            '    {"destination": str, "available": bool, "flight_details": str, '
            '"estimated_cost": int}\n'
            "  ]\n"
            "}\n"
            "Only include destinations you verified with the tools. "
            "estimated_cost must be an integer dollar amount (e.g. 350), "
            "extracted from the flight info."
        ),
        tools=[get_destinations, check_availability, get_flight_info],
    )

    response = await structured_agent.run(
        "I want to fly from London Heathrow to somewhere warm in Europe. "
        "Check what's available and give me the flight details."
    )

    # 将 Agent 的文本回复解析为结构化数据并用 Pydantic 校验
    raw = _extract_json(response.text)
    try:
        result = TravelPlan.model_validate(json.loads(raw))
        print("✓ 结构化解析成功，Pydantic 校验通过:\n")
        print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
        # 演示下游代码可放心访问强类型字段
        print("\n下游按字段访问:")
        for rec in result.recommendations:
            print(
                f"  → {rec.destination}: available={rec.available}, "
                f"cost=${rec.estimated_cost}"
            )
    except (json.JSONDecodeError, ValidationError) as e:
        # 解析失败时回退展示原始回复，便于排查
        print("✗ 解析/校验失败，以下是 Agent 原始回复:")
        print(response.text)
        print(f"\n错误: {e}")
    print()


# ============================================================
# Demo 3: 工具审批控制 —— approval_mode
# ============================================================
async def demo_tool_approval():
    """approval_mode 控制工具调用是否需要人工确认 (人在回路)。"""
    print("=" * 60)
    print("Demo 3: Tool Approval Patterns (approval_mode)")
    print("=" * 60)

    # 对比只读工具与敏感操作工具的审批模式
    print("只读 / 查询类工具 (自动执行，无需确认):")
    print(
        f"  {check_availability.name:.<24} approval_mode = "
        f"{check_availability.approval_mode}"
    )
    print(
        f"  {get_flight_info.name:.<24} approval_mode = "
        f"{get_flight_info.approval_mode}"
    )

    print("\n敏感操作工具 (每次调用都需人工确认):")
    print(
        f"  {book_flight.name:.<24} approval_mode = "
        f"{book_flight.approval_mode}"
    )

    print("\n说明:")
    print(
        '  - "never_require": 工具自动执行，适合只读 / 查询 '
        "(如查目的地、查航班信息)"
    )
    print(
        '  - "always_require": 每次调用前需人工确认，'
        "适合有副作用的操作 (如预订航班、扣款)"
    )
    print(
        "  在交互式运行器中，always_require 的工具会暂停并请求用户确认；"
    )
    print('  用户确认后才真正执行，从而实现"人在回路"，避免误操作。')
    print()


async def main():
    await demo_multiple_tools()
    await demo_structured_output()
    await demo_tool_approval()


if __name__ == "__main__":
    asyncio.run(main())
