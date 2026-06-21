"""
Lesson 09 - Metacognition Design Pattern (纯 Python 版)

核心主题: 演示"元认知设计模式 (Metacognition Design Pattern)"，以"航班查询 Agent"
场景串联:
  1. 主备工具与错误恢复 (Fallback Strategy) — 主工具失败时 Agent 自动检测错误、
     透明切换到备用工具，体现"对自身行为的监控与纠错"
  2. 自反思 Agent (Self-Reflecting Agent) — Agent 在指令中被要求解释发生了什么、
     透明地汇报回退过程，并在每次回复后自我评估是否完整回答了用户问题
  3. 自评估模式 (Self-Evaluation Pattern) — 用独立的评估 Agent 对前一个 Agent 的
     回复从完整性、准确性、有用性三个维度打分并提出改进建议

对应 README 概念:
  - Metacognition = "thinking about thinking"
  - Self-Reflection: Agent 监控自身推理过程
  - Error Correction: 检测错误并自主纠正 (fallback)
  - Adaptation: 根据反馈调整策略

API 变更说明（agent-framework >= 1.8.0）:
  - AzureAIProjectAgentProvider 已移除 → 改用 OpenAIChatCompletionClient + GitHub Models
  - await provider.create_agent(...) 已废弃 → 改用 client.as_agent(...)
  - 工具异常处理: @tool 函数抛出异常后，框架会把异常信息作为工具结果返回给 Agent，
    Agent 据此感知失败并调用备用工具——这正是元认知"错误检测→策略调整"的体现
"""

import os
import sys
import asyncio
from typing import Annotated

from dotenv import load_dotenv
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
# 主备工具 (Primary & Backup Tools)
#
# 对应知识点: Fallback Strategy — 元认知中的"错误检测与恢复"
#   - 主工具覆盖部分目的地，查不到时抛出 404 异常
#   - 备用工具覆盖另一批目的地，作为兜底
#   - Agent 先试主工具，失败后自主切换到备用工具
# ============================================================
@tool(approval_mode="never_require")
def get_flight_times(
    destination: Annotated[str, "The destination city"],
) -> str:
    """Get available flight times for a destination (primary source)."""
    print(f"  [tool] get_flight_times({destination!r}) called...")
    flights = {
        "Paris": "Departures: 08:00, 12:30, 17:45 — from $350",
        "Tokyo": "Departures: 11:00, 23:30 — from $890",
        "Barcelona": "Departures: 07:15, 14:00, 19:30 — from $280",
    }
    if destination in flights:
        return flights[destination]
    # 主系统查不到 → 抛异常，框架会把错误信息返回给 Agent
    raise Exception(f"404: No flights found for {destination} in primary system")


@tool(approval_mode="never_require")
def get_flight_times_backup(
    destination: Annotated[str, "The destination city"],
) -> str:
    """Get available flight times from backup system (used when primary fails)."""
    print(f"  [tool] get_flight_times_backup({destination!r}) called...")
    backup_flights = {
        "Berlin": "Departures: 09:00, 16:00 — from $220",
        "Sydney": "Departures: 22:00 — from $1200",
        "New York City": "Departures: 06:00, 10:30, 15:00, 20:00 — from $450",
    }
    return backup_flights.get(
        destination,
        f"No flights found for {destination} in any system. Please try again later.",
    )


# ============================================================
# Demo 1: 自反思 Agent — 主备工具回退与错误恢复
#
# 对应知识点:
#   - Self-Reflection: Agent 监控自身工具调用结果，感知失败
#   - Error Recovery: 主工具 404 后，透明切换到备用工具
#   - Transparency: 向用户解释发生了什么 (回退过程)
#
# 两个测试:
#   Test 1 — 目的地在主系统中 (Paris): 主工具直接命中
#   Test 2 — 目的地仅在备用系统中 (Berlin): 主工具 404 → 回退到备用工具
# ============================================================
async def demo_self_reflecting_agent() -> str:
    """自反思 Agent: 先试主工具，失败则回退到备用工具，并透明汇报。"""
    print("=" * 60)
    print("Demo 1: Self-Reflecting Agent — Fallback & Error Recovery")
    print("=" * 60)

    agent = client.as_agent(
        name="FlightBookingAgent",
        instructions=(
            "You are a flight booking agent with self-reflection capabilities.\n\n"
            "When looking up flights:\n"
            "1. Try the primary flight system first (get_flight_times)\n"
            "2. If the primary system fails (404 error), acknowledge the error "
            "and try the backup system (get_flight_times_backup)\n"
            "3. Always explain to the user what happened — be transparent about "
            "fallbacks\n"
            "4. If both systems fail, apologize and suggest alternatives\n\n"
            "After each response, briefly evaluate whether your answer was "
            "complete and helpful."
        ),
        tools=[get_flight_times, get_flight_times_backup],
    )

    # Test 1: 目的地在主系统中 —— 主工具直接命中，无需回退
    print("\n--- Test 1: 目的地在主系统中 (Paris) ---")
    response1 = await agent.run("What flights are available to Paris?")
    print("\n[Agent 回复]:")
    print(response1.text)

    # Test 2: 目的地仅在备用系统中 —— 主工具 404，回退到备用工具
    print("\n--- Test 2: 目的地仅在备用系统中 (Berlin) ---")
    response2 = await agent.run("What flights are available to Berlin?")
    print("\n[Agent 回复]:")
    print(response2.text)

    print()
    return response2.text  # 供 Demo 2 评估使用


# ============================================================
# Demo 2: 自评估模式 — 独立评估 Agent 打分
#
# 对应知识点:
#   - Self-Evaluation: 用另一个 Agent (或同一 Agent 的第二轮) 审查回复质量
#   - 三维评分: Completeness (完整性) / Accuracy (准确性) / Helpfulness (有用性)
#   - 改进建议: 评估 Agent 不仅打分，还提出一条改进建议
# ============================================================
async def demo_self_evaluation(agent_response: str):
    """独立评估 Agent 对航班查询回复进行三维打分。"""
    print("=" * 60)
    print("Demo 2: Self-Evaluation — Response Evaluator Agent")
    print("=" * 60)

    evaluation_agent = client.as_agent(
        name="ResponseEvaluator",
        instructions=(
            "You are a quality evaluator for travel agent responses.\n"
            "Given a travel question and the agent's response, evaluate:\n"
            "1. Completeness: Did it answer all parts of the question? (1-5)\n"
            "2. Accuracy: Is the information correct? (1-5)\n"
            "3. Helpfulness: Would a traveler find this useful? (1-5)\n"
            "Provide a brief evaluation with scores and one suggestion for "
            "improvement."
        ),
        tools=[get_flight_times, get_flight_times_backup],
    )

    eval_prompt = (
        "Question: What flights are available to Berlin?\n"
        f"Agent Response: {agent_response}\n\n"
        "Please evaluate the above response."
    )

    print("\n调用评估 Agent (ResponseEvaluator)...\n")
    evaluation = await evaluation_agent.run(eval_prompt)
    print("--- 评估结果 ---")
    print(evaluation.text)
    print()


async def main():
    # Demo 1: 自反思 Agent — 主备工具回退，返回 Berlin 的回复供 Demo 2 评估
    berlin_response = await demo_self_reflecting_agent()
    # Demo 2: 自评估 — 评估 Agent 对 Demo 1 的回复打分
    await demo_self_evaluation(berlin_response)


if __name__ == "__main__":
    asyncio.run(main())
