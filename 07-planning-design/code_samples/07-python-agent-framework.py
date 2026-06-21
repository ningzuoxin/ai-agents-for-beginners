"""
Lesson 07 - Planning Design Pattern (纯 Python 版)

核心主题: 演示"规划设计模式 (Planning Design Pattern)"，以"旅行规划"场景串联:
  1. 任务分解 (Task Decomposition) — 用 Pydantic 模型定义结构化子任务与旅行计划
  2. 规划 Agent + 结构化输出 — 前台规划 Agent 把复杂请求分解为带优先级、依赖的子任务，
     通过 response_format 直接返回可校验的 TravelPlan 对象
  3. 计划执行 (Plan Execution) — 礼宾 Agent 持有专家工具 (book_flight / reserve_hotel /
     book_activity)，按依赖顺序逐一执行子任务并汇总结果

设计理念: 把"做什么 (规划)"与"怎么做 (执行)"分离，让 Agent 更模块化、可测试、易扩展。

API 变更说明（agent-framework >= 1.8.0）:
  - AzureAIProjectAgentProvider 已移除 → 改用 OpenAIChatCompletionClient + GitHub Models
  - await provider.create_agent(...) 已废弃 → 改用 client.as_agent(...)
  - 原笔记本隐式使用 response_format 返回 Pydantic 对象；本版通过 get_response() 的
    ChatOptions(response_format=TravelPlan) 实现原生结构化输出，再用 model_validate_json 校验。
    (注: as_agent() 不支持 response_format，因此规划 Agent 用 get_response() 直接调用，
     礼宾 Agent 需要工具则用 as_agent()。两种 API 各擅其长。)
"""

import os
import sys
import json
import asyncio
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from agent_framework import tool, Message, ChatOptions
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
# 结构化输出模型 (Pydantic) — 定义"旅行计划"的 schema
#
# 对应知识点: Task Decomposition
#   - 把复杂请求分解为多个子任务，每个子任务有明确职责
#   - priority 标识优先级，dependencies 标识任务间依赖
#   - 整个计划包含目的地、天数、预算等汇总信息
# ============================================================
class TravelSubTask(BaseModel):
    task_id: int
    description: str
    assigned_agent: str  # "flight_agent", "hotel_agent", "activity_agent"
    priority: str  # "high", "medium", "low"
    dependencies: list[int] = []  # 依赖的其他 task_id 列表


class TravelPlan(BaseModel):
    destination: str
    trip_duration_days: int
    subtasks: list[TravelSubTask]
    total_estimated_budget_usd: int
    notes: str


# ============================================================
# 专家工具 (@tool) — 礼宾 Agent 执行计划时调用的 specialists
#
# 每个工具负责一类子任务:
#   - book_flight:     航班查询与预订
#   - reserve_hotel:   酒店预订
#   - book_activity:   活动 / 门票预订
# ============================================================
@tool(approval_mode="never_require")
def book_flight(
    destination: Annotated[str, "The destination city"],
    departure_date: Annotated[str, "Departure date (YYYY-MM-DD)"],
    return_date: Annotated[str, "Return date (YYYY-MM-DD)"],
) -> str:
    """Search and book flights for the trip."""
    print(f"  [tool] book_flight({destination!r}, {departure_date!r}, {return_date!r}) called...")
    return (
        f"Flight booked to {destination}: {departure_date} -> {return_date}, "
        f"confirmation #FLT-{hash(destination) % 10000:04d}"
    )


@tool(approval_mode="never_require")
def reserve_hotel(
    city: Annotated[str, "The city for the hotel"],
    check_in: Annotated[str, "Check-in date (YYYY-MM-DD)"],
    check_out: Annotated[str, "Check-out date (YYYY-MM-DD)"],
    guests: Annotated[int, "Number of guests"],
) -> str:
    """Reserve a hotel room in the destination city."""
    print(
        f"  [tool] reserve_hotel({city!r}, {check_in!r}, {check_out!r}, {guests!r}) called..."
    )
    return (
        f"Hotel reserved in {city}: {check_in} to {check_out} for {guests} guests, "
        f"confirmation #HTL-{hash(city) % 10000:04d}"
    )


@tool(approval_mode="never_require")
def book_activity(
    activity_name: Annotated[str, "Name of the activity or tour"],
    date: Annotated[str, "Date of the activity (YYYY-MM-DD)"],
    participants: Annotated[int, "Number of participants"],
) -> str:
    """Book a tour, museum visit, or other activity."""
    print(
        f"  [tool] book_activity({activity_name!r}, {date!r}, {participants!r}) called..."
    )
    return (
        f"Activity booked: {activity_name} on {date} for {participants} people, "
        f"confirmation #ACT-{hash(activity_name) % 10000:04d}"
    )


# ============================================================
# Demo 1: 规划 Agent — 任务分解 + 结构化输出
#
# 对应知识点:
#   - Task Decomposition: 把"7 天巴黎之旅"分解为航班、酒店、活动等子任务
#   - Structured Output: response_format=TravelPlan 让 LLM 直接返回可校验的 JSON
#   - Priorities & Dependencies: 子任务带优先级与依赖，便于后续按序执行
# ============================================================
async def demo_planning_agent() -> TravelPlan | None:
    """规划 Agent 接收高层旅行请求，输出结构化的 TravelPlan。"""
    print("=" * 60)
    print("Demo 1: Planning Agent — Task Decomposition & Structured Output")
    print("=" * 60)

    planning_instructions = (
        "You are a travel planning agent. When given a travel request:\n"
        "1. Break it into specific subtasks (flights, hotels, activities, logistics)\n"
        "2. Assign each subtask to the appropriate specialist agent "
        '(use "flight_agent", "hotel_agent", or "activity_agent")\n'
        "3. Set priorities (high/medium/low) and identify dependencies between tasks\n"
        "4. Estimate the total budget\n"
        "5. Add any helpful notes for the traveler"
    )

    user_request = (
        "Plan a 7-day trip to Paris for a couple interested in art, "
        "cuisine, and history. Budget around $5000."
    )

    print(f"\n用户请求: {user_request}")
    print(f"\n调用规划 Agent (response_format=TravelPlan)...\n")

    response = await client.get_response(
        messages=[
            Message(role="system", contents=planning_instructions),
            Message(role="user", contents=user_request),
        ],
        options=ChatOptions(
            response_format=TravelPlan,
            temperature=0.7,
        ),
    )

    # response_format 约束 LLM 输出 JSON，再用 Pydantic 校验
    try:
        plan = TravelPlan.model_validate_json(response.text)
        print("✓ 结构化解析成功，Pydantic 校验通过:\n")
        print(f"  目的地:   {plan.destination}")
        print(f"  天数:     {plan.trip_duration_days} days")
        print(f"  预算:     ${plan.total_estimated_budget_usd}")
        print(f"  备注:     {plan.notes}")
        print(f"\n  子任务 ({len(plan.subtasks)} 个):")
        for task in plan.subtasks:
            deps = f", deps={task.dependencies}" if task.dependencies else ""
            print(
                f"    [{task.priority:>6}] {task.task_id}. {task.description}"
                f"  -> {task.assigned_agent}{deps}"
            )
        print()
        return plan
    except ValidationError as e:
        print("✗ Pydantic 校验失败，以下是 Agent 原始回复:")
        print(response.text)
        print(f"\n错误: {e}")
        print()
        return None


# ============================================================
# Demo 2: 礼宾 Agent — 按计划执行子任务
#
# 对应知识点:
#   - Plan Execution: 把规划 Agent 的输出转化为执行 prompt，交给礼宾 Agent
#   - Specialist Tools: 礼宾 Agent 持有 book_flight / reserve_hotel / book_activity
#   - Dependency Ordering: 指令要求礼宾 Agent 尊重子任务间的依赖顺序
# ============================================================
async def demo_execute_plan(plan: TravelPlan):
    """礼宾 Agent 持有专家工具，按依赖顺序执行旅行计划并汇总结果。"""
    print("=" * 60)
    print("Demo 2: Concierge Agent — Execute the Plan with Specialist Tools")
    print("=" * 60)

    # 把结构化计划拼接为礼宾 Agent 可理解的执行 prompt
    subtask_lines = "\n".join(
        f"  - [{t.priority}] {t.task_id}. {t.description} "
        f"(agent: {t.assigned_agent}, deps: {t.dependencies})"
        for t in plan.subtasks
    )
    execution_prompt = (
        f"Execute the following travel plan for {plan.destination} "
        f"({plan.trip_duration_days} days, ${plan.total_estimated_budget_usd} budget):\n"
        f"{subtask_lines}\n\n"
        f"Use the available tools to fulfil each subtask. Work through the subtasks "
        f"in order, respecting dependencies. Use reasonable dates. "
        f"Summarise the results when finished."
    )

    print(f"\n执行 prompt (摘要):\n{execution_prompt[:200]}...\n")

    concierge_agent = client.as_agent(
        name="Concierge",
        instructions=(
            "You are a travel concierge executing a structured travel plan. "
            "Use the available tools to fulfil each subtask. Work through the "
            "subtasks in order, respecting dependencies. Summarise the results "
            "when finished."
        ),
        tools=[book_flight, reserve_hotel, book_activity],
    )

    response = await concierge_agent.run(execution_prompt)
    print("--- 礼宾 Agent 回复 ---")
    print(response.text)
    print()


# ============================================================
# Demo 3: 迭代规划 — 基于用户反馈重新规划
#
# 对应知识点 (README "Iterative Planning"):
#   - 子任务结果可能影响后续计划 (如航班时间变了，酒店也要调整)
#   - 用户反馈触发部分重新规划 (如用户想改预算或天数)
#   - 把"当前计划 + 新需求"一起传给规划 Agent，生成更新后的计划
# ============================================================
async def demo_iterative_planning(original_plan: TravelPlan):
    """基于用户反馈（调整预算与天数），让规划 Agent 重新规划。"""
    print("=" * 60)
    print("Demo 3: Iterative Planning — Re-plan Based on User Feedback")
    print("=" * 60)

    feedback = (
        "The traveler changed their mind: they now want a 5-day trip instead of 7, "
        "and the budget is reduced to $3000. Please update the plan accordingly."
    )

    # 把原始计划序列化为 JSON，连同新反馈一起交给规划 Agent
    original_plan_json = original_plan.model_dump_json(indent=2)

    replan_instructions = (
        "You are a travel planning agent. You previously created a travel plan. "
        "The traveler has provided new feedback. Update the plan to reflect the "
        "new requirements while keeping the same destination and interests.\n\n"
        "Rules:\n"
        "1. Adjust subtasks to fit the new duration and budget\n"
        "2. Keep priorities and dependencies consistent\n"
        "3. Update the total estimated budget\n"
        "4. Note what changed in the notes field"
    )

    user_message = (
        f"Previous plan:\n{original_plan_json}\n\n"
        f"New feedback:\n{feedback}"
    )

    print(f"\n用户反馈: {feedback}")
    print(f"\n调用规划 Agent 重新规划 (response_format=TravelPlan)...\n")

    response = await client.get_response(
        messages=[
            Message(role="system", contents=replan_instructions),
            Message(role="user", contents=user_message),
        ],
        options=ChatOptions(
            response_format=TravelPlan,
            temperature=0.7,
        ),
    )

    try:
        updated_plan = TravelPlan.model_validate_json(response.text)
        print("✓ 重新规划成功:\n")
        print(f"  目的地:   {updated_plan.destination}")
        print(f"  天数:     {updated_plan.trip_duration_days} days "
              f"(原: {original_plan.trip_duration_days})")
        print(f"  预算:     ${updated_plan.total_estimated_budget_usd} "
              f"(原: ${original_plan.total_estimated_budget_usd})")
        print(f"  备注:     {updated_plan.notes}")
        print(f"\n  更新后的子任务 ({len(updated_plan.subtasks)} 个):")
        for task in updated_plan.subtasks:
            deps = f", deps={task.dependencies}" if task.dependencies else ""
            print(
                f"    [{task.priority:>6}] {task.task_id}. {task.description}"
                f"  -> {task.assigned_agent}{deps}"
            )
        print()
    except ValidationError as e:
        print("✗ Pydantic 校验失败，以下是 Agent 原始回复:")
        print(response.text)
        print(f"\n错误: {e}")
        print()


async def main():
    # Demo 1: 规划 Agent 分解任务，生成结构化 TravelPlan
    plan = await demo_planning_agent()
    if plan is None:
        print("规划失败，跳过后续 Demo。")
        return

    # Demo 2: 礼宾 Agent 按计划执行子任务
    await demo_execute_plan(plan)

    # Demo 3: 基于用户反馈迭代重新规划
    await demo_iterative_planning(plan)


if __name__ == "__main__":
    asyncio.run(main())
