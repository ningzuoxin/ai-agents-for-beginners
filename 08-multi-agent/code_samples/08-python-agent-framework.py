"""
Lesson 08 - Multi-Agent Design Pattern (纯 Python 版)

核心主题: 演示"多智能体设计模式 (Multi-Agent Design Pattern)"，以"旅行规划"场景串联:
  1. 创建专家 Agent (Specialized Agents) — 规划师、礼宾、预算审核员各司其职
  2. 顺序工作流 (Sequential Workflow) — 用 WorkflowBuilder + add_edge 把 Agent 串成流水线
  3. 流式输出 (Streaming Output) — 实时追踪每个 Agent 的发言，观察 Agent 间的协作过程
  4. 扩展工作流 (Extend Workflow) — 不改动既有 Agent，直接往链上追加新 Agent

设计理念 (对应 README):
  - 专业化 (Specialization): 每个 Agent 专注一个领域，比通才 Agent 质量更高
  - 可扩展 (Scalability): 新增 Agent 不需重写既有工作流，只需 add_edge 接入
  - 可观测 (Visibility): 流式输出让 Agent 间的信息传递一目了然，便于调试与优化

API 变更说明（agent-framework >= 1.8.0）:
  - AzureAIProjectAgentProvider 已移除 → 改用 OpenAIChatCompletionClient + GitHub Models
  - await provider.create_agent(...) 已废弃 → 改用 client.as_agent(...)
  - WorkflowBuilder 需显式传 output_from="all"，否则触发 DeprecationWarning
    (控制每个 executor 的输出是否作为 'output' 事件流出，'all' 表示全部流出)
  - 流式事件: event.type == "output" 时 event.data 为 AgentResponseUpdate，
    其 .author_name 标识当前发言的 Agent，.text 为增量文本块
"""

import os
import sys
import asyncio

from dotenv import load_dotenv
from agent_framework import AgentResponseUpdate, WorkflowBuilder
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
# 创建专家 Agent (Specialized Agents)
#
# 对应知识点: Specialization
#   - 每个 Agent 拥有聚焦的职责与专属指令，而非一个通才包揽全部
#   - 规划师负责起草行程，礼宾负责评审增强，预算审核员负责控成本
# ============================================================
def create_specialized_agents():
    """创建三个职责各异的专家 Agent。"""
    planner_agent = client.as_agent(
        name="TravelPlanner",
        instructions=(
            "You are a travel planning specialist. Create detailed trip itineraries "
            "based on the traveler's preferences. Include daily schedules, must-see "
            "attractions, and logistical tips."
        ),
    )

    concierge_agent = client.as_agent(
        name="TravelConcierge",
        instructions=(
            "You are a travel concierge who reviews and enhances trip plans. Review "
            "the plan for completeness, add local insider tips, suggest restaurants, "
            "and identify potential issues. Provide your feedback in a constructive "
            "format."
        ),
    )

    budget_agent = client.as_agent(
        name="BudgetReviewer",
        instructions=(
            "You are a budget-conscious travel advisor. Review the proposed trip plan "
            "and concierge enhancements against the traveler's stated budget. Estimate "
            "costs for flights, hotels, meals, and activities. Flag anything that risks "
            "exceeding the budget and suggest cost-saving alternatives while preserving "
            "the trip's quality."
        ),
    )

    return planner_agent, concierge_agent, budget_agent


# ============================================================
# 流式输出辅助函数 — 实时打印工作流中每个 Agent 的发言
#
# 对应知识点: Visibility into Multi-Agent Interactions
#   - 流式遍历 WorkflowEvent，按 author_name 分段，清晰展示 Agent 间的协作过程
#   - event.type == "output" → Agent 产出的增量文本 (AgentResponseUpdate)
#   - event.type == "executor_invoked" / "executor_completed" → Agent 调用起止
#   - event.type == "superstep_*" → 工作流的"超步"边界 (一轮 Agent 执行)
# ============================================================
async def stream_workflow(workflow, user_request: str):
    """流式运行工作流，按 Agent 分段打印输出。"""
    last_author = None
    # stream=True 同步返回 ResponseStream（异步迭代器），无需 await
    events = workflow.run(user_request, stream=True)
    async for event in events:
        if event.type == "output" and isinstance(event.data, AgentResponseUpdate):
            update = event.data
            author = update.author_name
            # 检测 Agent 切换，打印分隔标题
            if author != last_author:
                if last_author is not None:
                    print()
                print(f"\n{'=' * 50}")
                print(f"[{author}]:")
                print(f"{'=' * 50}")
                last_author = author
            # 打印增量文本块（流式效果）
            print(update.text, end="", flush=True)
    print()


# ============================================================
# Demo 1: 顺序工作流 — TravelPlanner → TravelConcierge
#
# 对应知识点:
#   - Sequential Workflow: WorkflowBuilder 把 Agent 串成有向图流水线
#   - add_edge(source, target): 前一个 Agent 的输出作为后一个 Agent 的输入
#   - Streaming: 实时观察规划师起草 → 礼宾评审增强的协作过程
# ============================================================
async def demo_sequential_workflow():
    """两步流水线: 规划师起草行程，礼宾评审增强。"""
    print("=" * 60)
    print("Demo 1: Sequential Workflow — Planner -> Concierge")
    print("=" * 60)

    planner_agent, concierge_agent, _ = create_specialized_agents()

    # 用 WorkflowBuilder 构建顺序工作流: planner -> concierge
    # output_from="all" 让每个 Agent 的输出都作为 'output' 事件流出
    workflow = (
        WorkflowBuilder(start_executor=planner_agent, output_from="all")
        .add_edge(planner_agent, concierge_agent)
        .build()
    )

    user_request = (
        "Plan a 5-day trip to Paris for a food-loving couple on a $3000 budget."
    )
    print(f"\n用户请求: {user_request}")
    print("\n启动顺序工作流 (流式输出):\n")

    await stream_workflow(workflow, user_request)
    print()


# ============================================================
# Demo 2: 扩展工作流 — 追加 BudgetReviewer
#
# 对应知识点:
#   - Extend Workflow: 不改动既有 Agent，直接往链上 add_edge 新 Agent
#   - Scalability: 新增 Agent 无需重写工作流，体现多 Agent 模式的可扩展性
#   - 三段式流水线: 规划师 → 礼宾 → 预算审核员，逐步精炼结果
# ============================================================
async def demo_extended_workflow():
    """三步流水线: 规划师 → 礼宾 → 预算审核员，展示工作流的可扩展性。"""
    print("=" * 60)
    print("Demo 2: Extended Workflow — Planner -> Concierge -> BudgetReviewer")
    print("=" * 60)

    planner_agent, concierge_agent, budget_agent = create_specialized_agents()

    # 在原有两步链上追加 budget_agent，构建三步流水线
    # 既有 Agent 与边不变，只需新增一条边——这正是可扩展性的体现
    extended_workflow = (
        WorkflowBuilder(start_executor=planner_agent, output_from="all")
        .add_edge(planner_agent, concierge_agent)
        .add_edge(concierge_agent, budget_agent)
        .build()
    )

    user_request = (
        "Plan a 5-day trip to Paris for a food-loving couple on a $3000 budget."
    )
    print(f"\n用户请求: {user_request}")
    print("\n启动扩展工作流 (三段式，流式输出):\n")

    await stream_workflow(extended_workflow, user_request)
    print()


async def main():
    # Demo 1: 两步顺序工作流 (规划师 → 礼宾)
    await demo_sequential_workflow()
    # Demo 2: 扩展为三步工作流 (规划师 → 礼宾 → 预算审核员)
    await demo_extended_workflow()


if __name__ == "__main__":
    asyncio.run(main())
