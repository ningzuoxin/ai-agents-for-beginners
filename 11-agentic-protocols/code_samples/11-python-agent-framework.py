"""
Lesson 11 - Agentic Protocols: A2A & MCP (纯 Python 版)

核心主题: 演示两大智能体开放协议，以"旅行规划"场景串联:

  协议 1 — Agent-to-Agent (A2A) Protocol
    Demo 1: 三个专家 Agent (汇率/活动/旅行经理) 通过 WorkflowBuilder 串联，
            模拟 A2A 的发现、消息传递与任务生命周期
  协议 2 — Model Context Protocol (MCP)
    Demo 2: @tool 工具模拟 MCP 连接的服务 (住宿搜索/本地体验)，
            Agent 在运行时"发现"并使用这些工具

对应两个原笔记本:
  - 11-a2a-agent-framework.ipynb → Demo 1 (A2A 多 Agent 协作)
  - 11-mcp-agent-framework.ipynb  → Demo 2 (MCP 工具发现)

API 变更说明（agent-framework >= 1.8.0）:
  - AzureAIProjectAgentProvider 已移除 → 改用 OpenAIChatCompletionClient + GitHub Models
  - await provider.create_agent(...) 已废弃 → 改用 client.as_agent(...)
  - WorkflowBuilder 需显式传 output_from="all"
  - 原笔记本用 @tool 函数"模拟"MCP 连接的服务；本版保持一致——
    真实 MCP 需运行 MCP Server 进程，通过 mcp client 动态发现工具。
"""

import os
import sys
import asyncio
from typing import Annotated

from dotenv import load_dotenv
from agent_framework import tool, AgentResponseUpdate, WorkflowBuilder
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
# 流式输出辅助函数 — 实时打印工作流中每个 Agent 的发言
# ============================================================
async def stream_workflow(workflow, user_request: str):
    """流式运行工作流，按 Agent 分段打印输出。"""
    last_author = None
    events = workflow.run(user_request, stream=True)
    async for event in events:
        if event.type == "output" and isinstance(event.data, AgentResponseUpdate):
            update = event.data
            author = update.author_name
            if author != last_author:
                if last_author is not None:
                    print()
                print(f"\n{'=' * 50}")
                print(f"[{author}]:")
                print(f"{'=' * 50}")
                last_author = author
            print(update.text, end="", flush=True)
    print()


# ============================================================
# Demo 1: A2A Protocol — 多 Agent 协作工作流
#
# 对应知识点:
#   - A2A Discovery: 每个 Agent 有明确的角色与能力 (Agent Card 的体现)
#   - A2A Message Passing: 前一个 Agent 的输出作为后一个 Agent 的输入
#   - A2A Task Lifecycle: 任务在工作流中依次流转 (submitted→working→completed)
#   - 三段式协作: 汇率专家 → 活动规划师 → 旅行经理 (综合)
# ============================================================
async def demo_a2a_protocol():
    """A2A 协议模拟: 三个专家 Agent 通过顺序工作流协作。"""
    print("=" * 60)
    print("Demo 1: A2A Protocol — Multi-Agent Collaboration Workflow")
    print("=" * 60)

    # --- 创建三个专家 Agent (各自有独立的角色与指令，模拟 Agent Card) ---
    currency_agent = client.as_agent(
        name="CurrencyExchangeAgent",
        instructions=(
            "You are a currency exchange specialist. You help travelers understand:\n"
            "- Current exchange rates between currencies\n"
            "- Best times to exchange money\n"
            "- Tips for getting the best rates\n"
            "When asked about a destination, provide relevant currency information."
        ),
    )

    activity_agent = client.as_agent(
        name="ActivityPlannerAgent",
        instructions=(
            "You are a local activities specialist. You recommend:\n"
            "- Must-see attractions and hidden gems\n"
            "- Local experiences and cultural activities\n"
            "- Restaurant and dining recommendations\n"
            "Tailor suggestions to the traveler's interests."
        ),
    )

    travel_manager = client.as_agent(
        name="TravelManagerAgent",
        instructions=(
            "You are a travel manager who coordinates between specialist agents.\n"
            "When planning a trip:\n"
            "1. Gather currency information from the currency specialist\n"
            "2. Get activity recommendations from the activity planner\n"
            "3. Synthesize everything into a cohesive travel brief\n"
            "Present the final plan in an organized, easy-to-read format."
        ),
    )

    # --- 构建 A2A 风格的顺序工作流: 汇率 → 活动 → 旅行经理 ---
    # 模拟 A2A 消息传递: 每个 Agent 的输出作为下一个 Agent 的输入
    workflow = (
        WorkflowBuilder(start_executor=currency_agent, output_from="all")
        .add_edge(currency_agent, activity_agent)
        .add_edge(activity_agent, travel_manager)
        .build()
    )

    user_request = (
        "Plan a week-long trip to Tokyo. I love food, temples, and technology."
    )
    print(f"\n用户请求: {user_request}")
    print("\n启动 A2A 协作工作流 (汇率 → 活动 → 旅行经理，流式输出):\n")

    await stream_workflow(workflow, user_request)
    print()


# ============================================================
# Demo 2: MCP Protocol — 动态工具发现与使用
#
# 对应知识点:
#   - MCP Client-Server 架构: Agent (client) 连接 MCP Server (tool provider)
#   - MCP Dynamic Discovery: Agent 在运行时发现可用工具 (而非硬编码)
#   - MCP Tools: search_accommodations / get_local_experiences
#     (模拟 MCP Server 暴露的工具；生产环境通过 mcp client 动态发现)
# ============================================================
@tool(approval_mode="never_require")
def search_accommodations(
    location: Annotated[str, "The city to search for accommodations"],
    check_in: Annotated[str, "Check-in date (YYYY-MM-DD)"],
    check_out: Annotated[str, "Check-out date (YYYY-MM-DD)"],
    guests: Annotated[int, "Number of guests"] = 2,
) -> str:
    """Search for accommodations (simulating an MCP-connected Airbnb tool).

    In production, this would be discovered via MCP from an accommodation service.
    """
    print(
        f"  [tool] search_accommodations({location!r}, {check_in!r}, "
        f"{check_out!r}, {guests!r}) called..."
    )
    listings = {
        "Tokyo": [
            {"name": "Shinjuku Modern Apartment", "price": 120, "rating": 4.8},
            {"name": "Traditional Ryokan in Asakusa", "price": 200, "rating": 4.9},
            {"name": "Shibuya Studio", "price": 85, "rating": 4.5},
        ],
        "Paris": [
            {"name": "Le Marais Charming Flat", "price": 150, "rating": 4.7},
            {"name": "Montmartre Artist Loft", "price": 110, "rating": 4.6},
        ],
        "Barcelona": [
            {"name": "Gothic Quarter Penthouse", "price": 130, "rating": 4.8},
            {"name": "Barceloneta Beach Flat", "price": 95, "rating": 4.4},
        ],
    }
    results = listings.get(location, [])
    if not results:
        return f"No accommodations found in {location}"
    output = f"Accommodations in {location} ({check_in} to {check_out}, {guests} guests):\n"
    for listing in results:
        output += f"  - {listing['name']}: ${listing['price']}/night (rating {listing['rating']})\n"
    return output


@tool(approval_mode="never_require")
def get_local_experiences(
    location: Annotated[str, "The city to find experiences in"],
    interest: Annotated[
        str, "Type of experience (food, culture, adventure, etc.)"
    ] = "all",
) -> str:
    """Get local experiences and activities (simulating an MCP-connected tourism tool)."""
    print(f"  [tool] get_local_experiences({location!r}, {interest!r}) called...")
    experiences = {
        "Tokyo": {
            "food": ["Tsukiji Market Tour ($45)", "Ramen Making Class ($60)", "Sake Tasting ($35)"],
            "culture": ["Tea Ceremony ($50)", "Samurai Museum ($15)", "Sumo Tournament ($80)"],
            "adventure": ["Mt. Fuji Day Trip ($120)", "Go-kart City Tour ($80)"],
        },
        "Paris": {
            "food": ["Wine & Cheese Tasting ($55)", "Cooking Class ($90)", "Market Tour ($40)"],
            "culture": ["Louvre Guided Tour ($35)", "Montmartre Art Walk ($25)"],
        },
    }
    city_exp = experiences.get(location, {})
    if not city_exp:
        return f"No experiences found in {location}"
    if interest != "all" and interest in city_exp:
        items = city_exp[interest]
        return f"{interest.title()} experiences in {location}:\n" + "\n".join(
            f"  - {e}" for e in items
        )
    output = f"All experiences in {location}:\n"
    for cat, items in city_exp.items():
        output += f"\n  {cat.title()}:\n"
        for item in items:
            output += f"    - {item}\n"
    return output


async def demo_mcp_protocol():
    """MCP 协议模拟: Agent 使用模拟 MCP 连接的工具发现住宿与体验。"""
    print("=" * 60)
    print("Demo 2: MCP Protocol — Dynamic Tool Discovery & Usage")
    print("=" * 60)

    # Agent 持有模拟 MCP 工具，在运行时"发现"并使用它们
    agent = client.as_agent(
        name="AccommodationAgent",
        instructions=(
            "You are an accommodation and travel experiences specialist "
            "powered by MCP-connected services.\n\n"
            "Help travelers find the perfect place to stay and things to do. "
            "When searching:\n"
            "1. Use the search_accommodations tool to find listings\n"
            "2. Use the get_local_experiences tool to suggest activities\n"
            "3. Compare options and make personalized recommendations\n"
            "4. Consider the traveler's budget, interests, and travel style"
        ),
        tools=[search_accommodations, get_local_experiences],
    )

    user_request = (
        "I'm visiting Tokyo for 5 nights in April with my partner. "
        "We love traditional Japanese culture and food. "
        "Find us a place to stay and suggest some experiences."
    )
    print(f"\n用户请求: {user_request}")
    print("\n调用 MCP 风格 Agent (工具动态发现):\n")

    response = await agent.run(user_request)
    print("--- Agent 回复 ---")
    print(response.text)
    print()


async def main():
    # Demo 1: A2A Protocol — 三个专家 Agent 顺序协作
    await demo_a2a_protocol()
    # Demo 2: MCP Protocol — Agent 使用模拟 MCP 连接的工具
    await demo_mcp_protocol()


if __name__ == "__main__":
    asyncio.run(main())
