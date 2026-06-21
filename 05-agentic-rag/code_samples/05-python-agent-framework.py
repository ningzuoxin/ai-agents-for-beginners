"""
Lesson 05 - Agentic RAG (纯 Python 版)

核心主题: 演示 Agentic RAG（智能体驱动的检索增强生成）模式，以"旅行知识库"场景串联:
  1. 把知识库包装为工具 — @tool 装饰器让 Agent 按需检索外部数据源（替代固定管线）
  2. 构建 RAG Agent — 指令约束"先检索再回答"，答案扎根于检索结果而非模型记忆
  3. 迭代检索 (Maker-Checker 模式) — Agent 多轮搜索、验证、补全，直到信息充分
  4. 自纠错与查询改写 — 检索无果时 Agent 自主改写查询、换关键词重试

API 变更说明（agent-framework >= 1.8.0）:
  - AzureAIProjectAgentProvider 已移除 → 改用 OpenAIChatCompletionClient + GitHub Models
  - await provider.create_agent(...) 已废弃 → 改用 client.as_agent(...)
  - 原笔记本使用 Azure AI Search 作为数据源，本版用内存知识库模拟；
    生产环境中只需把 search_travel_knowledge 的实现替换为对 Azure AI Search 的调用即可。
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
# 知识库: 模拟外部数据源（生产环境可替换为 Azure AI Search 索引）
# ============================================================
TRAVEL_KNOWLEDGE_BASE = {
    "Barcelona": (
        "Barcelona is Spain's cosmopolitan capital of Catalonia. "
        "Best visited Mar-May or Sep-Nov. Known for Gaudi architecture, "
        "La Rambla, beaches. Average daily cost: $150-200."
    ),
    "Tokyo": (
        "Tokyo is Japan's capital, mixing ultramodern with traditional. "
        "Best visited Mar-Apr (cherry blossoms) or Oct-Nov. "
        "Known for Shibuya, temples, sushi. Average daily cost: $200-250."
    ),
    "Paris": (
        "Paris is France's capital and a global center for art, fashion, "
        "and culture. Best visited Apr-Jun or Sep-Oct. Known for Eiffel "
        "Tower, Louvre, cuisine. Average daily cost: $180-250."
    ),
    "Cape Town": (
        "Cape Town sits on South Africa's southwest tip. "
        "Best visited Nov-Mar. Known for Table Mountain, wine regions, "
        "wildlife. Average daily cost: $100-150."
    ),
}


# ============================================================
# 检索工具: 把知识库暴露为 Agent 可调用的 @tool
#
# 要点:
#   - docstring  → 模型看到的"工具描述"，告诉它何时该调用此工具
#   - Annotated  → 参数说明，帮助模型构造正确的查询
#   - approval_mode="never_require" → 只读检索，自动执行无需审批
# ============================================================
@tool(approval_mode="never_require")
def search_travel_knowledge(
    query: Annotated[str, "The search query about a travel destination"],
) -> str:
    """Search the travel knowledge base for destination information."""
    print(f"  [tool] search_travel_knowledge({query!r}) called...")
    results = []
    for destination, info in TRAVEL_KNOWLEDGE_BASE.items():
        # 匹配条件（满足任一即可）：
        #   1. 完整查询词出现在目的地名中（如 "tokyo" 匹配 "Tokyo"）
        #   2. 查询词拆分后的任意单词出现在目的地信息中（如 "hiking tokyo" 拆成
        #      ["hiking", "tokyo"]，只要其中一个词命中信息就算匹配）
        if query.lower() in destination.lower() or any(
            word in info.lower() for word in query.lower().split()
        ):
            results.append(f"**{destination}**: {info}")
    return (
        "\n\n".join(results)
        if results
        else "No matching destinations found in the knowledge base."
    )


# ============================================================
# Demo 1: 构建 RAG Agent — "先检索再回答"
#
# 对应知识点:
#   - Agentic RAG: Agent 自主决定何时检索（而非固定管线）
#   - 工具即数据源: 知识库包装为 @tool，Agent 按需调用
# ============================================================
async def demo_rag_agent():
    """RAG Agent 被指令约束为"回答前必须先检索"，答案扎根于检索结果。"""
    print("=" * 60)
    print("Demo 1: RAG Agent — Always Retrieve Before Answering")
    print("=" * 60)

    agent = client.as_agent(
        name="TravelRAGAgent",
        instructions=(
            "You are a knowledgeable travel advisor. Before answering questions "
            "about destinations:\n"
            "1. ALWAYS search the travel knowledge base first\n"
            "2. Base your answers on retrieved information\n"
            "3. If information is not in the knowledge base, say so clearly\n"
            "4. Provide specific details like costs, best seasons, and highlights."
        ),
        tools=[search_travel_knowledge],
    )

    response = await agent.run(
        "I'm interested in visiting somewhere with great architecture. "
        "What destinations would you recommend?"
    )
    print("\n--- Agent 回复 ---")
    print(response.text)
    print()


# ============================================================
# Demo 2: 迭代检索 — Maker-Checker 模式
#
# 对应知识点:
#   - 迭代检索: Agent 多轮搜索，先检索初稿、再验证补全
#   - Maker-Checker: "生成 → 验证 → 补全"循环，直到信息充分
# ============================================================
async def demo_iterative_retrieval():
    """Checker Agent 对每个目的地逐一检索验证，多轮搜索后给出对比推荐。"""
    print("=" * 60)
    print("Demo 2: Iterative Retrieval — Maker-Checker Pattern")
    print("=" * 60)

    checker_agent = client.as_agent(
        name="TravelRAGCheckerAgent",
        instructions=(
            "You are a meticulous travel advisor who double-checks recommendations.\n"
            "When answering travel questions:\n"
            "1. Search for relevant destinations first\n"
            "2. For each destination found, search again with the destination name "
            "to get full details\n"
            "3. Compare the options using verified information\n"
            "4. Present a final recommendation with specific costs, best travel "
            "times, and highlights\n"
            "5. If any detail seems incomplete, search once more to confirm "
            "before responding."
        ),
        tools=[search_travel_knowledge],
    )

    response = await checker_agent.run(
        "I have a $175/day budget and want to travel in April. "
        "Which destinations fit my budget and timing?"
    )
    print("\n--- Agent 回复 ---")
    print(response.text)
    print()


# ============================================================
# Demo 3: 自纠错与查询改写
#
# 对应知识点:
#   - 自主推理: Agent 拥有推理过程，遇检索失败时自主改写查询
#   - 自纠错: 不返回低质量结果，而是换关键词、换策略重试
#     (对应 README "Handling Failure Modes and Self-Correction")
# ============================================================
async def demo_self_correction():
    """Agent 遇到检索无果时，自主改写查询、换关键词重试。"""
    print("=" * 60)
    print("Demo 3: Self-Correction — Query Rewriting on Retrieval Failure")
    print("=" * 60)

    self_correcting_agent = client.as_agent(
        name="TravelSelfCorrectingAgent",
        instructions=(
            "You are a resourceful travel advisor.\n"
            "When searching the travel knowledge base:\n"
            "1. If the first search returns 'No matching destinations found', "
            "do NOT give up — rewrite your query using synonyms or related terms "
            "(e.g. if 'hiking' fails, try 'mountain', 'nature', 'outdoor', "
            "'adventure')\n"
            "2. Try at most 3 different search terms before concluding\n"
            "3. Once you find matching destinations, search again by destination "
            "name to get full details\n"
            "4. Base your final answer only on information you successfully "
            "retrieved.\n"
            "5. Briefly mention which search terms you tried so the user can "
            "see your reasoning."
        ),
        tools=[search_travel_knowledge],
    )

    response = await self_correcting_agent.run(
        "I love hiking and outdoor adventures. "
        "Which destination would you recommend?"
    )
    print("\n--- Agent 回复 ---")
    print(response.text)
    print()


async def main():
    await demo_rag_agent()
    await demo_iterative_retrieval()
    await demo_self_correction()


if __name__ == "__main__":
    asyncio.run(main())
