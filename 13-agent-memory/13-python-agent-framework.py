"""
Lesson 13 - Agent Memory (纯 Python 版)

核心主题: 演示 AI Agent 的三种记忆类型及其实现，以"旅行预订"与"编程助手"场景串联:
  1. 工作记忆 (Working Memory) — agent.create_session() 提供会话内上下文
  2. 短期记忆 (Short-Term Memory) — 会话内积累的事实 (如偏好、预算)
  3. 长期记忆 (Long-Term Memory) — 跨会话持久化，通过 @tool 工具访问外部存储

对应两个原笔记本:
  - 13-agent-memory.ipynb        → Demo 1 (工作记忆) + Demo 2 (长期记忆: @tool 偏好存储)
  - 13-agent-memory-cognee.ipynb → Demo 3 (知识图谱式记忆: 结构化知识 + 语义搜索)

Cognee 版说明:
  原笔记本使用 Cognee 知识图谱 (需 Redis + LLM_API_KEY 等额外依赖)。
  本版用内存知识库 + 关键词检索模拟知识图谱的核心模式:
    - cognee.add()       → _knowledge_base.append()
    - cognee.cognify()   → 结构化数据已预存
    - cognee.search()    → search_knowledge() 工具按关键词检索
  生产环境替换为真实 Cognee 或 Azure AI Search 即可，知识点不变。

API 变更说明（agent-framework >= 1.8.0）:
  - AzureAIProjectAgentProvider 已移除 → 改用 OpenAIChatCompletionClient + GitHub Models
  - await provider.create_agent(...) 已废弃 → 改用 client.as_agent(...)
  - agent.create_session() + agent.run(session=...) 支持多轮对话
"""

import os
import sys
import json
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
# 长期记忆: 偏好存储 (对应 13-agent-memory.ipynb)
#
# 模拟持久化偏好数据库 (生产环境可替换为 Mem0 / Azure AI Search)
# ============================================================
preference_store: dict[str, list[str]] = {}


@tool(approval_mode="never_require")
def save_preference(
    user_id: Annotated[str, "User identifier"],
    preference: Annotated[str, "A travel preference to remember"],
) -> str:
    """Save a user travel preference to long-term memory."""
    print(f"  [tool] save_preference({user_id!r}, {preference!r}) called...")
    preference_store.setdefault(user_id, []).append(preference)
    return f"Stored: {preference}"


@tool(approval_mode="never_require")
def get_preferences(
    user_id: Annotated[str, "User identifier"],
) -> str:
    """Retrieve all saved travel preferences for a user."""
    print(f"  [tool] get_preferences({user_id!r}) called...")
    prefs = preference_store.get(user_id, [])
    if not prefs:
        return f"No saved preferences for {user_id}."
    return "Saved preferences:\n- " + "\n- ".join(prefs)


@tool(approval_mode="never_require")
def search_hotels(
    query: Annotated[str, "Search query — location, amenities, or tags"],
) -> str:
    """Search the hotel database for matching properties."""
    print(f"  [tool] search_hotels({query!r}) called...")
    hotels = [
        {"name": "Le Meurice Paris", "location": "Paris, France", "price": 850, "tags": ["luxury", "romantic", "spa"]},
        {"name": "Four Seasons Maui", "location": "Maui, Hawaii", "price": 695, "tags": ["beach", "family", "resort"]},
        {"name": "Aman Tokyo", "location": "Tokyo, Japan", "price": 780, "tags": ["luxury", "city", "spa"]},
        {"name": "Hotel Sacher Vienna", "location": "Vienna, Austria", "price": 420, "tags": ["historic", "accessible", "cultural"]},
        {"name": "Fairmont Whistler", "location": "Whistler, Canada", "price": 380, "tags": ["ski", "family", "mountain"]},
    ]
    q = query.lower()
    matches = [
        h for h in hotels
        if q in h["name"].lower()
        or q in h["location"].lower()
        or any(q in t for t in h["tags"])
    ]
    if not matches:
        matches = hotels[:3]
    return json.dumps(matches, indent=2)


# ============================================================
# 长期记忆: 知识图谱式存储 (对应 13-agent-memory-cognee.ipynb)
#
# 模拟 Cognee 知识图谱:
#   - cognee.add() + cognify() → _knowledge_base 预存结构化知识
#   - cognee.search()          → search_knowledge() 按关键词检索
#   - cognee.memify()          → _enriched_rules 规则化知识
# ============================================================
# 知识库 — 模拟 Cognee 知识图谱 (开发者画像 + Python 最佳实践 + 历史对话)
_knowledge_base: list[dict[str, str]] = [
    {
        "source": "developer_profile",
        "content": (
            "Developer is an AI/Backend engineer. Builds FastAPI services with "
            "Pydantic, heavy asyncio/aiohttp pipelines, and production testing "
            "via pytest-asyncio. Shipped low-latency APIs on AWS, Azure, GoogleCloud."
        ),
    },
    {
        "source": "principles_data",
        "content": (
            "Python Zen: Beautiful is better than ugly — prefer descriptive names. "
            "Explicit is better than implicit — be clear about types and imports. "
            "Simple is better than complex — choose straightforward solutions. "
            "Flat is better than nested — use early returns. "
            "Type hints reinforce explicitness. Context managers enforce safe "
            "resource handling. Dataclasses improve readability."
        ),
    },
    {
        "source": "conversations",
        "content": (
            "Past Q&A: async/await patterns — use asyncio with aiohttp, semaphore "
            "to cap concurrency, TCPConnector for pooling. "
            "dataclass vs pydantic — prefer Pydantic for API I/O (validation, "
            "JSON serialization, FastAPI integration). "
            "testing — use pytest-asyncio, async fixtures, isolated test DB. "
            "error handling — centralized with custom exceptions, structured "
            "logging, FastAPI middleware."
        ),
    },
]

# 规则化知识 — 模拟 cognee.memify() 提取的智能规则
_enriched_rules: list[str] = [
    "RULE: For async web scraping, combine asyncio + aiohttp + Semaphore",
    "RULE: Use Pydantic (not dataclass) when validation or JSON serialization is needed",
    "RULE: Test async code with pytest-asyncio and async fixtures",
    "RULE: Centralize error handling with custom exceptions + structured logging",
]


@tool(approval_mode="never_require")
def search_knowledge(
    query: Annotated[str, "Natural-language question to search the knowledge graph"],
) -> str:
    """Search the knowledge graph for relevant developer knowledge, best practices, and past conversations."""
    print(f"  [tool] search_knowledge({query!r}) called...")
    q = query.lower()
    # 模拟语义检索: 按关键词匹配知识条目
    matches = [
        entry["content"]
        for entry in _knowledge_base
        if any(word in entry["content"].lower() for word in q.split() if len(word) > 3)
    ]
    # 附加相关规则
    rule_matches = [r for r in _enriched_rules if any(w in r.lower() for w in q.split() if len(w) > 3)]
    if not matches and not rule_matches:
        return "No relevant knowledge found."
    parts = []
    if matches:
        parts.append("Knowledge:\n" + "\n".join(f"  - {m}" for m in matches))
    if rule_matches:
        parts.append("Rules:\n" + "\n".join(f"  - {r}" for r in rule_matches))
    return "\n".join(parts)


@tool(approval_mode="never_require")
def search_principles(
    query: Annotated[str, "Question about Python principles or best practices"],
) -> str:
    """Search only the Python principles subset of the knowledge graph."""
    print(f"  [tool] search_principles({query!r}) called...")
    principles_entry = next(
        (e for e in _knowledge_base if e["source"] == "principles_data"), None
    )
    if not principles_entry:
        return "No relevant principles found."
    return principles_entry["content"]


# ============================================================
# Demo 1: 工作记忆 (Working Memory) — 会话内上下文
#
# 对应知识点:
#   - Working Memory: agent.create_session() 提供会话内记忆
#   - 同一会话内 Agent 能回忆之前轮次的信息
#   - 新会话丢失上下文 (工作记忆不复存在)
# ============================================================
async def demo_working_memory():
    """工作记忆: 同一会话内保留上下文，新会话则丢失。"""
    print("=" * 60)
    print("Demo 1: Working Memory — Session-Based Context")
    print("=" * 60)

    agent = client.as_agent(
        name="TravelMemoryAgent",
        instructions=(
            "You are a travel agent who remembers user preferences across "
            "conversations. Track destinations mentioned, budget constraints, "
            "and travel dates."
        ),
        tools=[save_preference, get_preferences],
    )

    # --- 同一会话: 工作记忆保留上下文 ---
    session = agent.create_session()

    print("\n--- 同一会话 Turn 1 ---")
    r = await agent.run(
        "I love beach destinations and my budget is $3000",
        session=session,
    )
    print(f"[Agent]: {r.text}\n")

    print("--- 同一会话 Turn 2 (测试工作记忆) ---")
    r = await agent.run("What did I say my budget was?", session=session)
    print(f"[Agent]: {r.text}\n")

    # --- 新会话: 工作记忆丢失 ---
    print("--- 新会话 (工作记忆丢失) ---")
    new_session = agent.create_session()
    r = await agent.run("What is my budget?", session=new_session)
    print(f"[Agent]: {r.text}")
    print("  (新会话没有之前对话的记忆 — 工作记忆仅存在于会话生命周期内)\n")


# ============================================================
# Demo 2: 长期记忆 (Long-Term Memory) — @tool 偏好存储
#
# 对应知识点:
#   - Long-Term Memory: 通过 @tool 工具访问持久化存储
#   - 跨会话保留: 新会话中 Agent 用 get_preferences() 取回旧偏好
#   - 场景 1: 首次用户 (Sarah) 存储纪念日旅行偏好
#   - 场景 2: Sarah 数周后回来 (新会话)，Agent 取回偏好个性化推荐
# ============================================================
async def demo_long_term_memory():
    """长期记忆: @tool 偏好存储，跨会话保留用户信息。"""
    print("=" * 60)
    print("Demo 2: Long-Term Memory — Tool-Based Preference Store")
    print("=" * 60)

    # 清空偏好存储，确保 Demo 独立
    preference_store.clear()

    travel_agent = client.as_agent(
        name="TravelBookingAssistant",
        instructions=(
            "You are a personalized travel booking assistant with long-term memory.\n"
            "WORKFLOW:\n"
            "1. When a user starts a conversation, call get_preferences() to check "
            "for saved information.\n"
            "2. Store any new preferences the user mentions using save_preference().\n"
            "3. Use search_hotels() to find suitable options that match their "
            "preferences and budget.\n"
            "4. Do NOT recommend hotels that exceed the user's budget.\n\n"
            "IMPORTANT: Always use user_id='sarah_johnson_123' for all memory "
            "operations."
        ),
        tools=[save_preference, get_preferences, search_hotels],
    )

    # --- 场景 1: 首次用户，存储偏好 ---
    print("\n--- 场景 1: Sarah 首次到访 (存储偏好) ---")
    session_1 = travel_agent.create_session()
    r = await travel_agent.run(
        "Hi! I'm Sarah and I'm planning a trip for my 10th wedding anniversary. "
        "We love romantic destinations, fine dining, and spa experiences. "
        "My husband has mobility issues, so we need accessible accommodations. "
        "Our budget is around $700-800 per night.",
        session=session_1,
    )
    print(f"[Agent]: {r.text}\n")

    # --- 补充偏好 ---
    print("--- Sarah 补充偏好 (饮食限制) ---")
    r = await travel_agent.run(
        "The Hotel Sacher sounds perfect! We're both vegetarian and I have a "
        "severe nut allergy. Can you note that for future trips?",
        session=session_1,
    )
    print(f"[Agent]: {r.text}\n")

    # --- 验证存储内容 ---
    print("--- 偏好存储内容 (preference_store) ---")
    for uid, prefs in preference_store.items():
        print(f"  User: {uid}")
        for p in prefs:
            print(f"    - {p}")
    print()

    # --- 场景 2: Sarah 数周后回来 (新会话，长期记忆生效) ---
    print("--- 场景 2: Sarah 数周后回来 (新会话，长期记忆) ---")
    session_2 = travel_agent.create_session()
    r = await travel_agent.run(
        "Hi, my husband and I are planning another trip. Can you recommend "
        "a good hotel?",
        session=session_2,
    )
    print(f"[Agent]: {r.text}")
    print("  (新会话，但 Agent 通过 get_preferences() 取回了 Sarah 的旧偏好)\n")


# ============================================================
# Demo 3: 知识图谱式记忆 (Cognee 模拟) — 结构化知识 + 语义搜索
#
# 对应知识点 (13-agent-memory-cognee.ipynb):
#   - 知识图谱构建: cognee.add() + cognify() → _knowledge_base 预存结构化知识
#   - 图谱丰富: cognee.memify() → _enriched_rules 规则化知识
#   - MAF + Cognee 集成: @tool 包装 cognee.search() → search_knowledge / search_principles
#   - 工作记忆 + 长期记忆: AgentSession (会话内) + 知识图谱 (跨会话)
#   - 新会话中知识图谱仍可用
# ============================================================
async def demo_knowledge_graph_memory():
    """知识图谱式记忆: 结构化知识 + 语义检索，跨会话持久化。"""
    print("=" * 60)
    print("Demo 3: Knowledge Graph Memory — Structured Knowledge & Semantic Search")
    print("=" * 60)

    coding_agent = client.as_agent(
        name="CodingAssistant",
        instructions=(
            "You are an expert coding assistant with access to a knowledge graph "
            "containing developer profiles, Python best practices, and past "
            "conversations.\n\n"
            "WORKFLOW:\n"
            "1. Use search_knowledge() to find relevant information from the full "
            "knowledge graph.\n"
            "2. Use search_principles() when the question is specifically about "
            "Python best practices.\n"
            "3. Combine retrieved knowledge with your own expertise to give "
            "comprehensive answers.\n"
            "4. Reference the developer's known tech stack (FastAPI, asyncio, "
            "Pydantic) when relevant."
        ),
        tools=[search_knowledge, search_principles],
    )

    # --- 会话 1: 工作记忆 + 知识图谱 ---
    print("\n--- 会话 1 Turn 1: 查询异步爬虫与 Python 原则的契合度 ---")
    session = coding_agent.create_session()
    r = await coding_agent.run(
        "How does my AsyncWebScraper implementation align with Python's "
        "design principles?",
        session=session,
    )
    print(f"[Agent]: {r.text}\n")

    print("--- 会话 1 Turn 2: 工作记忆 + 知识图谱 (追问 dataclass vs Pydantic) ---")
    r = await coding_agent.run(
        "Based on what you just said, when should I pick dataclasses versus "
        "Pydantic for this work?",
        session=session,
    )
    print(f"[Agent]: {r.text}")
    print("  (Agent 结合了工作记忆 (上一轮回答) 与知识图谱)\n")

    # --- 会话 2: 新会话，工作记忆丢失但知识图谱仍可用 ---
    print("--- 会话 2 (新会话): 知识图谱长期记忆仍然可用 ---")
    session_2 = coding_agent.create_session()
    r = await coding_agent.run(
        "What logging guidance should I follow for incident reviews?",
        session=session_2,
    )
    print(f"[Agent]: {r.text}")
    print("  (新会话，但知识图谱中的历史对话仍可检索)\n")

    print("--- 会话 2 Turn 2: 查询 Python 最佳实践 ---")
    r = await coding_agent.run(
        "How should variables be named according to Python best practices?",
        session=session_2,
    )
    print(f"[Agent]: {r.text}\n")


async def main():
    # Demo 1: 工作记忆 — 会话内上下文 (同会话保留, 新会话丢失)
    await demo_working_memory()
    # Demo 2: 长期记忆 — @tool 偏好存储 (跨会话保留)
    await demo_long_term_memory()
    # Demo 3: 知识图谱式记忆 — 结构化知识 + 语义检索 (Cognee 模拟)
    await demo_knowledge_graph_memory()


if __name__ == "__main__":
    asyncio.run(main())
