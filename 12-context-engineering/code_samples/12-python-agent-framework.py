"""
Lesson 12 - Context Engineering (纯 Python 版)

核心主题: 演示"上下文工程 (Context Engineering)"——在长对话中管理上下文窗口，
以"旅行规划多轮对话"场景串联:
  1. 为什么上下文管理重要 — 对话轮次增多，token 线性增长，最终超出上下文窗口
  2. 上下文感知 Agent (Context-Aware Agent) — 跨多轮保留关键信息 (偏好/预算/日期)
  3. 上下文摘要模式 (Context Summarization Pattern) — 用 @tool 工具把偏好压缩为摘要
  4. Agent 暂存区 (Agent Scratchpad) — 持久化外部记忆，在上下文缩减后仍能保留关键事实

对应 README 的策略:
  - Compressing Context: 摘要 + 裁剪
  - Agent Scratchpad: 外部文件/运行时对象存储关键信息
  - Context Distraction: 摘要缓解上下文过大导致的注意力分散
  - Context Clash: 摘要中覆盖旧偏好，避免冲突信息残留

API 变更说明（agent-framework >= 1.8.0）:
  - AzureAIProjectAgentProvider 已移除 → 改用 OpenAIChatCompletionClient + GitHub Models
  - await provider.create_agent(...) 已废弃 → 改用 client.as_agent(...)
  - agent.create_session() + agent.run(session=...) 支持多轮对话 (会话状态保留)
  - @tool 的参数需用 Annotated 提供描述 (原笔记本 summarize_preferences 缺少注解)
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
# 上下文摘要工具 (Context Summarization Tool)
#
# 对应知识点:
#   - Context Summarization Pattern: Agent 主动调用工具把偏好压缩为摘要
#   - Agent Scratchpad: 摘要存储在工具返回值中，即使旧消息被裁剪也能保留
#   - Context Clash 缓解: 新偏好覆盖旧偏好时，摘要只保留最新版本
# ============================================================
# 全局暂存区 — 模拟文件/数据库持久化 (Agent Scratchpad)
# 生产环境可替换为文件、数据库或向量存储
_scratchpad: list[str] = []


@tool(approval_mode="never_require")
def summarize_preferences(
    conversation_notes: Annotated[
        str,
        "Key user preferences and facts to persist (destination, budget, "
        "dates, interests, accommodation, etc.)",
    ],
) -> str:
    """Summarize accumulated user preferences into a compact format.

    This tool acts as an agent scratchpad — preferences recorded here
    survive even if older conversation messages are removed to save tokens.
    """
    print(f"  [tool] summarize_preferences() called, recording {len(conversation_notes)} chars...")
    _scratchpad.append(conversation_notes)
    return f"[SUMMARY] User preferences recorded: {conversation_notes}"


@tool(approval_mode="never_require")
def recall_preferences() -> str:
    """Recall all previously recorded user preferences from the scratchpad."""
    print("  [tool] recall_preferences() called...")
    if not _scratchpad:
        return "[SCRATCHPAD] No preferences recorded yet."
    combined = "\n".join(f"  - {entry}" for entry in _scratchpad)
    return f"[SCRATCHPAD] Recorded preferences:\n{combined}"


# ============================================================
# Demo 1: 上下文感知 Agent — 多轮对话中的上下文保留
#
# 对应知识点:
#   - Context-Aware Agent: 跨多轮保留关键信息
#   - Context Accumulation: 对话轮次增多，token 线性增长
#   - Context Clash: 用户改变偏好 (日期从 April 改为 October)，Agent 需处理冲突
#
# 六轮对话演示:
#   Turn 1: 初始偏好 (日本/寿司/寺庙/摄影)
#   Turn 2: 补充细节 (预算 $3000/独自旅行/10天/4月)
#   Turn 3: 测试上下文保留 (推荐一个不可错过的)
#   Turn 4: 扩展话题 (住宿偏好 - 传统旅馆)
#   Turn 5: 改变计划 (日期改为10月看秋叶) → Context Clash
#   Turn 6: 测试变更后的保留 (总结完整计划)
# ============================================================
async def demo_context_aware_agent():
    """上下文感知 Agent: 六轮对话演示上下文保留与变更处理。"""
    print("=" * 60)
    print("Demo 1: Context-Aware Agent — Multi-Turn Conversation")
    print("=" * 60)

    agent = client.as_agent(
        name="ContextAwareAgent",
        instructions=(
            "You are a helpful travel planning assistant with excellent "
            "memory management.\n"
            "When conversations get long:\n"
            "1. Summarize previous context into key points\n"
            "2. Track user preferences mentioned earlier\n"
            "3. Reference previous decisions without repeating full details\n"
            "Always maintain continuity while being concise."
        ),
    )

    # 创建会话 — 后续所有轮次共享同一会话状态
    session = agent.create_session()

    # --- Turn 1: 初始偏好 ---
    print("\n--- Turn 1: 初始偏好 ---")
    r = await agent.run(
        "I'm planning a trip to Japan. I love sushi, temples, and photography.",
        session=session,
    )
    print(f"[Agent]: {r.text}\n")

    # --- Turn 2: 补充细节 ---
    print("--- Turn 2: 补充细节 (预算/天数/时间) ---")
    r = await agent.run(
        "My budget is $3000 and I'll be traveling solo for 10 days in April.",
        session=session,
    )
    print(f"[Agent]: {r.text}\n")

    # --- Turn 3: 测试上下文保留 ---
    print("--- Turn 3: 测试上下文保留 ---")
    r = await agent.run(
        "Based on everything I've told you so far, "
        "what's the one thing you'd recommend I not miss?",
        session=session,
    )
    print(f"[Agent]: {r.text}\n")

    # --- Turn 4: 扩展话题 ---
    print("--- Turn 4: 扩展话题 (住宿偏好) ---")
    r = await agent.run(
        "What about accommodation? I prefer traditional Japanese inns.",
        session=session,
    )
    print(f"[Agent]: {r.text}\n")

    # --- Turn 5: 改变计划 (Context Clash 场景) ---
    print("--- Turn 5: 改变计划 (Context Clash: April → October) ---")
    r = await agent.run(
        "Actually, I've changed my mind about the dates. "
        "I'll go in October instead for the autumn colors.",
        session=session,
    )
    print(f"[Agent]: {r.text}\n")

    # --- Turn 6: 测试变更后的保留 ---
    print("--- Turn 6: 测试变更后的保留 (总结完整计划) ---")
    r = await agent.run(
        "Summarize my complete travel plan so far — destination, budget, "
        "duration, interests, accommodation, and timing.",
        session=session,
    )
    print(f"[Agent]: {r.text}\n")


# ============================================================
# Demo 2: 上下文摘要模式 — 用工具压缩偏好
#
# 对应知识点:
#   - Context Summarization Pattern: Agent 主动调用 summarize_preferences 工具
#   - Agent Scratchpad: 偏好存储在工具返回值 / 全局暂存区中
#   - Context Distraction 缓解: 摘要后旧消息可安全裁剪，减少注意力分散
#   - recall_preferences: 从暂存区读取之前记录的偏好
# ============================================================
async def demo_context_summarization():
    """摘要 Agent: 用工具记录偏好摘要，后续从暂存区读取。"""
    print("=" * 60)
    print("Demo 2: Context Summarization — Tool-Based Preference Recording")
    print("=" * 60)

    # 清空暂存区，确保 Demo 独立
    _scratchpad.clear()

    summarizing_agent = client.as_agent(
        name="SummarizingTravelAgent",
        instructions=(
            "You are a helpful travel planning assistant that actively "
            "manages conversation context.\n\n"
            "CONTEXT MANAGEMENT RULES:\n"
            "1. After gathering several user preferences, call "
            "summarize_preferences() to record a compact summary\n"
            "2. When the user asks you to recall details, use "
            "recall_preferences() to retrieve recorded summaries\n"
            "3. Keep responses concise — avoid restating the entire history\n\n"
            "PLANNING PROCESS:\n"
            "1. Gather user preferences (destination, budget, dates, interests)\n"
            "2. Summarize preferences using the tool\n"
            "3. Create recommendations based on the summary\n"
            "4. Update the summary when preferences change"
        ),
        tools=[summarize_preferences, recall_preferences],
    )

    session = summarizing_agent.create_session()

    # --- 第一轮: 提供偏好并要求记录 ---
    print("\n--- 第一轮: 提供偏好，要求用工具记录 ---")
    r = await summarizing_agent.run(
        "I want to visit Greece. I love seafood, history, and island hopping. "
        "Budget is $4000 for two weeks. Traveling with my partner in June. "
        "Please record these preferences using your summarization tool.",
        session=session,
    )
    print(f"[Agent]: {r.text}\n")

    # --- 第二轮: 从暂存区读取并推荐 ---
    print("--- 第二轮: 基于记录的偏好推荐 ---")
    r = await summarizing_agent.run(
        "Now, based on what you've recorded, suggest the top 3 islands "
        "we should visit.",
        session=session,
    )
    print(f"[Agent]: {r.text}\n")

    # --- 展示暂存区内容 ---
    print("--- Agent Scratchpad (暂存区内容) ---")
    for i, entry in enumerate(_scratchpad, 1):
        print(f"  记录 {i}: {entry[:120]}{'...' if len(entry) > 120 else ''}")
    print()


async def main():
    # Demo 1: 上下文感知 Agent — 六轮多轮对话
    await demo_context_aware_agent()
    # Demo 2: 上下文摘要模式 — 工具记录 + 暂存区读取
    await demo_context_summarization()


if __name__ == "__main__":
    asyncio.run(main())
