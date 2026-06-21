"""
Lesson 06 - Building Trustworthy AI Agents (纯 Python 版)

核心主题: 演示"系统消息框架 (System Message Framework)"——用 LLM 自动生成
结构化、可复用的系统提示词，以"旅行预订 Agent"场景串联:
  1. 元提示 (Meta-Prompting) — 用一个"元系统消息"指导 LLM 为具体角色生成系统提示
  2. 生成参数控制 — temperature / max_tokens / top_p 影响生成结果的创造性与长度
  3. 应用生成的系统消息 — 把 LLM 生成的提示词作为 Agent 指令，验证其实际效果
  4. 迭代改进 — 微调基本提示、重新生成，对比不同版本的系统消息质量

对应 README 的四步框架:
  Step 1: Create a Meta System Message  (元系统消息模板)
  Step 2: Create a basic prompt          (基本提示，描述角色与职责)
  Step 3: Provide Basic System Message to LLM  (交给 LLM 优化)
  Step 4: Iterate and Improve            (迭代改进)

API 变更说明（agent-framework >= 1.8.0）:
  - 原笔记本使用 azure.ai.inference.ChatCompletionsClient 的同步 complete() 调用；
    本版改用 agent_framework.openai.OpenAIChatCompletionClient 的异步 get_response()，
    与 04/05 章保持一致的客户端与异步风格。
  - SystemMessage / UserMessage → 统一使用 agent_framework.Message(role=..., contents=...)
  - 可选参数 temperature / max_tokens / top_p 通过 ChatOptions 传入
"""

import os
import sys
import asyncio

from dotenv import load_dotenv
from agent_framework import Message, ChatOptions
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
# Step 1: 元系统消息 (Meta System Message)
#
# 这是给 LLM 的"系统提示"，告诉它"你是一个擅长为 AI Agent 撰写系统提示的专家"。
# 它是一个可复用的模板——后续只需更换基本提示，就能批量生成不同角色的系统消息。
# ============================================================
META_SYSTEM_MESSAGE = """You are an expert at creating AI agent assistants.
You will be provided a company name, role, responsibilities and other
information that you will use to provide a system prompt for.
To create the system prompt, be descriptive as possible and provide a structure \
that a system using an LLM can better understand the role and responsibilities \
of the AI assistant."""


# ============================================================
# Step 2: 基本提示 (Basic Prompt)
#
# 描述 Agent 的角色、公司、职责等基本信息，作为 LLM 生成系统消息的"素材"。
# 这一步对应 README 中的 "Create a basic prompt"。
# ============================================================
def build_basic_prompt(role: str, company: str, responsibility: str) -> str:
    """根据角色、公司、职责拼接基本提示。"""
    return f"You are {role} at {company} that is responsible for {responsibility}."


# ============================================================
# 辅助函数: 调用 LLM 生成系统消息
#
# 对应 README Step 3 "Provide Basic System Message to LLM":
#   - SystemMessage = 元系统消息 (META_SYSTEM_MESSAGE)
#   - UserMessage   = 基本提示 (basic_prompt)
#   - LLM 返回优化后的、结构化的系统提示词
#
# 可选参数说明 (对应笔记本中的 Optional parameters):
#   - temperature: 控制随机性，1.0 = 较有创造性，0.0 = 确定性输出
#   - max_tokens:  限制生成长度，避免过长
#   - top_p:       核采样，1.0 = 不限制（与 temperature 配合使用）
# ============================================================
async def generate_system_message(basic_prompt: str) -> str:
    """用元提示 + 基本提示调用 LLM，生成结构化的系统消息。"""
    response = await client.get_response(
        messages=[
            Message(role="system", contents=META_SYSTEM_MESSAGE),
            Message(role="user", contents=basic_prompt),
        ],
        options=ChatOptions(
            temperature=1.0,
            max_tokens=1000,
            top_p=1.0,
        ),
    )
    return response.text


# ============================================================
# Demo 1: 元提示生成系统消息 (Step 1-3)
#
# 对应知识点:
#   - Meta-Prompting: 用"元系统消息"指导 LLM 生成具体角色的系统提示
#   - SystemMessage + UserMessage: 两条消息构成一次完整的对话请求
#   - 可选生成参数: temperature / max_tokens / top_p
# ============================================================
async def demo_generate_system_message():
    """用元提示框架让 LLM 自动生成结构化的系统消息。"""
    print("=" * 60)
    print("Demo 1: Meta-Prompting — Generate a System Message")
    print("=" * 60)

    role = "travel agent"
    company = "contoso travel"
    responsibility = "booking flights"

    basic_prompt = build_basic_prompt(role, company, responsibility)
    print(f"\n[Step 2] 基本提示:")
    print(f"  {basic_prompt}")

    print(f"\n[Step 3] 调用 LLM 生成系统消息")
    print(f"  (temperature=1.0, max_tokens=1000, top_p=1.0)")
    print()

    system_message = await generate_system_message(basic_prompt)

    print("--- LLM 生成的系统消息 ---")
    print(system_message)
    print()

    return system_message


# ============================================================
# Demo 2: 用生成的系统消息驱动 Agent
#
# 对应知识点:
#   - 系统消息的工程价值: 生成的提示词可直接作为 Agent 指令
#   - 把 Demo 1 的输出接入 client.as_agent()，验证实际效果
# ============================================================
async def demo_use_generated_system_message(system_message: str):
    """把 LLM 生成的系统消息作为 Agent 指令，让 Agent 回答用户问题。"""
    print("=" * 60)
    print("Demo 2: Use the Generated System Message with an Agent")
    print("=" * 60)

    agent = client.as_agent(
        name="ContosoTravelAgent",
        instructions=system_message,
    )

    response = await agent.run(
        "I need to book a flight from London to Tokyo next Monday. "
        "Can you help me find available options?"
    )
    print("\n--- Agent 回复 ---")
    print(response.text)
    print()


# ============================================================
# Demo 3: 迭代改进 (Step 4)
#
# 对应知识点:
#   - Iterate and Improve: 微调基本提示，重新生成，对比系统消息质量
#   - 更详细的基本提示 → 更精准、更结构化的系统消息
# ============================================================
async def demo_iterate_and_improve():
    """对比"简略"与"详细"两种基本提示，展示迭代改进的效果。"""
    print("=" * 60)
    print("Demo 3: Iterate and Improve — Compare Basic Prompts")
    print("=" * 60)

    # 版本 A: 简略的基本提示 (与 Demo 1 相同)
    prompt_a = build_basic_prompt(
        role="travel agent",
        company="contoso travel",
        responsibility="booking flights",
    )

    # 版本 B: 更详细的基本提示——补充了具体任务与约束
    prompt_b = (
        "You are a travel agent at Contoso Travel that is responsible for "
        "booking flights. Your tasks include: looking up available flights, "
        "booking flights, asking for seating and time preferences, canceling "
        "previously booked flights, and alerting customers on delays or "
        "cancellations. You should always be polite, professional, and "
        "prioritize customer satisfaction."
    )

    print("\n[版本 A] 简略基本提示:")
    print(f"  {prompt_a}")

    print("\n[版本 B] 详细基本提示 (补充了具体任务与约束):")
    print(f"  {prompt_b}")

    print("\n--- 版本 A 生成的系统消息 ---")
    message_a = await generate_system_message(prompt_a)
    print(message_a)

    print("\n--- 版本 B 生成的系统消息 ---")
    message_b = await generate_system_message(prompt_b)
    print(message_b)

    print("\n[对比说明]")
    print("  版本 B 的基本提示更详细，生成的系统消息通常会:")
    print("  - 更明确地列出每项任务的具体操作步骤")
    print("  - 包含语气与风格要求 (polite, professional)")
    print("  - 强调客户满意度等约束条件")
    print("  这正是'迭代改进'的价值: 微调输入 → 对比输出 → 选择最佳版本。")
    print()


async def main():
    # Demo 1: 生成系统消息，并保留结果供 Demo 2 使用
    system_message = await demo_generate_system_message()
    # Demo 2: 把生成的系统消息接入 Agent
    await demo_use_generated_system_message(system_message)
    # Demo 3: 迭代改进
    await demo_iterate_and_improve()


if __name__ == "__main__":
    asyncio.run(main())
