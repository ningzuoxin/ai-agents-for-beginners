"""
Lesson 08 - Conditional Agent Workflows with GitHub Models (纯 Python 版)

对应笔记本: 04.python-agent-framework-workflow-aifoundry-condition.ipynb

核心主题: 演示 Microsoft Agent Framework 的 **条件工作流 (Conditional Workflow)**，
以"技术教程撰写与发布"场景演示基于内容的动态路由:
  1. AgentExecutor 包装 — 显式用 AgentExecutor 包装 Agent，使函数执行器可
     通过 AgentExecutorResponse / AgentExecutorRequest 与 Agent 交互
  2. 函数执行器 (@executor) — 在 Agent 之间做数据转换:
     - to_reviewer_result: 解析 reviewer 的 JSON 响应为 ReviewResult
     - handle_review: 审核不通过时输出失败信息
     - save_draft: 审核通过时将草稿转发给 publisher
  3. 条件路由 (add_multi_selection_edge_group) — 根据 review_result 动态选择
     下游执行器 (handle_review 或 save_draft)
  4. 函数工具 (@tool) — 用 @tool 定义 save_markdown_file 替代 Azure 托管的
     HostedCodeInterpreterTool
  5. Pydantic 结构化输出 — 用 BaseModel 定义响应 schema，手动解析 JSON
  6. 工作流可视化 (WorkflowViz) — Mermaid / DOT / SVG
  7. 自定义工作流事件 (Custom WorkflowEvent) — 概念桩: DatabaseEvent
  8. 流式执行 (Streaming) — workflow.run(task, stream=True) 替代已移除的 run_stream

与 01-03 的差异:
  - 01.basic: 顺序流水线，单输出，非流式
  - 02.sequential: 三步顺序流水线，单输出，非流式
  - 03.concurrent: 并发广播，双输出，非流式
  - 04.condition: 条件路由 (动态分支)，流式执行，引入 @executor / @tool / AgentExecutor

API 变更说明（agent-framework >= 1.8.0）:
  - 原笔记本使用的 AzureAIAgentClient (agent_framework.azure) 已移除
    → 改用 OpenAIChatCompletionClient (agent_framework.openai) + GitHub Models
  - 原笔记本使用的 AzureCliCredential (azure.identity) 已移除
    → 改用从 .env 读取 GITHUB_TOKEN
  - chat_client.create_agent(instructions=..., tools=..., response_format=...)
    → client.as_agent(instructions=..., tools=..., default_options=...)
  - 原笔记本使用的 HostedWebSearchTool (Azure 托管) 已移除
    → 移除该工具，大纲内容直接在 prompt 中提供 (LLM 依靠训练知识撰写)
  - 原笔记本使用的 HostedCodeInterpreterTool (Azure 托管) 已移除
    → 改用 @tool 装饰器定义 save_markdown_file 函数工具
  - WorkflowBuilder().set_start_executor(...) 已移除
    → WorkflowBuilder(start_executor=...) (构造参数)
  - ChatMessage(Role.USER, text=...) 已移除
    → Message(role="user", contents=[...])
  - workflow.run_stream(task) 已移除
    → workflow.run(task, stream=True) 返回 ResponseStream，可 async for 迭代
  - AgentExecutorResponse.agent_run_response 已重命名
    → AgentExecutorResponse.agent_response
  - AgentExecutor / @executor / add_multi_selection_edge_group / WorkflowViz 保持不变
"""

import os
import sys
import asyncio
import json
import re
from dataclasses import dataclass
from typing_extensions import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

from agent_framework import (
    AgentExecutor,
    AgentExecutorRequest,
    AgentExecutorResponse,
    AgentResponse,
    AgentResponseUpdate,
    Message,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowEvent,
    WorkflowViz,
    executor,
    tool,
)
from agent_framework.openai import OpenAIChatCompletionClient

# Windows 控制台默认 GBK 编码，无法输出 Unicode 字符，强制 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# ============================================================
# 配置 GitHub Models 客户端 (替代已废弃的 AzureAIAgentClient)
#
# 原笔记本: async with AzureCliCredential() as credential, \
#               AzureAIAgentClient(async_credential=credential) as chat_client:
# 现行 API: OpenAIChatCompletionClient 使用 GitHub Models 端点与 token
# ============================================================
client = OpenAIChatCompletionClient(
    model=os.getenv("GITHUB_MODEL_ID"),
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url=os.getenv("GITHUB_ENDPOINT"),
)


# ============================================================
# Agent 指令定义 — 布道师 / 内容审核员 / 发布员
#
# 对应知识点: Specialization + 条件路由分工
#   - Evangelist: 根据大纲撰写技术教程草稿 (JSON 输出，含 draft_content)
#   - Reviewer: 审核草稿是否满足要求 (JSON 输出，含 review_result / reason)
#   - Publisher: 调用 save_markdown_file 工具保存草稿为 Markdown 文件
# ============================================================
EVANGELIST_INSTRUCTIONS = """
You are a technology evangelist create a first draft for a technical tutorials.
1. Each knowledge point in the outline must include a link. Follow the link to access the content related to the knowledge point in the outline. Expand on that content.
2. Each knowledge point must be explained in detail.
3. Rewrite the content according to the entry requirements, including the title, outline, and corresponding content. It is not necessary to follow the outline in full order.
4. The content must be more than 200 words.
4. Output draft as Markdown format. set 'draft_content' to the draft content.
5. return result as JSON with fields 'draft_content' (string).
"""

CONTENT_REVIEWER_INSTRUCTIONS = """
You are a content reviewer for a publishing company. You need to check whether the tutorial's draft content meets the following requirements:

1. The draft content less than 200 words, set 'review_result' to 'No' and 'reason' to 'Content is too short'. If the draft content is more than 200 words, set 'review_result' to 'Yes' and 'reason' to 'The content is good'.
2. set 'draft_content' to the original draft content.
3. return result as JSON with fields 'review_result' (one of Yes, No) and 'reason' (string) and 'draft_content' (string).

"""

PUBLISHER_INSTRUCTIONS = """
You are the content publisher. Use the save_markdown_file tool to save the tutorial's draft content as a Markdown file.
The file name is marked with current date and time, such as yearmonthdayhourminsec. Note that if it is 1-9, you need to add 0, such as 20240101123045.md.
After saving, return result as JSON with fields 'file_path' (string) indicating the saved file path.
"""


# ============================================================
# 大纲内容 (OUTLINE_CONTENT)
#
# 原笔记本中 Evangelist 使用 HostedWebSearchTool 搜索链接内容，
# 现移除该工具后 LLM 依靠训练知识撰写，大纲内容直接在 prompt 中提供。
# ============================================================
OUTLINE_CONTENT = """
# Introduce AI Agent


## What's AI Agent

https://github.com/microsoft/ai-agents-for-beginners/tree/main/01-intro-to-ai-agents


***Note*** Don't create any sample code


## Introduce Microsoft Foundry Agent Service

https://learn.microsoft.com/en-us/azure/ai-foundry/agents/overview


***Note*** Don't create any sample code


## Microsoft Agent Framework

https://github.com/microsoft/agent-framework/tree/main/docs/docs-templates


***Note*** Don't create any sample code
"""


# ============================================================
# Pydantic 模型 — 用于解析 Agent 的 JSON 响应
#
# 对应知识点: Structured Output (结构化输出)
#   - 原笔记本通过 response_format=... 让 Azure Agent 直接返回结构化对象
#   - 现行 API 改为手动 model_validate_json 解析 (与原笔记本 reviewer 一致)
# ============================================================
class EvangelistAgent(BaseModel):
    draft_content: str


class ReviewAgent(BaseModel):
    review_result: Literal["Yes", "No"]
    reason: str
    draft_content: str


class PublisherAgent(BaseModel):
    file_path: str


# ============================================================
# 中间数据结构 — ReviewResult
#
# 对应知识点: 函数执行器间的数据传递
#   - reviewer 的 JSON 响应经 to_reviewer_result 解析为 ReviewResult
#   - ReviewResult 作为 select_targets 的输入，决定路由方向
#   - ReviewResult 作为 handle_review / save_draft 的输入
# ============================================================
@dataclass
class ReviewResult:
    review_result: str
    reason: str
    draft_content: str


def extract_json(text: str) -> str:
    """从可能含 markdown 代码围栏的文本中提取 JSON 字符串。

    LLM 经常返回 ```json ... ``` 包裹的 JSON，pydantic 的 model_validate_json
    无法直接解析，需要先剥离围栏。本函数还兼顾裸 JSON 与围栏 JSON 两种情况。
    """
    # 优先匹配 ```json ... ``` 或 ``` ... ``` 围栏
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    # 没有围栏时，尝试找到第一个 { 到最后一个 } 的范围
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    # 兜底: 原样返回，让 pydantic 报原始错误
    return text.strip()


# ============================================================
# 函数工具 (@tool) — save_markdown_file
#
# 对应知识点: Function Tool / 替代 Azure 托管工具
#   - 原笔记本使用 HostedCodeInterpreterTool 让 publisher 运行代码保存文件
#   - 现改为 @tool 装饰器定义本地函数工具，publisher 通过函数调用保存文件
#   - @tool 自动从函数签名生成 JSON schema，LLM 可自动调用
# ============================================================
@tool
def save_markdown_file(file_name: str, content: str) -> str:
    """Save the given content as a Markdown file with the specified file name.

    Args:
        file_name: The file name (e.g., '20240101123045.md').
        content: The Markdown content to save.
    """
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(content)
    return file_name


# ============================================================
# 函数执行器 (@executor) — Agent 间的数据转换
#
# 对应知识点: Function Executor / AgentExecutorResponse / AgentExecutorRequest
#
# 什么是 @executor / FunctionExecutor?
#   - @executor 是装饰器，把一个普通 async 函数转成 FunctionExecutor 实例
#   - FunctionExecutor 是 Executor 的子类 — 与 AgentExecutor 平级，
#     两者都是工作流图中的"节点"，都能作为 add_edge / add_multi_selection_edge_group
#     的 source 或 target
#   - 区别: AgentExecutor 包装一个 Agent (内部调用 LLM)；
#           FunctionExecutor 包装一个普通函数 (纯逻辑，不调 LLM)
#   - 因此 to_reviewer_result / handle_review / save_draft 虽然不是 Agent，
#     但它们是 Executor，可以和 AgentExecutor 一样接入工作流图
#
# 为什么需要在 Agent 之间插入函数执行器?
#   - Agent 的输入/输出是 AgentExecutorResponse / AgentExecutorRequest (聊天消息)
#   - 但条件路由 (selection_func) 需要结构化数据 (ReviewResult) 才能做判断
#   - Agent 无法直接发出 ReviewResult，所以需要一个"中间人"做格式转换:
#       reviewer (Agent)  → 输出 AgentExecutorResponse (含 JSON 文本)
#       to_reviewer_result (函数) → 解析 JSON，发出 ReviewResult
#       selection_func → 根据 ReviewResult.review_result 路由
#   - 这就是 to_reviewer_result 的作用: 把 Agent 的非结构化文本输出
#     转成下游可用的结构化数据 (ReviewResult)
#
# AgentExecutorResponse / AgentExecutorRequest 说明:
#   - AgentExecutorResponse: AgentExecutor 的输出，含 agent_response + full_conversation
#   - AgentExecutorRequest: AgentExecutor 的输入，含 messages + should_respond
#
# 三个函数执行器:
#   1. to_reviewer_result: 接收 reviewer 的 AgentExecutorResponse，
#      解析 JSON 为 ReviewResult，发送给条件路由
#   2. handle_review: 接收 ReviewResult (审核不通过)，
#      yield_output 输出失败信息 (成为工作流输出)
#   3. save_draft: 接收 ReviewResult (审核通过)，
#      构造 AgentExecutorRequest 发送给 publisher
# ============================================================
@executor(id="to_reviewer_result")
async def to_reviewer_result(
    response: AgentExecutorResponse, ctx: WorkflowContext[ReviewResult]
) -> None:
    """解析 reviewer agent 的 JSON 响应为 ReviewResult 并转发。

    本执行器不是 Agent (不调用 LLM)，而是一个纯数据转换节点:
      输入: AgentExecutorResponse — reviewer agent 的原始响应 (JSON 文本)
      输出: ReviewResult — 解析后的结构化数据，发送给下游条件路由

    原笔记本: response.agent_run_response.text
    现行 API: response.agent_response.text (属性已重命名)
    """
    print(f"Raw response from reviewer agent: {response.agent_response.text}")

    # LLM 可能返回 ```json ... ``` 包裹的 JSON，需先剥离围栏再解析
    json_str = extract_json(response.agent_response.text)
    parsed = ReviewAgent.model_validate_json(json_str)
    await ctx.send_message(
        ReviewResult(
            review_result=parsed.review_result,
            reason=parsed.reason,
            draft_content=parsed.draft_content,
        )
    )


def select_targets(review: ReviewResult, target_ids: list[str]) -> list[str]:
    """条件路由选择函数 — 根据 review_result 决定下游执行器。

    对应知识点: add_multi_selection_edge_group + selection_func
      - target_ids 顺序与 add_multi_selection_edge_group 的 targets 列表一致
      - review_result == "Yes" → 路由到 save_draft (审核通过，进入发布流程)
      - review_result == "No"  → 路由到 handle_review (审核不通过，输出失败信息)
    """
    handle_review_id, save_draft_id = target_ids
    if review.review_result == "Yes":
        return [save_draft_id]
    else:
        return [handle_review_id]


@executor(id="handle_review")
async def handle_review(review: ReviewResult, ctx: WorkflowContext[str]) -> None:
    """处理审核不通过的情况 — 输出失败信息作为工作流输出。

    注意: select_targets 仅在 review_result == "No" 时路由到此处，
    因此 if 分支总是命中; else 分支为保留的兜底逻辑 (原笔记本即如此)。
    """
    if review.review_result == "No":
        await ctx.yield_output(f"Review failed: {review.reason}, please revise the draft.")
    else:
        await ctx.send_message(
            AgentExecutorRequest(
                messages=[Message(role="user", contents=[review.draft_content])],
                should_respond=True,
            )
        )


@executor(id="save_draft")
async def save_draft(
    review: ReviewResult, ctx: WorkflowContext[AgentExecutorRequest]
) -> None:
    """审核通过 — 构造 AgentExecutorRequest 转发草稿给 publisher agent。

    对应知识点: AgentExecutorRequest / Message 构造
      - 原笔记本: ChatMessage(Role.USER, text=...)
      - 现行 API: Message(role="user", contents=[...])
      - AgentExecutorRequest 包装 messages 列表，should_respond=True 触发 Agent 执行
    """
    await ctx.send_message(
        AgentExecutorRequest(
            messages=[Message(role="user", contents=[review.draft_content])],
            should_respond=True,
        )
    )


# ============================================================
# 创建 AgentExecutor — 显式包装 Agent
#
# 对应知识点: AgentExecutor / tools / @tool
#   - AgentExecutor(agent, id=...) 显式包装 Agent，使函数执行器可通过
#     AgentExecutorResponse 接收其输出
#   - evangelist: 无工具 (原用 HostedWebSearchTool，已移除)
#   - reviewer: 无工具
#   - publisher: 使用 save_markdown_file 函数工具 (替代 HostedCodeInterpreterTool)
#
# 原笔记本: AgentExecutor(chat_client.create_agent(instructions=..., tools=...), id=...)
# 现行 API: AgentExecutor(client.as_agent(instructions=..., tools=...), id=...)
# ============================================================
evangelist_agent = AgentExecutor(
    client.as_agent(
        instructions=EVANGELIST_INSTRUCTIONS,
    ),
    id="evangelist_agent",
)

reviewer_agent = AgentExecutor(
    client.as_agent(
        instructions=CONTENT_REVIEWER_INSTRUCTIONS,
    ),
    id="reviewer_agent",
)

publisher_agent = AgentExecutor(
    client.as_agent(
        instructions=PUBLISHER_INSTRUCTIONS,
        tools=[save_markdown_file],
    ),
    id="publisher_agent",
)


# ============================================================
# 构建条件工作流 (Conditional Workflow)
#
# 对应知识点:
#   - WorkflowBuilder(start_executor=...): evangelist_agent 作为入口
#     (原笔记本用 .set_start_executor()，已移除)
#   - add_edge: 顺序连接 (evangelist → reviewer → to_reviewer_result)
#   - add_multi_selection_edge_group: 条件路由
#     to_reviewer_result 根据 select_targets 动态选择 handle_review 或 save_draft
#   - add_edge: save_draft → publisher_agent (审核通过后发布)
#
# 拓扑:
#   evangelist → reviewer → to_reviewer_result ──┬→ handle_review (No: 输出失败)
#                                                  └→ save_draft → publisher (Yes: 发布)
# ============================================================
workflow = (
    WorkflowBuilder(start_executor=evangelist_agent)
    .add_edge(evangelist_agent, reviewer_agent)
    .add_edge(reviewer_agent, to_reviewer_result)
    .add_multi_selection_edge_group(
        to_reviewer_result,
        [handle_review, save_draft],
        selection_func=select_targets,
    )
    .add_edge(save_draft, publisher_agent)
    .build()
)


# ============================================================
# 工作流可视化 (Workflow Visualization)
#
# 对应知识点: WorkflowViz (同 01-03)
# ============================================================
def visualize_workflow():
    print("Generating workflow visualization...")
    viz = WorkflowViz(workflow)
    print("Mermaid string: \n=======")
    print(viz.to_mermaid())
    print("=======")
    print("DiGraph string: \n=======")
    print(viz.to_digraph())
    print("=======")
    try:
        svg_file = viz.export(format="svg")
        print(f"SVG file saved to: {svg_file}")
    except ImportError as e:
        print(f"SVG export skipped (install graphviz to enable): {e}")


# ============================================================
# 自定义工作流事件 (Custom WorkflowEvent) — 概念桩
#
# 对应知识点: WorkflowEvent / 可观测性扩展 (同 01-02)
#   - 本 demo 使用流式执行 (run stream=True)，所有事件 (含标准事件) 都会
#     通过 ResponseStream 迭代出来
#   - DatabaseEvent 仍为概念桩，不在此 demo 中实例化或发出
# ============================================================
class DatabaseEvent(WorkflowEvent):
    """自定义工作流事件概念桩: 表示一次数据库访问。

    仅演示继承语法，本 demo 不会实际实例化或发出。
    """

    ...


# ============================================================
# 运行工作流 (流式执行)
#
# 对应知识点:
#   - workflow.run(task, stream=True): 返回 ResponseStream[WorkflowEvent, WorkflowRunResult]
#     原笔记本用 workflow.run_stream(task)，已移除
#   - async for event in stream: 迭代所有工作流事件 (output / intermediate / status 等)
#   - event.data: 事件负载 (AgentResponse / str / None 等，取决于事件类型)
#
# 流式 vs 非流式 (对比 01-03):
#   - 01-03 用 await workflow.run(...) (非流式)，等全部完成后用 get_outputs() 取结果
#   - 04 用 workflow.run(..., stream=True) (流式)，实时迭代每个事件，适合长流程监控
# ============================================================
async def main():
    # 1. 工作流可视化
    visualize_workflow()

    # 2. 构造任务 (大纲内容嵌入 prompt)
    task = """
    You are a evangelist, need to write a draft based on the following outline and
    the content provided in the link corresponding to the outline. After draft create,
    the reviewer check it, if it meets the requirements, it will be submitted to the
    publisher and save it as a Markdown file, otherwise need to rewrite draft until it
    meets the requirements.
    The provided outline content and related links is as follows:
    """ + OUTLINE_CONTENT

    print("\n运行条件工作流 (evangelist → reviewer → [handle_review | save_draft → publisher])...")
    print("流式输出事件:\n")

    # 3. 流式运行工作流
    #    run(stream=True) 返回 ResponseStream，可 async for 迭代
    #
    # 流式模式下 AgentExecutor 会把每个 token 作为 AgentResponseUpdate 发出 output 事件，
    # 其中很多 update 的 text 为空 (role 标记、工具调用 delta 等)，需要过滤掉以减少噪音。
    # 有意义的 update 逐字打印 .text，其他事件类型 (executor_completed / status 等) 摘要显示。
    stream = workflow.run(task, stream=True)
    current_agent = ""  # 跟踪当前正在输出的 agent，用于分隔不同 agent 的流式输出
    async for event in stream:
        if isinstance(event, DatabaseEvent):
            print(f"\n[DatabaseEvent] {event}")
            continue

        if not isinstance(event, WorkflowEvent):
            continue

        data = event.data

        # 流式 token: AgentResponseUpdate — 只显示有文本内容的，跳过空 chunk
        if isinstance(data, AgentResponseUpdate):
            text = data.text
            if text and text.strip():
                # 检测是否切换到新 agent (通过 author_name)
                agent_name = data.author_name or ""
                if agent_name and agent_name != current_agent:
                    current_agent = agent_name
                    print(f"\n--- {current_agent} 流式输出 ---")
                print(text, end="", flush=True)
            continue

        # 完整响应: AgentResponse — 显示摘要
        if isinstance(data, AgentResponse):
            text = data.text
            if text and text.strip():
                print(f"\n[AgentResponse] {text[:300]}{'...' if len(text) > 300 else ''}")
            continue

        # AgentExecutorResponse: executor 完成时的完整响应
        if isinstance(data, AgentExecutorResponse):
            print(f"\n[executor_completed] {data.executor_id}")
            continue

        # 其他事件类型 (status / superstep_* / executor_invoked 等) — 摘要显示
        if data is not None:
            data_str = str(data)
            if data_str and data_str.strip():
                print(f"[{event.type}] {data_str[:200]}")

    print("\n\n=== 工作流执行完成 ===")


if __name__ == "__main__":
    asyncio.run(main())
