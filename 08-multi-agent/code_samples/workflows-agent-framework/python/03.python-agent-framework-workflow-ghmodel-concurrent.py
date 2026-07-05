"""
Lesson 08 - Concurrent Agent Workflows with GitHub Models (纯 Python 版)

对应笔记本: 03.python-agent-framework-workflow-ghmodel-concurrent.ipynb

核心主题: 演示 Microsoft Agent Framework 的 **并发工作流 (Concurrent Workflow)**，
以"旅行规划"场景演示 fan-out 模式 — 把同一个用户请求广播给多个 Agent 并行处理:
  1. 自定义 Executor (InputDispatcher) — 用 @handler + ctx.send_message 转发输入
  2. 并发工作流 (Fan-Out Workflow) — add_fan_out_edges 把输入广播给多个 Agent
  3. 多输出执行器 (Multiple Output Executors) — output_from 指定多个 Agent 都产生输出
  4. 工作流可视化 (Workflow Visualization) — 用 WorkflowViz 输出 Mermaid / DOT / SVG
  5. 结果聚合 (Result Aggregation) — get_outputs() 返回多个 AgentResponse，逐个打印

与 01.basic / 02.sequential 的差异:
  - 01.basic: 顺序流水线 (FrontDesk → Concierge)，两步，单输出
  - 02.sequential: 顺序流水线 (Sales → Price → Quote)，三步，单输出
  - 03.concurrent: 并发广播 (Dispatcher → [Researcher, Planner])，两个 Agent 并行，双输出
  - 本 demo 引入自定义 Executor (InputDispatcher)，展示 @handler / WorkflowContext 用法

API 变更说明（agent-framework >= 1.8.0）:
  - 原笔记本使用的 AzureAIProjectAgentProvider (agent_framework.azure) 已移除
    → 改用 OpenAIChatCompletionClient (agent_framework.openai) + GitHub Models
  - 原笔记本使用的 AzureCliCredential (azure.identity) 已移除
    → 改用从 .env 读取 GITHUB_TOKEN
  - await provider.create_agent(...) 已废弃 → 改用 client.as_agent(...) (同步调用)
  - 原笔记本使用的 output_executors=... 已废弃 (DeprecationWarning)
    → 改用 output_from=... (推荐写法)
  - Executor / handler / WorkflowContext / add_fan_out_edges 保持不变
  - 非流式运行: await workflow.run(...) 返回 WorkflowRunResult，用 get_outputs() 取结果
"""

import os
import sys
import asyncio

from dotenv import load_dotenv
from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    WorkflowViz,
    handler,
)
from agent_framework.openai import OpenAIChatCompletionClient

# Windows 控制台默认 GBK 编码，无法输出 ✓/✗ 等 Unicode 字符，强制 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# ============================================================
# 配置 GitHub Models 客户端 (替代已废弃的 AzureAIProjectAgentProvider)
#
# 原笔记本: provider = AzureAIProjectAgentProvider(credential=AzureCliCredential())
# 现行 API: OpenAIChatCompletionClient 使用 GitHub Models 端点与 token
# ============================================================
client = OpenAIChatCompletionClient(
    model=os.getenv("GITHUB_MODEL_ID"),
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url=os.getenv("GITHUB_ENDPOINT"),
)


# ============================================================
# Agent 指令定义 — 旅行研究员与旅行规划师
#
# 对应知识点: Specialization + 并发分工
#   - Researcher: 分析目的地、列出相关景点、为每个景点制定详细方案
#   - Planner: 基于研究员的发现创建详细旅行计划
#   两个 Agent 接收相同的用户输入，并行独立运行，互不依赖
# ============================================================
RESEARCHER_AGENT_NAME = "Researcher-Agent"
RESEARCHER_AGENT_INSTRUCTIONS = (
    "You are my travel researcher, working with me to analyze the destination, "
    "list relevant attractions, and make detailed plans for each attraction."
)

PLAN_AGENT_NAME = "Plan-Agent"
PLAN_AGENT_INSTRUCTIONS = (
    "You are my travel planner, working with me to create a detailed travel plan "
    "based on the researcher's findings."
)


# ============================================================
# 创建专家 Agent (Specialized Agents)
#
# 原笔记本: await provider.create_agent(name=..., instructions=...)
# 现行 API: client.as_agent(name=..., instructions=...) (同步，无需 await)
# ============================================================
research_agent = client.as_agent(
    name=RESEARCHER_AGENT_NAME,
    instructions=RESEARCHER_AGENT_INSTRUCTIONS,
)

plan_agent = client.as_agent(
    name=PLAN_AGENT_NAME,
    instructions=PLAN_AGENT_INSTRUCTIONS,
)


# ============================================================
# 自定义 Executor — 输入分发器 (InputDispatcher)
#
# 对应知识点: Executor / @handler / WorkflowContext
#   - Executor: 工作流执行器基类，自定义执行逻辑需继承它
#   - @handler: 装饰器，标记 Executor 中处理消息的协程方法
#   - WorkflowContext[str]: 执行上下文，泛型参数声明该 Executor 的输出消息类型
#   - ctx.send_message(text): 把消息发送给所有下游 Executor (由 add_fan_out_edges 连接)
#
# 作用: 这是一个 passthrough 执行器，把用户输入原样广播给所有下游 Agent。
#   不能直接把 Agent 作为 start_executor + fan-out，因为 fan-out 需要 source 先
#   接收输入再广播；InputDispatcher 承担这个"分发"角色。
# ============================================================
class InputDispatcher(Executor):
    """Forward the user input unchanged to all participating agents."""

    @handler
    async def forward(self, text: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(text)


# ============================================================
# 构建并发工作流 (Concurrent / Fan-Out Workflow)
#
# 对应知识点:
#   - WorkflowBuilder(start_executor=...): dispatcher 作为入口执行器
#   - add_fan_out_edges(source, targets): 把 source 的消息广播给所有 targets
#     → research_agent 与 plan_agent 并行接收同一份输入，同时调用 LLM
#   - output_from=agents: 指定两个 Agent 都是输出执行器 (原笔记本用已废弃的
#     output_executors=agents，现改为推荐的 output_from=)
#
# 拓扑:
#                   ┌→ research_agent (并行)
#   dispatcher ────┤
#                   └→ plan_agent    (并行)
# ============================================================
dispatcher = InputDispatcher(id="dispatcher")
agents = [research_agent, plan_agent]

workflow = (
    WorkflowBuilder(
        start_executor=dispatcher,
        output_from=agents,
    )
    .add_fan_out_edges(dispatcher, agents)
    .build()
)


# ============================================================
# 工作流可视化 (Workflow Visualization)
#
# 对应知识点: WorkflowViz
#   - to_mermaid(): 输出 Mermaid 流程图字符串 (可在 Markdown / Jupyter 渲染)
#   - to_digraph(): 输出 DOT (DiGraph) 格式字符串 (graphviz 源)
#   - export(format="svg"): 导出 SVG 文件，需可选的 graphviz 依赖及系统二进制
#     缺失时优雅降级 (ImportError)
# ============================================================
def visualize_workflow():
    print("Generating workflow visualization...")
    viz = WorkflowViz(workflow)
    # 输出 Mermaid 字符串
    print("Mermaid string: \n=======")
    print(viz.to_mermaid())
    print("=======")
    # 输出 DiGraph (DOT) 字符串
    print("DiGraph string: \n=======")
    print(viz.to_digraph())
    print("=======")
    # SVG 导出需要可选的 graphviz extra 及系统二进制；缺失时优雅降级
    try:
        svg_file = viz.export(format="svg")
        print(f"SVG file saved to: {svg_file}")
        return svg_file
    except ImportError as e:
        print(f"SVG export skipped (install graphviz to enable): {e}")
        return None


# ============================================================
# 运行工作流并取结果 (Run Workflow & Get Outputs)
#
# 对应知识点:
#   - 非流式运行: await workflow.run(message) 返回 WorkflowRunResult
#   - get_outputs(): 返回各输出执行器的 AgentResponse 列表
#     由于 output_from=[research_agent, plan_agent]，返回两个响应
#   - outputs 顺序与 output_from 列表顺序一致 (research_agent 在前)
#
# 并发执行说明:
#   dispatcher 收到输入后通过 ctx.send_message 广播，
#   research_agent 与 plan_agent 同时收到输入并并行调用 LLM，
#   两个 LLM 请求并发执行 (而非串行)，总耗时 ≈ max(两个请求) 而非两者之和。
# ============================================================
async def main():
    # 1. 工作流可视化
    visualize_workflow()

    # 2. 运行并发工作流 (非流式)
    user_request = "Plan a trip to Seattle in December"
    print(f"\n用户请求: {user_request}")
    print("\n运行并发工作流 (dispatcher → [researcher, planner] 并行)...\n")

    events = await workflow.run(user_request)
    outputs = events.get_outputs()

    # 3. 打印聚合结果
    #    outputs 是 AgentResponse 列表，每个输出执行器一个，
    #    顺序与 output_from 中给出的顺序一致 (research_agent, plan_agent)
    if outputs:
        print("===== Final Aggregated Responses =====")
        for i, response in enumerate(outputs, start=1):
            print(f"{'-' * 60}\n\n{i:02d}:\n{response.text}")


if __name__ == "__main__":
    asyncio.run(main())
