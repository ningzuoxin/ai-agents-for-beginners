"""
Lesson 08 - Basic Agent Workflows with GitHub Models (纯 Python 版)

对应笔记本: 01.python-agent-framework-workflow-ghmodel-basic.ipynb

核心主题: 演示 Microsoft Agent Framework 的 **Workflow Builder** 能力，以"酒店旅行推荐"
场景串联多智能体顺序工作流:
  1. 创建专家 Agent (Specialized Agents) — 前台接待员 (FrontDesk) 与礼宾评审员 (Concierge)
  2. 顺序工作流 (Sequential Workflow) — 用 WorkflowBuilder + add_edge 把 Agent 串成流水线
  3. 工作流可视化 (Workflow Visualization) — 用 WorkflowViz 输出 Mermaid / DOT / SVG
  4. 自定义工作流事件 (Custom WorkflowEvent) — 继承 WorkflowEvent 定义领域事件 (DatabaseEvent)
  5. 非流式取结果 (Get Outputs) — await workflow.run(...) 后用 get_outputs() 取结果

API 变更说明（agent-framework >= 1.8.0）:
  - 原笔记本使用的 AzureAIProjectAgentProvider (agent_framework.azure) 已移除
    → 改用 OpenAIChatCompletionClient (agent_framework.openai) + GitHub Models
  - 原笔记本使用的 AzureCliCredential (azure.identity) 已移除
    → 改用从 .env 读取 GITHUB_TOKEN
  - await provider.create_agent(...) 已废弃 → 改用 client.as_agent(...) (同步调用)
  - WorkflowBuilder / WorkflowEvent / WorkflowViz 保持不变
  - 非流式运行: await workflow.run(...) 返回 WorkflowRunResult，用 get_outputs() 取结果
"""

import os
import sys
import asyncio

from dotenv import load_dotenv
from agent_framework import WorkflowBuilder, WorkflowEvent, WorkflowViz
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
# Agent 指令定义 — 礼宾评审员与前台接待员
#
# 对应知识点: Specialization
#   - 前台 (FrontDesk): 简洁高效，每次只给单一推荐
#   - 礼宾 (Concierge): 关注本地真实体验，评审并改进前台的建议
# ============================================================
REVIEWER_NAME = "Concierge"
REVIEWER_INSTRUCTIONS = """
    You are an are hotel concierge who has opinions about providing the most local and authentic experiences for travelers.
    The goal is to determine if the front desk travel agent has recommended the best non-touristy experience for a traveler.
    If so, state that it is approved.
    If not, provide insight on how to refine the recommendation without using a specific example. 
    """

FRONTDESK_NAME = "FrontDesk"
FRONTDESK_INSTRUCTIONS = """
    You are a Front Desk Travel Agent with ten years of experience and are known for brevity as you deal with many customers.
    The goal is to provide the best activities and locations for a traveler to visit.
    Only provide a single recommendation per response.
    You're laser focused on the goal at hand.
    Don't waste time with chit chat.
    Consider suggestions when refining an idea.
    """


# ============================================================
# 创建专家 Agent (Specialized Agents)
#
# 原笔记本: await provider.create_agent(name=..., instructions=...)
# 现行 API: client.as_agent(name=..., instructions=...) (同步，无需 await)
# ============================================================
reviewer_agent = client.as_agent(
    name=REVIEWER_NAME,
    instructions=REVIEWER_INSTRUCTIONS,
)

front_desk_agent = client.as_agent(
    name=FRONTDESK_NAME,
    instructions=FRONTDESK_INSTRUCTIONS,
)


# ============================================================
# 构建顺序工作流 (Sequential Workflow)
#
# 对应知识点:
#   - WorkflowBuilder: 工作流编排引擎，start_executor 指定入口执行器
#   - add_edge(source, target): 前一个 Agent 的输出作为后一个 Agent 的输入
#   - front_desk_agent → reviewer_agent 形成两步流水线
# ============================================================
workflow = (
    WorkflowBuilder(start_executor=front_desk_agent)
    .add_edge(front_desk_agent, reviewer_agent)
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
# 自定义工作流事件 (Custom WorkflowEvent)
#
# 对应知识点: WorkflowEvent / 可观测性扩展
#   - WorkflowEvent 是工作流执行过程中的统一事件类型，按 type 区分
#     (output / intermediate / executor_invoked / executor_completed /
#      superstep_started / superstep_completed 等)
#   - 继承 WorkflowEvent 可定义领域专属事件 (如 DatabaseEvent)，
#     用于在自定义 Executor 中携带业务语义数据，增强可观测性与审计能力
# ============================================================
class DatabaseEvent(WorkflowEvent):
    """自定义工作流事件示例: 表示一次数据库访问。

    继承 WorkflowEvent 即可携带自定义数据，配合工作流事件流 (event stream)
    实现领域化的监控与审计。例如在自定义 Executor 中发出该事件，记录每一次
    数据库读写，便于追踪多 Agent 系统中的状态变更。
    """


# ============================================================
# 运行工作流并取结果 (Run Workflow & Get Outputs)
#
# 对应知识点:
#   - 非流式运行: await workflow.run(message) 返回 WorkflowRunResult
#   - get_outputs(): 返回各输出执行器 (末端 reviewer) 的 AgentResponse 列表
#   - outputs[0].text: 取最终结果的文本
# ============================================================
async def main():
    # 1. 工作流可视化
    svg_file = visualize_workflow()

    # 2. 运行顺序工作流 (非流式)
    #    旅客请求 → 前台起草推荐 → 礼宾评审 → 返回最终结果
    user_request = "I would like to go to Paris."
    print(f"\n用户请求: {user_request}")
    print("\n运行顺序工作流 (front_desk -> concierge)...\n")

    events = await workflow.run(user_request)
    outputs = events.get_outputs()
    result = outputs[0].text if outputs else ""

    print("--- 工作流最终结果 ---")
    print(result.replace("None", ""))


if __name__ == "__main__":
    asyncio.run(main())
