"""
Lesson 08 - Sequential Agent Workflows with GitHub Models (纯 Python 版)

对应笔记本: 02.python-agent-framework-workflow-ghmodel-sequential.ipynb

核心主题: 在 basic 顺序工作流基础上扩展为 **三阶段顺序流水线**，以"家具采购"
场景串联多智能体:
  1. 创建专家 Agent (Specialized Agents) — 销售顾问 (Sales) / 价格专家 (Price) / 报价生成 (Quote)
  2. 顺序工作流 (Sequential Workflow) — Sales → Price → Quote 三步流水线
  3. 工作流可视化 (Workflow Visualization) — 用 WorkflowViz 输出 Mermaid / DOT / SVG
  4. 自定义工作流事件 (Custom WorkflowEvent) — 概念桩: DatabaseEvent (仅演示继承语法)
  5. 非流式取结果 (Get Outputs) — await workflow.run(...) 后用 get_outputs() 取结果

与 01.basic 的差异:
  - 流水线从两步 (FrontDesk → Concierge) 扩展为三步 (Sales → Price → Quote)
  - 展示更贴近企业场景的"分析 → 估价 → 出单"模式
  - DatabaseEvent 在本 demo 中仍为概念桩 (见下方说明)

API 变更说明（agent-framework >= 1.8.0）:
  - 原笔记本使用的 AzureAIProjectAgentProvider (agent_framework.azure) 已移除
    → 改用 OpenAIChatCompletionClient (agent_framework.openai) + GitHub Models
  - 原笔记本使用的 AzureCliCredential (azure.identity) 已移除
    → 改用从 .env 读取 GITHUB_TOKEN
  - await provider.create_agent(...) 已废弃 → 改用 client.as_agent(...) (同步调用)
  - 原笔记本使用多模态 Message (带客厅图片)，现行 Message 不再提供
    TextContent/DataContent 辅助，改用等价的文本描述以聚焦顺序工作流机制
  - WorkflowBuilder / WorkflowEvent / WorkflowViz 保持不变
  - 非流式运行: await workflow.run(...) 返回 WorkflowRunResult，用 get_outputs() 取结果
"""

import os
import sys
import asyncio

from dotenv import load_dotenv
from agent_framework import Message, WorkflowBuilder, WorkflowEvent, WorkflowViz
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
# Agent 指令定义 — 销售顾问 / 价格专家 / 报价生成
#
# 对应知识点: Specialization + 顺序流水线分工
#   - Sales: 从用户描述中识别家具需求并给出采购建议
#   - Price: 对销售建议逐件估价，给出预算/中端/高端三档价格
#   - Quote: 汇总前两步，生成结构化的 Markdown 采购报价单
# ============================================================
SALES_AGENT_NAME = "Sales-Agent"
SALES_AGENT_INSTRUCTIONS = (
    "You are my furniture sales consultant, you can find different furniture "
    "elements from the pictures and give me a purchase suggestion"
)

PRICE_AGENT_NAME = "Price-Agent"
PRICE_AGENT_INSTRUCTIONS = """You are a furniture pricing specialist and budget consultant. Your responsibilities include:
        1. Analyze furniture items and provide realistic price ranges based on quality, brand, and market standards
        2. Break down pricing by individual furniture pieces
        3. Provide budget-friendly alternatives and premium options
        4. Consider different price tiers (budget, mid-range, premium)
        5. Include estimated total costs for room setups
        6. Suggest where to find the best deals and shopping recommendations
        7. Factor in additional costs like delivery, assembly, and accessories
        8. Provide seasonal pricing insights and best times to buy
        Always format your response with clear price breakdowns and explanations for the pricing rationale."""

QUOTE_AGENT_NAME = "Quote-Agent"
QUOTE_AGENT_INSTRUCTIONS = """You are a assistant that create a quote for furniture purchase.
        1. Create a well-structured quote document that includes:
        2. A title page with the document title, date, and client name
        3. An introduction summarizing the purpose of the document
        4. A summary section with total estimated costs and recommendations
        5. Use clear headings, bullet points, and tables for easy readability
        6. All quotes are presented in markdown form"""


# ============================================================
# 创建专家 Agent (Specialized Agents)
#
# 原笔记本: await provider.create_agent(name=..., instructions=...)
# 现行 API: client.as_agent(name=..., instructions=...) (同步，无需 await)
# ============================================================
sales_agent = client.as_agent(
    name=SALES_AGENT_NAME,
    instructions=SALES_AGENT_INSTRUCTIONS,
)

price_agent = client.as_agent(
    name=PRICE_AGENT_NAME,
    instructions=PRICE_AGENT_INSTRUCTIONS,
)

quote_agent = client.as_agent(
    name=QUOTE_AGENT_NAME,
    instructions=QUOTE_AGENT_INSTRUCTIONS,
)


# ============================================================
# 构建顺序工作流 (Sequential Workflow)
#
# 对应知识点:
#   - WorkflowBuilder: 工作流编排引擎，start_executor 指定入口执行器
#   - add_edge(source, target): 前一个 Agent 的输出作为后一个 Agent 的输入
#   - sales_agent → price_agent → quote_agent 形成三步流水线
#   - 与 01.basic 的两步流水线相比，多一个估价中间阶段
# ============================================================
workflow = (
    WorkflowBuilder(start_executor=sales_agent)
    .add_edge(sales_agent, price_agent)
    .add_edge(price_agent, quote_agent)
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
# 自定义工作流事件 (Custom WorkflowEvent) — 概念桩
#
# 对应知识点: WorkflowEvent / 可观测性扩展
#   - WorkflowEvent 是工作流执行过程中的统一事件类型，按 type 区分
#     (output / intermediate / executor_invoked / executor_completed /
#      superstep_started / superstep_completed 等)
#   - 继承 WorkflowEvent 可定义领域专属事件 (如 DatabaseEvent)
#
# 注意: 此处仅演示 "WorkflowEvent 可被继承" 这一语法概念。
#   本 demo 使用 client.as_agent() 创建标准 Agent，只会发出标准事件
#   (AgentResponse / AgentResponseUpdate)，并不会发出 DatabaseEvent。
#   要让 DatabaseEvent 真正进入事件流，需要编写自定义 WorkflowExecutor，
#   在其 execute() 中通过 ctx.yield_output() 主动发出 — 详见 advanced notebook。
# ============================================================
class DatabaseEvent(WorkflowEvent):
    """自定义工作流事件概念桩: 表示一次数据库访问。

    仅演示继承语法，本 demo 不会实际实例化或发出。
    """

    ...


# ============================================================
# 运行工作流并取结果 (Run Workflow & Get Outputs)
#
# 对应知识点:
#   - 非流式运行: await workflow.run(message) 返回 WorkflowRunResult
#   - get_outputs(): 返回各输出执行器 (末端 quote_agent) 的 AgentResponse 列表
#   - outputs[0].text: 取最终结果的文本
#
# 原笔记本使用多模态 Message (带客厅图片 home.png)，现行 Message 类不再
# 提供 TextContent/DataContent 辅助，故改用等价的文本描述，聚焦顺序工作流机制。
# ============================================================
async def main():
    # 1. 工作流可视化
    svg_file = visualize_workflow()

    # 2. 构造用户请求 (文本描述替代原笔记本的客厅图片)
    #    销售 → 估价 → 报价 三步流水线
    # Message 签名: Message(role, contents, *, ...)，无 text 关键字参数。
    # 文本通过 contents 传入，字符串会被自动转为 TextContent。
    message = Message(
        role="user",
        contents=[
            "I am furnishing a modern living room and want pieces that fit a warm, "
            "inviting style: a comfortable three-seat sofa, two accent armchairs, a "
            "wooden coffee table, a TV stand, a floor lamp, and a soft area rug. "
            "Please find appropriate furniture and give the corresponding price for "
            "each piece, then produce a final purchase quote."
        ],
    )
    print("\n用户请求:")
    print(message.text)
    print("\n运行顺序工作流 (sales -> price -> quote)...\n")

    # 3. 运行三步顺序工作流 (非流式)
    events = await workflow.run(message)
    outputs = events.get_outputs()
    result = outputs[0].text if outputs else ""

    print("--- 工作流最终结果 (Quote-Agent 输出) ---")
    print(result.replace("None", ""))


if __name__ == "__main__":
    asyncio.run(main())
