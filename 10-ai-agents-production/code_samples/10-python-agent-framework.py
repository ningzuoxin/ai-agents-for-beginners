"""
Lesson 10 - AI Agents in Production (纯 Python 版)

核心主题: 演示 AI Agent 从原型走向生产所需的三大支柱，以"旅行 Agent"与
"差旅报销"两个场景串联:

  支柱 1 — 可观测性 (Observability)
    Demo 1: 用计时 (timing) 监控 Agent 调用延迟，为接入 OpenTelemetry 打基础
  支柱 2 — 评估 (Evaluation)
    Demo 1 (续): 用独立的评估 Agent 对主 Agent 回复打分 (完整性/准确性/有用性)
  支柱 3 — 成本管理 (Cost Management)
    Demo 3: 模型选择 (gpt-4o-mini vs gpt-4o)、token 预算 (max_tokens)、缓存

  综合实战 — 差旅报销工作流 (Expense Claim Demo)
    Demo 2: Pydantic 结构化模型 + @tool 工具 + WorkflowBuilder 顺序工作流
            (OCR Agent → Email Agent) + 流式输出

对应两个原笔记本:
  - 10-python-agent-framework.ipynb → Demo 1 (可观测性 + 评估) + Demo 3 (成本管理)
  - 10-expense_claim-demo.ipynb     → Demo 2 (报销工作流)

API 变更说明（agent-framework >= 1.8.0）:
  - AzureAIProjectAgentProvider 已移除 → 改用 OpenAIChatCompletionClient + GitHub Models
  - await provider.create_agent(...) 已废弃 → 改用 client.as_agent(...)
  - WorkflowBuilder 需显式传 output_from="all"
  - 原笔记本的 load_receipt_image 返回 base64 data URI，依赖 Azure AI Foundry
    自动将工具结果中的图片传给视觉模型；本版用 GitHub Models 不支持该能力，
    故 OCR Agent 直接接收文本形式的费用数据来结构化 (知识点不变)。
"""

import os
import sys
import time
import base64
import asyncio
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from agent_framework import tool, AgentResponseUpdate, WorkflowBuilder, ChatOptions
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
# 旅行工具 (Travel Tools) — Demo 1 使用
# ============================================================
@tool(approval_mode="never_require")
def get_flight_info(destination: Annotated[str, "The destination city"]) -> str:
    """Get flight information for a destination."""
    print(f"  [tool] get_flight_info({destination!r}) called...")
    flights = {
        "Paris": "BA 304, 08:30-11:45, $350",
        "Tokyo": "JL 044, 11:00-07:00+1, $890",
        "Barcelona": "VY 7821, 07:15-10:30, $280",
    }
    return flights.get(destination, f"No flights found to {destination}")


@tool(approval_mode="never_require")
def get_activity_suggestions(destination: Annotated[str, "The destination city"]) -> str:
    """Get activity suggestions for a destination."""
    print(f"  [tool] get_activity_suggestions({destination!r}) called...")
    activities = {
        "Paris": "Louvre Museum, Eiffel Tower, Seine River Cruise, Montmartre walking tour",
        "Tokyo": "Senso-ji Temple, Tsukiji Market tour, Shibuya Crossing, teamLab Borderless",
        "Barcelona": "Sagrada Familia, Park Güell, La Boqueria Market, Gothic Quarter walk",
    }
    return activities.get(destination, f"No activities found for {destination}")


# ============================================================
# 报销数据模型 (Expense Models) — Demo 2 使用
#
# 对应知识点: Structured Output — Pydantic 模型约束 + 校验费用数据
# ============================================================
class Expense(BaseModel):
    date: str = Field(..., description="Date of expense in dd-MMM-yyyy format")
    description: str = Field(..., description="Expense description")
    amount: float = Field(..., description="Expense amount")
    category: str = Field(
        ...,
        description="Expense category (e.g., Transportation, Meals, Accommodation, Miscellaneous)",
    )


class ExpenseFormatter(BaseModel):
    raw_query: str = Field(..., description="Raw query input containing expense details")

    def parse_expenses(self) -> list[Expense]:
        """Parse 'date|description|amount|category' entries separated by semicolons."""
        expense_list = []
        for expense_str in self.raw_query.split(";"):
            if expense_str.strip():
                parts = expense_str.strip().split("|")
                if len(parts) == 4:
                    date, description, amount, category = parts
                    try:
                        expense = Expense(
                            date=date.strip(),
                            description=description.strip(),
                            amount=float(amount.strip()),
                            category=category.strip(),
                        )
                        expense_list.append(expense)
                    except ValueError as e:
                        print(f"  [LOG] Parse Error: Invalid data in '{expense_str}': {e}")
        return expense_list


# ============================================================
# 报销工具 (Expense Tools) — Demo 2 使用
# ============================================================
@tool(approval_mode="never_require")
def generate_expense_email(
    expense_data: Annotated[
        str,
        "Semicolon-separated expense entries in 'date|description|amount|category' format",
    ],
) -> str:
    """Generate an email to submit an expense claim to the Finance Team."""
    print(f"  [tool] generate_expense_email(...) called...")
    formatter = ExpenseFormatter(raw_query=expense_data)
    expenses = formatter.parse_expenses()
    if not expenses:
        return "No valid expenses found to include in the email."
    total_amount = sum(e.amount for e in expenses)
    email_body = "Dear Finance Team,\n\n"
    email_body += "Please find below the details of my expense claim:\n\n"
    for e in expenses:
        email_body += f"- {e.date} | {e.description}: ${e.amount:.2f} ({e.category})\n"
    email_body += f"\nTotal Amount: ${total_amount:.2f}\n\n"
    email_body += "Receipts for all expenses are attached for your reference.\n\n"
    email_body += "Thank you,\n[Your Name]"
    return email_body


# 收据图片路径 (相对于本脚本所在目录)
_RECEIPT_PATH = os.path.join(os.path.dirname(__file__), "receipt.jpg")


@tool(approval_mode="never_require")
def load_receipt_image(
    image_path: Annotated[str, "Path to the receipt image file"] = _RECEIPT_PATH,
) -> str:
    """Load a receipt image and return its base64-encoded data URI for OCR extraction."""
    print(f"  [tool] load_receipt_image({image_path!r}) called...")
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{image_data}"
    except Exception as e:
        error_msg = f"[LOG] Error loading image '{image_path}': {e}"
        print(f"  {error_msg}")
        return error_msg


# ============================================================
# Demo 1: 可观测性 + 评估 (Observability & Evaluation)
#
# 对应知识点:
#   - Observability: 用 time.time() 计时 Agent 调用，监控延迟
#     (生产环境可接入 OpenTelemetry / Langfuse 等追踪后端)
#   - Evaluation: 用独立评估 Agent 对主 Agent 回复打分
#     (完整性 / 准确性 / 有用性 / 总分，1-5 分制)
# ============================================================
async def demo_observability_and_evaluation() -> str:
    """可观测的旅行 Agent + 评估 Agent 打分。"""
    print("=" * 60)
    print("Demo 1: Observable Travel Agent + Evaluation")
    print("=" * 60)

    travel_agent = client.as_agent(
        name="TravelAgent",
        instructions=(
            "You are a helpful travel agent. Use the available tools to help "
            "users plan their trips. Provide comprehensive, actionable travel advice."
        ),
        tools=[get_flight_info, get_activity_suggestions],
    )

    # --- 可观测性: 计时 Agent 调用 ---
    user_query = (
        "I want to plan a day trip in Paris. "
        "What flights and activities do you recommend?"
    )
    print(f"\n用户请求: {user_query}")
    print("\n[Observability] 开始计时...\n")

    start_time = time.time()
    response = await travel_agent.run(user_query)
    elapsed = time.time() - start_time

    print(f"\n--- Agent 回复 (耗时 {elapsed:.2f}s) ---")
    print(response.text)

    # --- 评估: 独立评估 Agent 打分 ---
    print("\n[Evaluation] 调用评估 Agent...\n")
    evaluator = client.as_agent(
        name="ResponseEvaluator",
        instructions=(
            "You evaluate travel agent responses on these criteria:\n"
            "1. Completeness (1-5): Did it cover flights AND activities?\n"
            "2. Accuracy (1-5): Is the information consistent?\n"
            "3. Helpfulness (1-5): Would a traveler find this actionable?\n"
            "4. Overall Score (1-5)\n"
            "Provide scores and a brief explanation for each."
        ),
    )

    evaluation = await evaluator.run(
        f"Evaluate this travel agent response:\n\n{response.text}"
    )
    print("--- 评估结果 ---")
    print(evaluation.text)
    print()
    return response.text


# ============================================================
# Demo 2: 差旅报销工作流 (Expense Claim Workflow)
#
# 对应知识点:
#   - Pydantic 结构化模型: Expense / ExpenseFormatter 解析费用数据
#   - @tool 工具: generate_expense_email (生成报销邮件) / load_receipt_image (加载收据)
#   - WorkflowBuilder 顺序工作流: OCR Agent → Email Agent
#   - 流式输出: 实时观察两个 Agent 的协作过程
#
# 注: 原笔记本通过 load_receipt_image 加载图片后由视觉模型 OCR。
#     GitHub Models 免费版不支持工具结果中的图片传递，
#     故 OCR Agent 直接接收文本形式的费用数据来结构化。
# ============================================================
async def demo_expense_claim_workflow():
    """OCR Agent 结构化费用数据 → Email Agent 生成报销邮件。"""
    print("=" * 60)
    print("Demo 2: Expense Claim Workflow — OCR Agent -> Email Agent")
    print("=" * 60)

    ocr_agent = client.as_agent(
        name="OCRAgent",
        tools=[load_receipt_image],
        instructions=(
            "You are an expert OCR assistant specialized in extracting structured "
            "data from receipt text. Extract travel-related expense details and "
            "format them as 'date|description|amount|category' separated by "
            "semicolons. Follow these rules:\n"
            "- Date: Convert dates to 'dd-MMM-yyyy' (e.g., '04-Apr-2025').\n"
            "- Description: Extract item names.\n"
            "- Amount: Use numeric values (e.g., '4.50' from '$4.50').\n"
            "- Category: Infer from context (e.g., 'Meals' for food, "
            "'Transportation' for travel, 'Accommodation' for lodging, "
            "'Miscellaneous' otherwise).\n"
            "Ignore totals, subtotals, or service charges unless they are "
            "itemized expenses. Return only the structured data, no additional text."
        ),
    )

    email_agent = client.as_agent(
        name="EmailAgent",
        tools=[generate_expense_email],
        instructions=(
            "You are an expense claim email generator. Take the travel expense "
            "data from the previous agent (in 'date|description|amount|category' "
            "format separated by semicolons) and use the 'generate_expense_email' "
            "tool to produce a professional expense claim email. Pass the "
            "semicolon-separated expense data directly to the tool."
        ),
    )

    # 构建顺序工作流: OCR Agent → Email Agent
    workflow = (
        WorkflowBuilder(start_executor=ocr_agent, output_from="all")
        .add_edge(ocr_agent, email_agent)
        .build()
    )

    # 模拟从收据中 OCR 提取的原始文本 (生产环境由 load_receipt_image + 视觉模型完成)
    receipt_text = (
        "Please extract expenses from this receipt:\n"
        "Contoso Travel Receipt\n"
        "Date: 04-Apr-2025\n"
        "Flight to Paris - $675.99\n"
        "Taxi to airport - $42.50\n"
        "Hotel (2 nights) - $320.00\n"
        "Dinner - $58.75\n"
        "Airport parking - $25.00\n"
        "Then generate a professional expense claim email."
    )

    print(f"\n输入 (模拟收据文本):\n{receipt_text}")
    print("\n启动报销工作流 (OCR → Email，流式输出):\n")

    # 流式输出: 按 Agent 分段打印
    last_author = None
    events = workflow.run(receipt_text, stream=True)
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
# Demo 3: 成本管理 (Cost Management)
#
# 对应知识点 (README "Managing Costs"):
#   - Model Selection: 简单任务用小模型 (gpt-4o-mini)，复杂推理用大模型 (gpt-4o)
#   - Token Budgets: 用 max_tokens 限制生成长度，避免意外长回复
#   - Caching: 缓存工具结果，避免重复 API 调用
# ============================================================
# 简单工具结果缓存 — 避免相同查询的重复调用
_tool_cache: dict[str, str] = {}


@tool(approval_mode="never_require")
def get_flight_info_cached(destination: Annotated[str, "The destination city"]) -> str:
    """Get flight info with caching to avoid redundant calls."""
    cache_key = f"flight:{destination.lower()}"
    if cache_key in _tool_cache:
        print(f"  [tool] get_flight_info_cached({destination!r}) → CACHE HIT")
        return _tool_cache[cache_key]
    print(f"  [tool] get_flight_info_cached({destination!r}) → CACHE MISS, calling...")
    flights = {
        "Paris": "BA 304, 08:30-11:45, $350",
        "Tokyo": "JL 044, 11:00-07:00+1, $890",
        "Barcelona": "VY 7821, 07:15-10:30, $280",
    }
    result = flights.get(destination, f"No flights found to {destination}")
    _tool_cache[cache_key] = result
    return result


async def demo_cost_management():
    """演示模型选择、token 预算与缓存三种成本管理策略。"""
    print("=" * 60)
    print("Demo 3: Cost Management — Model Selection / Token Budgets / Caching")
    print("=" * 60)

    # --- 策略 1: 模型选择 ---
    # 简单任务 (如分类、提取) 用 gpt-4o-mini，成本约为 gpt-4o 的 1/30
    # 复杂推理用 gpt-4o
    print("\n--- 策略 1: Model Selection (模型选择) ---")
    print("  gpt-4o-mini  → 简单任务 (分类、提取、短回复)，成本低")
    print("  gpt-4o       → 复杂推理 (规划、多步工具调用)，质量高")
    print("  生产实践中用路由模型 (router) 按复杂度自动分流")

    # 用 gpt-4o-mini 做简单分类任务
    mini_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
        api_key=os.getenv("GITHUB_TOKEN"),
        base_url=os.getenv("GITHUB_ENDPOINT"),
    )
    classifier = mini_client.as_agent(
        name="QueryClassifier",
        instructions=(
            "Classify the user's travel query into one word: "
            "'simple' (single fact lookup) or 'complex' (multi-step planning). "
            "Reply with ONLY the label."
        ),
    )
    # 策略 2: Token Budgets — 限制生成长度
    print("\n--- 策略 2: Token Budgets (max_tokens 限制) ---")
    classification = await classifier.run(
        "What's the weather like in Paris?",
        options=ChatOptions(max_tokens=10),  # 分类任务只需几个 token
    )
    print(f"  分类结果 (max_tokens=10): {classification.text!r}")

    # --- 策略 3: 缓存 ---
    print("\n--- 策略 3: Caching (工具结果缓存) ---")
    print("  第一次调用 (CACHE MISS):")
    result1 = get_flight_info_cached("Paris")
    print(f"  结果: {result1}")
    print("  第二次调用相同查询 (CACHE HIT):")
    result2 = get_flight_info_cached("Paris")
    print(f"  结果: {result2}")
    print("  → 缓存命中，跳过了重复的数据查询")
    print()


async def main():
    # Demo 1: 可观测性 + 评估
    await demo_observability_and_evaluation()
    # Demo 2: 差旅报销工作流
    await demo_expense_claim_workflow()
    # Demo 3: 成本管理
    await demo_cost_management()


if __name__ == "__main__":
    asyncio.run(main())
