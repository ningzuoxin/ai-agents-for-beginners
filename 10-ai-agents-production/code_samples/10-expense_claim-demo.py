"""
Lesson 10 - AI Agents in Production: Expense Claim Analysis (纯 Python 版)

核心主题: 演示一个"生产级"费用报销 Agent 流程 —— 用工具(插件)处理本地收据图片，
          提取差旅费用数据，生成报销邮件。多个 Agent 根据任务上下文动态选择调用函数。

流程 (对应原 10-expense_claim-demo.ipynb):
  1. OCR Agent  —— 调用 extract_receipt_expenses 工具(视觉模型直接识图)提取结构化费用数据
     (原 load_receipt_image 把 base64 当纯文本、识别不出，已保留但改为注释备查)
  2. Email Agent —— 调用 generate_expense_email 工具，把提取的数据整理成专业报销邮件
  3. WorkflowBuilder —— 用 add_edge 把两个 Agent 串成顺序流水线: OCR Agent → Email Agent

补充实验 (本脚本新增, 非原笔记内容):
  - demo_vision_ocr: 以多模态消息直传图片，验证 GITHUB_MODEL_ID 指定的视觉模型能否真正
    识别收据内容。对比上面「base64 当纯文本」的已知限制，正确做法是把图片作为 image
    内容 (Content.from_data) 放进用户消息，由 OpenAIChatCompletionClient 自动转成
    image_url 多模态消息 —— 全程走框架客户端，无需手写 HTTP 请求。

API 变更说明（agent-framework >= 1.8.0）:
  - AzureAIProjectAgentProvider / FoundryChatClient 已移除
    → 改用 OpenAIChatCompletionClient + GitHub Models
  - await provider.create_agent(...) 已废弃 → 改用 client.as_agent(...)

已知限制（对应原笔记 Note，涉及 gpt-4）:
  - 本流程把收据图片作为 base64 文本传给模型，多数聊天模型(包括 gpt-4o)不会把它
    当图像解析；且可能超出上下文窗口。
  - 生产环境建议: 用 Azure AI Vision 等 OCR 工具先提取文本，或以 image_url 多模态
    消息形式传图；若只想规避上下文报错，可用更小的图片或上下文更大的模型。
  - 注意: 原笔记写的是 gpt-4o；代码里的模型由 GITHUB_MODEL_ID 指定，并未硬编码。

运行前准备:
  - 在本目录放置 receipt.jpg（仓库已提供示例图）
  - 配置 .env: GITHUB_MODEL_ID / GITHUB_TOKEN / GITHUB_ENDPOINT
"""

import os
import sys
import asyncio
import base64
from typing import Annotated

from pydantic import BaseModel, Field

from dotenv import load_dotenv
from agent_framework import tool, AgentResponseUpdate, WorkflowBuilder, Message, Content
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
# 报销数据模型 (Expense Models)
#
# 对应知识点: Structured Output — 用 Pydantic 模型约束并校验费用数据
#   每条费用格式: {'date': '07-Mar-2025', 'description': '...', 'amount': 675.99, 'category': '...'}
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
        """
        Parses the raw query into a list of Expense objects.
        Expected format: "date|description|amount|category" separated by semicolons.
        """
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
                        print(f"[LOG] Parse Error: Invalid data in '{expense_str}': {e}")
        return expense_list


# ============================================================
# 工具: 生成报销邮件 (Generate Expense Email)
# ============================================================
@tool(approval_mode="never_require")
def generate_expense_email(
    expense_data: Annotated[
        str,
        "Semicolon-separated expense entries in 'date|description|amount|category' format",
    ]
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


# ============================================================
# 工具: 从收据图片提取费用 (Load Receipt Image)
#
# 对应知识点: 把收据图片 base64 编码后作为 data URI 交给 Agent 做 OCR
# 注: 原笔记把图片作为 base64 文本传给模型，多数模型(含 gpt-4o)不会当图像解析，
#     这是原笔记明确标注的已知限制 —— 此处忠实保留该行为。
# ============================================================
# 收据图片路径 (相对于本脚本所在目录，仓库已提供 receipt.jpg)
_RECEIPT_PATH = os.path.join(os.path.dirname(__file__), "receipt.jpg")


@tool(approval_mode="never_require")
def load_receipt_image(
    image_path: Annotated[str, "Path to the receipt image file"] = _RECEIPT_PATH,
) -> str:
    """Load a receipt image and return its base64-encoded data URI for OCR extraction."""
    print(f"  [tool] load_receipt_image({image_path!r}) called...")

    # 解析候选路径: 优先用传入路径，缺失时回退到脚本同目录的 receipt.jpg，
    # 以兼容「从其它目录运行 / Agent 传入相对文件名(如 'receipt.jpg')」的情况。
    _SCRIPT_DIR = os.path.dirname(__file__)
    candidates = [image_path, _RECEIPT_PATH]
    resolved = None
    for cand in candidates:
        if not cand:
            continue
        # 相对路径统一按脚本所在目录解析，避免依赖运行时的当前工作目录(cwd)
        cand_abs = cand if os.path.isabs(cand) else os.path.join(_SCRIPT_DIR, cand)
        if os.path.exists(cand_abs):
            resolved = cand_abs
            break

    if resolved is None:
        error_msg = (
            f"[LOG] Error: receipt image not found. Searched: {candidates}. "
            f"请确认 receipt.jpg 与本脚本在同一目录: {_SCRIPT_DIR}"
        )
        print(f"  {error_msg}")
        return error_msg

    try:
        with open(resolved, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{image_data}"
    except Exception as e:
        error_msg = f"[LOG] Error loading image '{resolved}': {e}"
        print(f"  {error_msg}")
        return error_msg


# ============================================================
# 工具: 用视觉模型直接识别收据 (Vision OCR) —— load_receipt_image 的正确替代
#
# 对应知识点: 视觉模型多模态输入 —— 用 Content.from_data 把图片作为 image 内容交给
#   client.get_response，由框架自动转成 image_url 消息，模型可真正"看图"识别；
#   相比上面 load_receipt_image(把 base64 当纯文本，识别不出) 这才是有效做法。
# 注: @tool 支持 async 函数，工具内部可 await client.get_response。
# ============================================================
@tool(approval_mode="never_require")
async def extract_receipt_expenses(
    image_path: Annotated[str, "Path to the receipt image file"] = _RECEIPT_PATH,
) -> str:
    """用视觉模型识别收据图片，返回 'date|description|amount|category' 分号分隔的结构化费用数据。"""
    print(f"  [tool] extract_receipt_expenses({image_path!r}) called...")

    # 路径解析与回退，逻辑同 load_receipt_image
    _SCRIPT_DIR = os.path.dirname(__file__)
    candidates = [image_path, _RECEIPT_PATH]
    resolved = None
    for cand in candidates:
        if not cand:
            continue
        cand_abs = cand if os.path.isabs(cand) else os.path.join(_SCRIPT_DIR, cand)
        if os.path.exists(cand_abs):
            resolved = cand_abs
            break

    if resolved is None:
        error_msg = (
            f"[LOG] Error: receipt image not found. Searched: {candidates}. "
            f"请确认 receipt.jpg 与本脚本在同一目录: {_SCRIPT_DIR}"
        )
        print(f"  {error_msg}")
        return error_msg

    try:
        with open(resolved, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        error_msg = f"[LOG] Error loading image '{resolved}': {e}"
        print(f"  {error_msg}")
        return error_msg

    # 把图片构造为 image 内容；框架客户端会自动转成 image_url 多模态消息
    image_content = Content.from_data(data=image_bytes, media_type="image/jpeg")
    user_message = Message(
        "user",
        [
            "请识别这张收据图片中的差旅费用，按 "
            "'date|description|amount|category' 格式(分号分隔)提取，"
            "日期转为 dd-MMM-yyyy。只返回结构化数据，不要多余文字。",
            image_content,
        ],
    )
    try:
        response = await client.get_response(messages=[user_message])
        return response.text
    except Exception as e:
        error_msg = (
            f"[LOG] Vision OCR 失败 (请确认 GITHUB_MODEL_ID 为支持视觉的模型, 如 gpt-4o): {e}"
        )
        print(f"  {error_msg}")
        return error_msg


# ============================================================
# 流式输出辅助函数
#
# 对应知识点: 按 Agent 分段实时打印工作流协作过程 (见 §5.4 约定)
# ============================================================
async def stream_workflow(workflow, user_request: str):
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
                print(f"# Agent - {author}:")
                print(f"{'=' * 50}")
                last_author = author
            print(update.text, end="", flush=True)
    print()


# ============================================================
# Demo: 顺序工作流 OCR Agent → Email Agent
#
# 对应知识点:
#   - WorkflowBuilder + add_edge 构建顺序流水线
#   - 流式输出 (stream=True): 按 Agent 分段实时观察协作过程
# ============================================================
async def demo_expense_claim():
    print("=" * 60)
    print("Demo: 报销工作流 (OCR Agent → Email Agent)")
    print("=" * 60)

    ocr_agent = client.as_agent(
        # 原 base64 文本方式(把图片当纯文本，识别不出)，保留备查:
        # tools=[load_receipt_image],
        tools=[extract_receipt_expenses],
        name="OCRAgent",
        instructions=(
            "You are an expert OCR assistant specialized in extracting structured data "
            "from receipt images. "
            "Use the 'extract_receipt_expenses' tool to recognize the receipt image; it "
            "returns travel-related expense details in the format: "
            "'date|description|amount|category' separated by semicolons. "
            "Follow these rules: "
            "- Date: Convert dates (e.g., '4/4/22') to 'dd-MMM-yyyy' (e.g., '04-Apr-2022'). "
            "- Description: Extract item names. "
            "- Amount: Use numeric values (e.g., '4.50' from '$4.50'). "
            "- Category: Infer from context (e.g., 'Meals' for food, 'Transportation' for "
            "travel, 'Accommodation' for lodging, 'Miscellaneous' otherwise). "
            "Ignore totals, subtotals, or service charges unless they are itemized expenses. "
            "If no expenses are found, return 'No expenses detected'. "
            "Return only the structured data, no additional text."
        ),
    )

    email_agent = client.as_agent(
        name="EmailAgent",
        tools=[generate_expense_email],
        instructions=(
            "You are an expense claim email generator. Take the travel expense data from "
            "the previous agent (in 'date|description|amount|category' format separated by "
            "semicolons) and use the 'generate_expense_email' tool to produce a professional "
            "expense claim email. "
            "Pass the semicolon-separated expense data directly to the tool."
        ),
    )

    # 构建顺序工作流: OCR Agent → Email Agent
    workflow = (
        WorkflowBuilder(start_executor=ocr_agent, output_from="all")
        .add_edge(ocr_agent, email_agent)
        .build()
    )

    prompt = (
        "Please extract the raw text from the receipt image at 'receipt.jpg', "
        "focusing on travel expenses like dates, descriptions, amounts, and categories "
        "(e.g., Transportation, Accommodation, Meals, Miscellaneous). "
        "Then generate a professional expense claim email."
    )

    print("启动报销工作流 (OCR → Email，流式输出):\n")
    await stream_workflow(workflow, prompt)


# ============================================================
# Demo: 多模态 Vision 实验 (以 image 内容直传图片)
#
# 对应知识点: 视觉模型支持多模态输入 —— 正确做法是用 Content.from_data 把图片作为
#   image 内容放进用户消息，由 OpenAIChatCompletionClient 在发送时自动转成 image_url
#   多模态消息(见 agent_framework_openai 的 _prepare_content_for_openai)，而非把 base64
#   当纯文本。全程走框架客户端 client.get_response，无需手写 HTTP 请求。
# ============================================================
async def demo_vision_ocr():
    print("=" * 60)
    print("Demo: 多模态 Vision 实验 (image 内容直传图片)")
    print("=" * 60)

    # 复用脚本同目录的 receipt.jpg (绝对路径，避免 cwd 问题)
    try:
        with open(_RECEIPT_PATH, "rb") as f:
            image_bytes = f.read()
    except Exception as e:
        print(f"  [LOG] 无法读取收据图片 '{_RECEIPT_PATH}': {e}")
        return

    # 把图片构造为 image 内容；框架客户端会自动转成 image_url 多模态消息
    image_content = Content.from_data(data=image_bytes, media_type="image/jpeg")

    user_message = Message(
        "user",
        [
            "请识别这张收据图片中的差旅费用，按 "
            "'date|description|amount|category' 格式(分号分隔)提取，"
            "日期转为 dd-MMM-yyyy。只返回结构化数据。",
            image_content,
        ],
    )

    print("向视觉模型发送图片 (image 内容)，请其识别收据内容...\n")
    try:
        response = await client.get_response(messages=[user_message])
        print("视觉模型返回:\n")
        print(response.text)
    except Exception as e:
        print(
            f"  [LOG] Vision 调用失败 (请确认 GITHUB_MODEL_ID 为支持视觉的模型,"
            f" 如 gpt-4o): {e}"
        )


async def main():
    await demo_expense_claim()
    # await demo_vision_ocr()


if __name__ == "__main__":
    asyncio.run(main())
