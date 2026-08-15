"""
Lesson 16 - Deploying Scalable Agents (纯 Python 版)

核心主题: 把"原型 Agent"改造成"生产就绪的客服 Agent"。本章的难点不在于
推理循环，而在于环绕在模型之外的那套生产骨架 (operational skeleton)。
本文件用纯 Python + Microsoft Agent Framework + GitHub Models，把原笔记
本 (16-python-agent-framework.ipynb) 的全部 8 个生产关注点复现出来:

  1. Tool calling        — 订单查询 / 工单创建 / 退款 (带人工审批)
  2. RAG                 — 政策知识库 (用内存关键词检索模拟 Azure AI Search)
  3. Memory              — 跨轮次的客户档案 (用 dict 模拟内存服务)
  4. Model routing       — 简单/复杂请求分流到小模型/大模型
  5. Response caching    — 重复问题直接从缓存返回，不调用模型
  6. Human approval      — 超过阈值的退款暂停，等待人工签核
  7. Evaluation gate     — 离线测试集给 Agent 打分，作为发布门禁
  8. Observability       — 每次请求包一层 OpenTelemetry span (无 OTel 时降级为空操作)

与原笔记本的差异:
  - 原笔记本用 FoundryChatClient + AzureCliCredential 连接 Microsoft Foundry。
    本版改用 OpenAIChatCompletionClient + GitHub Models，免 Azure 订阅即可本地运行。
  - 原笔记本的 issue_refund 用 approval_mode="always_require" 由框架暂停运行、
    把控制权交回代码，等待真人批准/拒绝后再继续。纯 Python 版不连 Foundry，
    但"暂停→真人决策→继续/拒绝"这个 human-in-the-loop 本质完全可以复现:
    本版用 request_human_approval() 在 issue_refund 内部直接做交互式审批
    (终端输入 y/n)，比平台 UI 更接近真实的 human-in-the-loop。默认 auto 模式
    自动批准，便于 CI / Demo 无交互跑通;可用 REFUND_APPROVAL_MODE=interactive
    体验真人逐项审批，或 =reject 演示被拒后模型转人工工单。
  - "Hosted Agent 部署 / 版本化 / 上线可观测" 属于 Microsoft Foundry 平台能力，
    无法本地复现；本版保留其教学核心: release() 模拟"先过评估门，再发布"的开关。

API 变更说明（agent-framework >= 1.8.0）:
  - FoundryChatClient 连接 Foundry → 本版用 OpenAIChatCompletionClient + GitHub Models
  - agent.run(prompt) 同步返回响应文本 (.text) 的用法不变
"""

import os
import re
import sys
import asyncio
from typing import Annotated
from contextlib import contextmanager

from dotenv import load_dotenv
from agent_framework import tool
from agent_framework.openai import OpenAIChatCompletionClient

# Windows 控制台默认 GBK 编码，无法输出 ✓/✗ 等 Unicode 字符，强制 UTF-8 输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# 本地运行用 GitHub Models (免 Azure 订阅)。
# 环境变量: GITHUB_TOKEN / GITHUB_ENDPOINT / GITHUB_MODEL_ID
client = OpenAIChatCompletionClient(
    model=os.getenv("GITHUB_MODEL_ID"),
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url=os.getenv("GITHUB_ENDPOINT"),
)


# ============================================================
# 1. Tools — 工具
#
# 生产工具对真实系统做真实工作。这里用纯 Python 函数模拟订单库与工单系统。
# issue_refund 超过阈值需要人工审批 (human-in-the-loop)。
# ============================================================
# 模拟后端系统 (生产环境这些是带作用域身份的 API 调用)
ORDERS = {
    "A1001": {"status": "shipped", "total": 42.00, "eta": "2 days"},
    "A1002": {"status": "processing", "total": 128.50, "eta": "5 days"},
    "A1003": {"status": "delivered", "total": 19.99, "eta": "delivered"},
}
TICKETS: list[dict] = []
REFUND_APPROVAL_THRESHOLD = 50.0


@tool(approval_mode="never_require")
def get_order_status(order_id: Annotated[str, "The customer's order ID, e.g. A1001"]) -> str:
    """Look up the status of a customer order."""
    print(f"  [tool] get_order_status({order_id!r}) called...")
    order = ORDERS.get(order_id.upper())
    if not order:
        return f"No order found with ID {order_id}."
    return (
        f"Order {order_id.upper()}: status={order['status']}, "
        f"total=${order['total']:.2f}, eta={order['eta']}."
    )


@tool(approval_mode="never_require")
def open_ticket(
    subject: Annotated[str, "Short subject line for the support ticket"],
    details: Annotated[str, "Full description of the customer's issue"],
) -> str:
    """Open a support ticket for issues that need human follow-up."""
    print(f"  [tool] open_ticket({subject!r}, ...) called...")
    ticket_id = f"T{1000 + len(TICKETS) + 1}"
    TICKETS.append({"id": ticket_id, "subject": subject, "details": details})
    return f"Ticket {ticket_id} opened: {subject}"


def refund_needs_approval(amount: float) -> bool:
    """Refunds above the threshold require a human approver."""
    return amount > REFUND_APPROVAL_THRESHOLD


# ------------------------------------------------------------
# Human-in-the-loop: 真人在终端批准/拒绝退款
#
# 原笔记本用 Foundry 的 approval_mode="always_require" 触发平台审批 UI —— 框架
# 暂停运行、把控制权交回你的代码、等你决定继续还是拒绝、再把结果喂回去。
# 纯 Python 版没有 Foundry,但"暂停→真人决策→继续/拒绝"这个本质完全可以复现:
# 这里直接做一个交互式审批,比平台 UI 更接近真实的 human-in-the-loop。
#
# 三种模式 (环境变量 REFUND_APPROVAL_MODE, 默认 auto):
#   interactive — 真人在终端输入 y / n 决定批准或拒绝
#   auto        — 自动批准 (便于 CI / Demo 无交互跑通)
#   reject       — 自动拒绝 (演示被拒时模型转人工工单的路径)
# ------------------------------------------------------------
APPROVAL_MODE = os.getenv("REFUND_APPROVAL_MODE", "auto").lower()
REFUNDS_APPROVED: list[str] = []


def request_human_approval(order_id: str, amount: float) -> bool:
    """暂停运行,请求真人批准退款。返回 True = 批准, False = 拒绝。"""
    if APPROVAL_MODE == "interactive":
        while True:
            ans = input(
                f"  ⏸ HUMAN APPROVAL REQUIRED: refund ${amount:.2f} for "
                f"order {order_id.upper()}? [y/n] "
            ).strip().lower()
            if ans in ("y", "yes"):
                return True
            if ans in ("n", "no"):
                return False
            print("    Please enter 'y' or 'n'.")
    if APPROVAL_MODE == "reject":
        return False
    # auto (默认): 非交互环境直接批准,演示"审批通过"分支
    return True


@tool(approval_mode="never_require")
def issue_refund(
    order_id: Annotated[str, "The order to refund"],
    amount: Annotated[float, "Refund amount in USD"],
) -> str:
    """Issue a refund. Refunds above the threshold pause for human approval."""
    oid = order_id.upper()
    print(f"  [tool] issue_refund({order_id!r}, ${amount:.2f}) called...")
    if not refund_needs_approval(amount):
        REFUNDS_APPROVED.append(oid)
        return f"Refund of ${amount:.2f} issued for order {oid}."
    # 超过阈值 → 暂停运行,等真人决策 (human-in-the-loop)
    approved = request_human_approval(oid, amount)
    if approved:
        REFUNDS_APPROVED.append(oid)
        return f"✅ Approved by human. Refund of ${amount:.2f} issued for order {oid}."
    return (
        f"❌ Refund of ${amount:.2f} for order {oid} was DENIED by the approver. "
        f"Please open a support ticket for manual follow-up."
    )


print("Tools defined.")


# ============================================================
# 2. RAG — Policy Knowledge Base
#
# 政策问题 ("退货期多久?") 应从权威来源回答，而非模型记忆。
# 生产环境是 Azure AI Search；这里用内存关键词检索，使脚本随处可运行。
# ============================================================
KNOWLEDGE_BASE = {
    "returns": "Contoso accepts returns within 30 days of delivery for a full refund. Items must be unused and in original packaging.",
    "shipping": "Standard shipping takes 3-5 business days. Express shipping (1-2 days) is available at checkout for an extra fee.",
    "warranty": "All Contoso electronics carry a 12-month limited warranty covering manufacturing defects.",
    "refund_policy": "Refunds are processed to the original payment method within 5 business days of approval. Refunds over $50 require a supervisor's approval.",
}


def _in_memory_search(query: str) -> str:
    """用关键词检索模拟 Azure AI Search 的语义检索。"""
    q = query.lower()
    hits = [text for key, text in KNOWLEDGE_BASE.items() if key.replace("_", " ") in q or key in q]
    if not hits:
        # 粗粒度关键词兜底，让工具仍有用
        hits = [text for text in KNOWLEDGE_BASE.values() if any(w in text.lower() for w in q.split())]
    return "\n".join(hits) if hits else "No matching policy found."


@tool(approval_mode="never_require")
def search_policies(query: Annotated[str, "The policy question to look up"]) -> str:
    """Search Contoso support policies to answer customer questions."""
    print(f"  [tool] search_policies({query!r}) called...")
    # 纯 Python 版固定走内存检索 (生产环境这里会切换到 Azure AI Search)
    return _in_memory_search(query)


print("RAG ready. Using in-memory search.")


# ============================================================
# 3. Memory — 记忆
#
# 生产环境是内存服务 (见 Lesson 13)；这里用 dict 让"跨轮次记住客户"可见。
# ============================================================
CUSTOMER_MEMORY: dict[str, dict] = {
    "cust-42": {"name": "Dana", "tier": "enterprise", "recent_order": "A1002"},
    "cust-99": {"name": "Sam", "tier": "standard", "recent_order": "A1003"},
}


def memory_context(customer_id: str) -> str:
    profile = CUSTOMER_MEMORY.get(customer_id)
    if not profile:
        return "This is a new customer with no history."
    return (
        f"Customer {profile['name']} ({profile['tier']} tier). "
        f"Most recent order: {profile['recent_order']}."
    )


# ============================================================
# 4 & 5. Model Routing + Response Caching
#
# 两个成本杠杆接在同一个请求处理器里:
#   - Routing:   便宜的启发式分类器决定走小模型还是大模型
#   - Caching:   规范化后的重复问题直接从缓存返回，不调用模型
# ============================================================
# GitHub Models 是单个模型，这里用一个 client 模拟 "小/大模型两档"。
# 生产环境可分别建 FoundryChatClient 指向不同部署。
SMALL_MODEL = os.getenv("GITHUB_MODEL_ID")
LARGE_MODEL = os.getenv("GITHUB_MODEL_ID")

response_cache: dict[str, str] = {}
route_counters = {"small": 0, "large": 0, "cache": 0}


def normalize(query: str) -> str:
    return re.sub(r"\s+", " ", query.lower().strip())


COMPLEX_SIGNALS = ("refund", "cancel", "complaint", "escalate", "broken", "wrong", "why")


def is_simple(query: str) -> bool:
    """复杂或高风险请求走大模型，其余走小模型。"""
    q = query.lower()
    if any(signal in q for signal in COMPLEX_SIGNALS):
        return False
    return len(q.split()) <= 20


def choose_model(query: str) -> str:
    return SMALL_MODEL if is_simple(query) else LARGE_MODEL


print(f"Small model: {SMALL_MODEL} | Large model: {LARGE_MODEL}")


# ============================================================
# 6 & 8. The Agent, Human Approval, and Observability
#
# 用上面的工具组装 Agent，并在每次请求外面包一层 OpenTelemetry span。
# handle_support_request 是生产请求处理器: cache → route → trace → run → cache。
# 人工审批: issue_refund 超过阈值会暂停运行、向真人要批准 (human-in-the-loop)，
# 被拒时模型会改用 open_ticket 转人工。
# ============================================================
# Tracing: 优先用 Agent Framework 的 OTel tracer，否则降级为空操作使脚本随处可运行。
try:
    from agent_framework.observability import get_tracer
    tracer = get_tracer()
except Exception:  # observability extras 未安装
    class _NoopSpan:
        def set_attribute(self, *_args, **_kwargs):
            pass

    class _NoopTracer:
        @contextmanager
        def start_as_current_span(self, _name):
            yield _NoopSpan()

    tracer = _NoopTracer()
    print("OpenTelemetry not available — using no-op tracer.")


SUPPORT_INSTRUCTIONS = (
    "You are Contoso's customer support agent. Be concise, friendly, and accurate. "
    "Use search_policies for policy questions, get_order_status for orders, "
    "open_ticket when a human needs to follow up, and issue_refund for refunds. "
    "Never invent policy details. If a refund requires human approval, "
    "tell the customer it is pending and open a ticket for follow-up."
)

_TOOLS = [get_order_status, open_ticket, search_policies, issue_refund]
support_agent = client.as_agent(
    name="ContosoSupportAgent",
    instructions=SUPPORT_INSTRUCTIONS,
    tools=_TOOLS,
)
print("Support agent assembled.")


async def handle_support_request(query: str, customer_id: str) -> str:
    # 1. 命中缓存则直接返回 (不调模型)
    key = normalize(query)
    if key in response_cache:
        route_counters["cache"] += 1
        print(f"  [cache HIT] 直接返回缓存结果 (未调用模型)")
        return response_cache[key]

    # 2. 按复杂度分流，控制成本
    chosen_model = choose_model(query)
    route_counters["small" if chosen_model == SMALL_MODEL else "large"] += 1
    model_label = "small" if chosen_model == SMALL_MODEL else "large"

    # 3. 注入客户记忆到 prompt
    context = memory_context(customer_id)
    prompt = f"[Customer context: {context}]\n\n{query}"

    # 4. 在 trace span 内运行，便于可观测
    with tracer.start_as_current_span("support_request") as span:
        span.set_attribute("customer.id", customer_id)
        span.set_attribute("routed.model", model_label)
        print(f"  [route → {model_label} model] 调用 Agent...")
        response = await support_agent.run(prompt)

    text = response.text
    response_cache[key] = text
    return text


# ============================================================
# Demo A: 请求处理器 (路由 + 缓存 + 记忆 + 人工审批)
# ============================================================
async def demo_support_request_handler():
    print("=" * 60)
    print("Demo A: Support Request Handler (route + cache + memory + approval)")
    print("=" * 60)

    # 第 1 个简单 (小模型)；第 2 个是订单查询；第 3 个重复第 1 个 → 命中缓存
    r1 = await handle_support_request("What is your return window?", "cust-99")
    print(f"\n[Agent]: {r1}")
    print("-" * 50)

    r2 = await handle_support_request("Where is my order A1002?", "cust-42")
    print(f"\n[Agent]: {r2}")
    print("-" * 50)

    r1b = await handle_support_request("What is your return window?", "cust-99")
    print(f"\n[Agent]: {r1b}")
    print("-" * 50)

    # 退款超过阈值 → 触发人工审批路径
    r3 = await handle_support_request("I want a refund of 100 dollars for order A1002", "cust-42")
    print(f"\n[Agent]: {r3}")
    print("-" * 50)

    print(f"Routing counters: {route_counters}")
    print()


# ============================================================
# 7. Evaluation Gate — 评估门禁
#
# 离线测试集给 Agent 打分；只有通过率超过阈值才允许"发布"。
# 评分器用简单的关键词重叠检查，保持脚本自包含 (生产环境用 LLM-as-judge)。
# ============================================================
TEST_CASES = [
    {"input": "How long do I have to return an item?", "expected": ["30 days", "refund"]},
    {"input": "How fast is standard shipping?", "expected": ["3-5", "business days"]},
    {"input": "What is the status of order A1001?", "expected": ["shipped", "A1001"]},
    {"input": "Do your electronics have a warranty?", "expected": ["12-month", "warranty"]},
]


def score_response(actual: str, expected_keywords: list[str]) -> float:
    actual_l = actual.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in actual_l)
    return hits / len(expected_keywords)


async def evaluation_gate(test_cases: list[dict], threshold: float = 0.8) -> bool:
    passed = 0
    for case in test_cases:
        # 评估门走默认 Agent (不做路由)
        result = await support_agent.run(case["input"])
        s = score_response(result.text, case["expected"])
        status = "PASS" if s >= 0.5 else "FAIL"
        print(f"[{status}] {case['input']}  (score={s:.0%})")
        if s >= 0.5:
            passed += 1
    pass_rate = passed / len(test_cases)
    print(f"\nEvaluation pass rate: {pass_rate:.0%} (gate: {threshold:.0%})")
    return pass_rate >= threshold


# ============================================================
# Putting It Together: A Simulated Release — 模拟发布
#
# 整个循环: 先跑评估门，通过才"发布"。这是 CI 里发布前该跑的模式。
# ============================================================
async def release(test_cases: list[dict]) -> None:
    print("Running pre-deployment evaluation gate...\n")
    if await evaluation_gate(test_cases, threshold=0.8):
        print("\n✅ Gate passed — promoting agent version to the Foundry Agent Service.")
    else:
        print("\n❌ Gate failed — release blocked. Fix the agent and re-run.")


async def demo_evaluation_gate():
    print("=" * 60)
    print("Demo B: Evaluation Gate (release gate)")
    print("=" * 60)
    await release(TEST_CASES)
    print()


async def main():
    # Demo A: 请求处理器 (路由 / 缓存 / 记忆 / 人工审批)
    await demo_support_request_handler()
    # Demo B: 评估门禁 (发布前质量门)
    await demo_evaluation_gate()


if __name__ == "__main__":
    asyncio.run(main())
