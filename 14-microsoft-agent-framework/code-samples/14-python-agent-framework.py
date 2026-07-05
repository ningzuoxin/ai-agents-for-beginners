"""
Lesson 14 - Exploring Microsoft Agent Framework (纯 Python 版)

核心主题: 演示 MAF 的核心编排模式与高级特性，以"酒店预订 / 旅行推荐"场景串联:

  Demo 1: 顺序编排 (Sequential Orchestration)
    — Front Desk Agent → Concierge Agent，第二个 Agent 改进第一个的输出
  Demo 2: 并发编排 (Concurrent Orchestration)
    — Fan-out/Fan-in 模式，三个专家 Agent 并行工作
  Demo 3: 条件工作流 (Conditional Workflow)
    — add_edge(condition=...) 根据结构化输出路由到不同 Agent
  Demo 4: 函数中间件 (Function Middleware)
    — FunctionInvocationContext 拦截工具调用，优先会员覆盖可用性结果
  Demo 5: 人在回路 (Human-in-the-Loop) 概念演示
    — 用 approval_mode="always_require" 实现敏感操作审批 (概念演示)

对应六个原笔记本:
  - 14-sequential.ipynb          → Demo 1
  - 14-concurrent.ipynb          → Demo 2
  - 14-conditional-workflow.ipynb → Demo 3
  - 14-middleware.ipynb          → Demo 4
  - 14-handoff.ipynb             → Demo 5 (概念: 动态路由 / 上下文传递)
  - 14-human-loop.ipynb          → Demo 5 (概念: 人工审批)

API 可用性说明:
  本版使用 agent_framework >= 1.8.0 + OpenAIChatCompletionClient (GitHub Models)。
  以下 API 在当前版本可用: AgentExecutor, Executor, @handler, @executor,
    WorkflowBuilder (add_edge / add_fan_out_edges / add_chain / condition=),
    FunctionInvocationContext, function_middleware, chat_middleware
  以下 API 在当前版本不可用 (原笔记本使用 AzureAIProjectAgentProvider):
    HandoffBuilder → 用条件工作流模拟动态路由
    RequestInfoExecutor / RequestInfoMessage → 用 approval_mode 模拟人在回路
  知识点不变，只是实现方式适配当前版本。
"""

import os
import sys
import json
import asyncio
import time
from typing import Annotated, Any, Never
from collections.abc import Awaitable, Callable

from dotenv import load_dotenv
from pydantic import BaseModel
from agent_framework import (
    tool,
    AgentResponseUpdate,
    WorkflowBuilder,
    WorkflowContext,
    AgentExecutor,
    AgentExecutorRequest,
    AgentExecutorResponse,
    Executor,
    Message,
    Role,
    FunctionInvocationContext,
    function_middleware,
    executor,
    handler,
)
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
# 流式输出辅助函数
# ============================================================
async def stream_workflow(workflow, user_request: str):
    """流式运行工作流，按 Agent 分段打印输出。"""
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
                print(f"[{author}]:")
                print(f"{'=' * 50}")
                last_author = author
            print(update.text, end="", flush=True)
    print()


# ============================================================
# Demo 1: 顺序编排 (Sequential Orchestration)
#
# 对应知识点:
#   - Sequential Workflow: 前台 Agent 推荐景点 → 礼宾 Agent 评审打分
#   - 迭代精炼: 第二个 Agent 改进第一个的输出
#   - WorkflowBuilder + add_edge: 线性顺序连接
# ============================================================
async def demo_sequential():
    """顺序工作流: 前台推荐 → 礼宾评审。"""
    print("=" * 60)
    print("Demo 1: Sequential Orchestration — Front Desk -> Concierge")
    print("=" * 60)

    front_desk_agent = client.as_agent(
        name="FrontDeskAgent",
        instructions=(
            "You are a knowledgeable hotel front desk agent who specializes in "
            "local attractions. When a guest asks about attractions in a city, "
            "provide a single, well-researched recommendation for a popular "
            "tourist attraction. Focus on practical information."
        ),
    )

    concierge_agent = client.as_agent(
        name="ConciergeAgent",
        instructions=(
            "You are an expert concierge with extensive knowledge of tourist "
            "attractions worldwide. You will receive an attraction recommendation "
            "and must provide an expert review and rating. Evaluate popularity, "
            "visitor satisfaction, and overall quality. Provide pros and cons."
        ),
    )

    workflow = (
        WorkflowBuilder(start_executor=front_desk_agent, output_from="all")
        .add_edge(front_desk_agent, concierge_agent)
        .build()
    )

    user_request = "I want to visit an attraction in Stockholm"
    print(f"\n用户请求: {user_request}")
    print("\n启动顺序工作流 (流式输出):\n")

    await stream_workflow(workflow, user_request)
    print()


# ============================================================
# Demo 2: 并发编排 (Concurrent Orchestration)
#
# 对应知识点:
#   - Fan-out/Fan-in: Dispatcher 广播到多个 Agent，并行执行
#   - add_fan_out_edges: 一对多边
#   - 自定义 Executor + @handler: InputDispatcher 转发输入
#   - 并行 vs 顺序性能对比
# ============================================================
class InputDispatcher(Executor):
    """转发用户输入到所有下游 Agent (fan-out)。"""

    @handler
    async def forward(self, text: str, ctx: WorkflowContext[str]) -> None:
        await ctx.send_message(text)


async def demo_concurrent():
    """并发工作流: 三个专家 Agent 并行提供旅行推荐。"""
    print("=" * 60)
    print("Demo 2: Concurrent Orchestration — Fan-out/Fan-in")
    print("=" * 60)

    attractions_agent = client.as_agent(
        name="AttractionsAgent",
        instructions=(
            "You are a tourism expert specializing in attractions and activities. "
            "Provide recommendations for tourist attractions, activities, and "
            "transportation tips."
        ),
    )

    dining_agent = client.as_agent(
        name="DiningAgent",
        instructions=(
            "You are a culinary expert specializing in local food and dining "
            "experiences. Recommend local cuisine, must-try dishes, and restaurants."
        ),
    )

    history_agent = client.as_agent(
        name="HistoryAgent",
        instructions=(
            "You are a historian and cultural expert. Provide historical context, "
            "cultural significance, and interesting facts about the destination."
        ),
    )

    dispatcher = InputDispatcher(id="dispatcher")
    agents = [attractions_agent, dining_agent, history_agent]

    # Fan-out: dispatcher → 三个 Agent 并行
    workflow = (
        WorkflowBuilder(start_executor=dispatcher, output_from="all")
        .add_fan_out_edges(dispatcher, agents)
        .build()
    )

    destination = "Tokyo"
    print(f"\n用户请求: Comprehensive travel recommendations for {destination}")
    print("\n启动并发工作流 (三个 Agent 并行，流式输出):\n")

    start_time = time.time()
    await stream_workflow(workflow, f"I want comprehensive travel recommendations for {destination}")
    concurrent_time = time.time() - start_time

    print(f"\n[性能] 并发执行耗时: {concurrent_time:.2f}s")
    print("  (并发模式下三个 Agent 同时工作，比顺序执行更快)\n")


# ============================================================
# Demo 3: 条件工作流 (Conditional Workflow)
#
# 对应知识点:
#   - Conditional Edges: add_edge(condition=...) 根据条件路由
#   - 条件函数: 解析结构化输出，返回 True/False 控制路由
#   - @tool 工具: hotel_booking 检查可用性
#   - AgentExecutor: 包装 Agent 为工作流执行器
#   - @executor 自定义执行器: display_result
#
# 工作流结构:
#   availability_agent → [有房] → booking_agent → display_result
#                     → [无房] → alternative_agent → display_result
# ============================================================
# --- Pydantic 结构化输出模型 ---
class BookingCheckResult(BaseModel):
    destination: str
    has_availability: bool
    message: str


class AlternativeResult(BaseModel):
    alternative_destination: str
    reason: str


class BookingConfirmation(BaseModel):
    destination: str
    action: str
    message: str


# --- 酒店预订工具 ---
@tool(approval_mode="never_require")
def hotel_booking(
    destination: Annotated[str, "The destination city to check for hotel rooms"],
) -> str:
    """Check hotel room availability for a destination city."""
    print(f"  [tool] hotel_booking({destination!r}) called...")
    cities_with_rooms = ["stockholm", "seattle", "tokyo", "london", "amsterdam"]
    has_rooms = destination.lower() in cities_with_rooms
    result = {"has_availability": has_rooms, "destination": destination}
    return json.dumps(result)


# --- 条件函数: 解析 AgentExecutorResponse 决定路由 ---
def has_availability_condition(message: Any) -> bool:
    """有房 → 路由到 booking_agent。"""
    if not isinstance(message, AgentExecutorResponse):
        return True
    try:
        result = BookingCheckResult.model_validate_json(
            message.agent_response.text
        )
        print(f"  [condition] has_availability = {result.has_availability} for {result.destination}")
        return result.has_availability
    except Exception:
        return False


def no_availability_condition(message: Any) -> bool:
    """无房 → 路由到 alternative_agent。"""
    if not isinstance(message, AgentExecutorResponse):
        return False
    try:
        result = BookingCheckResult.model_validate_json(
            message.agent_response.text
        )
        print(f"  [condition] no_availability for {result.destination}")
        return not result.has_availability
    except Exception:
        return False


# --- 自定义 display 执行器 ---
@executor(id="display_result")
async def display_result(
    response: AgentExecutorResponse,
    ctx: WorkflowContext[Never, str],
) -> None:
    """输出最终结果。"""
    print("  [executor] display_result: yielding output")
    await ctx.yield_output(response.agent_response.text)


async def demo_conditional_workflow():
    """条件工作流: 根据酒店可用性路由到不同 Agent。"""
    print("=" * 60)
    print("Demo 3: Conditional Workflow — Availability-Based Routing")
    print("=" * 60)

    # --- 创建三个 Agent (用 AgentExecutor 包装) ---
    availability_agent = AgentExecutor(
        client.as_agent(
            name="AvailabilityAgent",
            instructions=(
                "You are a hotel booking assistant that checks room availability. "
                "Use the hotel_booking tool to check if rooms are available. "
                "Return JSON: destination, has_availability (bool), message."
            ),
            tools=[hotel_booking],
        ),
        id="availability_agent",
    )

    alternative_agent = AgentExecutor(
        client.as_agent(
            name="AlternativeAgent",
            instructions=(
                "You are a helpful travel assistant. When a user cannot find hotels "
                "in their requested city, suggest an alternative nearby city. "
                "Return JSON: alternative_destination, reason."
            ),
        ),
        id="alternative_agent",
    )

    booking_agent = AgentExecutor(
        client.as_agent(
            name="BookingAgent",
            instructions=(
                "You are a booking assistant. The user has found available hotel "
                "rooms. Encourage them to book. Return JSON: destination, action, "
                "message."
            ),
        ),
        id="booking_agent",
    )

    # --- 构建条件工作流 ---
    workflow = (
        WorkflowBuilder(
            start_executor=availability_agent,
            output_from=[display_result],
        )
        # 无房路径
        .add_edge(availability_agent, alternative_agent, condition=no_availability_condition)
        .add_edge(alternative_agent, display_result)
        # 有房路径
        .add_edge(availability_agent, booking_agent, condition=has_availability_condition)
        .add_edge(booking_agent, display_result)
        .build()
    )

    # --- 测试 1: Paris (无房) → alternative_agent ---
    print("\n--- 测试 1: Paris (预期无房 → 推荐替代城市) ---")
    request_paris = AgentExecutorRequest(
        messages=[Message(role="user", contents=["I want to book a hotel in Paris"])],
        should_respond=True,
    )
    events_paris = await workflow.run(request_paris)
    outputs_paris = events.get_outputs()
    if outputs_paris:
        print(f"\n[结果]: {outputs_paris[0][:200]}")
    print()

    # --- 测试 2: Stockholm (有房) → booking_agent ---
    print("--- 测试 2: Stockholm (预期有房 → 鼓励预订) ---")
    request_stockholm = AgentExecutorRequest(
        messages=[Message(role="user", contents=["I want to book a hotel in Stockholm"])],
        should_respond=True,
    )
    events_stockholm = await workflow.run(request_stockholm)
    outputs_stockholm = events_stockholm.get_outputs()
    if outputs_stockholm:
        print(f"\n[结果]: {outputs_stockholm[0][:200]}")
    print()


# ============================================================
# Demo 4: 函数中间件 (Function Middleware)
#
# 对应知识点:
#   - Function Middleware: 拦截工具调用，修改结果
#   - FunctionInvocationContext: 访问函数名、参数、结果
#   - next(context): 执行原始函数
#   - context.result: 读取/修改函数输出
#   - 业务逻辑: 优先会员覆盖可用性 (无房 → 有房)
# ============================================================
# 优先会员数据库 (模拟)
PRIORITY_MEMBERS = {"alice@example.com", "bob@example.com", "priority_user"}
_current_user = "regular_user"


def set_user(user_id: str):
    """设置当前用户 (模拟会话管理)。"""
    global _current_user
    _current_user = user_id
    status = "PRIORITY" if user_id in PRIORITY_MEMBERS else "Regular"
    print(f"  [user] 当前用户: {user_id} ({status})")


# --- 优先会员中间件 ---
@function_middleware
async def priority_check_middleware(
    context: FunctionInvocationContext,
    next: Callable[[FunctionInvocationContext], Awaitable[None]],
) -> None:
    """拦截 hotel_booking 工具调用，优先会员无房时覆盖为有房。"""
    function_name = context.function.name
    print(f"  [middleware] 拦截 {function_name}...")

    # 执行原始函数
    await next(context)

    # 检查结果并可能覆盖
    if context.result and function_name == "hotel_booking":
        result_data = json.loads(context.result)
        has_availability = result_data.get("has_availability", False)
        is_priority = _current_user in PRIORITY_MEMBERS

        if is_priority and not has_availability:
            print(f"  [middleware] PRIORITY OVERRIDE: 为优先会员 {_current_user} 覆盖可用性!")
            result_data["has_availability"] = True
            context.result = json.dumps(result_data)
        elif not has_availability:
            print(f"  [middleware] 无优先覆盖 (用户: {_current_user})")


async def demo_middleware():
    """函数中间件: 优先会员覆盖酒店可用性。"""
    print("=" * 60)
    print("Demo 4: Function Middleware — Priority Member Override")
    print("=" * 60)

    # 带中间件的酒店预订工具
    @tool(approval_mode="never_require")
    def hotel_booking_mw(
        destination: Annotated[str, "The destination city"],
    ) -> str:
        """Check hotel room availability (with middleware)."""
        print(f"  [tool] hotel_booking_mw({destination!r}) called...")
        cities_with_rooms = ["stockholm", "seattle", "tokyo", "london", "amsterdam"]
        has_rooms = destination.lower() in cities_with_rooms
        return json.dumps({"has_availability": has_rooms, "destination": destination})

    # --- 测试 1: 普通用户预订 Paris (无覆盖) ---
    print("\n--- 测试 1: 普通用户 + Paris (预期无房，无覆盖) ---")
    set_user("regular_user")
    # 直接调用工具函数测试中间件效果 (模拟中间件逻辑)
    result_raw = hotel_booking_mw("Paris")
    result_data = json.loads(result_raw)
    is_priority = _current_user in PRIORITY_MEMBERS
    if is_priority and not result_data["has_availability"]:
        result_data["has_availability"] = True
        print(f"  [middleware] PRIORITY OVERRIDE activated")
    else:
        print(f"  [middleware] 无覆盖 (普通用户)")
    print(f"  结果: has_availability={result_data['has_availability']}")
    print()

    # --- 测试 2: 优先会员预订 Paris (覆盖!) ---
    print("--- 测试 2: 优先会员 + Paris (预期无房 → 中间件覆盖为有房!) ---")
    set_user("priority_user")
    result_raw = hotel_booking_mw("Paris")
    result_data = json.loads(result_raw)
    is_priority = _current_user in PRIORITY_MEMBERS
    if is_priority and not result_data["has_availability"]:
        print(f"  [middleware] PRIORITY OVERRIDE: 为优先会员覆盖可用性!")
        result_data["has_availability"] = True
    else:
        print(f"  [middleware] 无覆盖")
    print(f"  结果: has_availability={result_data['has_availability']}")
    print()

    # --- 测试 3: 优先会员预订 Stockholm (本来有房，无需覆盖) ---
    print("--- 测试 3: 优先会员 + Stockholm (本来有房，无需覆盖) ---")
    set_user("priority_user")
    result_raw = hotel_booking_mw("Stockholm")
    result_data = json.loads(result_raw)
    is_priority = _current_user in PRIORITY_MEMBERS
    if is_priority and not result_data["has_availability"]:
        print(f"  [middleware] PRIORITY OVERRIDE activated")
    else:
        print(f"  [middleware] 无需覆盖 (本来有房)")
    print(f"  结果: has_availability={result_data['has_availability']}")
    print()

    # --- 说明中间件在 Agent 中的使用方式 ---
    print("--- 中间件注入方式说明 ---")
    print("  在 AgentExecutor 中通过 middleware=[priority_check_middleware] 注入:")
    print("    agent = AgentExecutor(")
    print("        client.as_agent(..., tools=[hotel_booking]),")
    print("        middleware=[priority_check_middleware],  # 中间件注入")
    print("        id='agent_with_middleware',")
    print("    )")
    print("  中间件会在每次工具调用时自动拦截，可修改 context.result")
    print()


# ============================================================
# Demo 5: 人在回路 (Human-in-the-Loop) 概念演示
#
# 对应知识点:
#   - Human-in-the-Loop: AI 暂停执行，请求人工审批后再继续
#   - approval_mode="always_require": 敏感操作需人工确认
#   - 对比原笔记本的 RequestInfoExecutor (当前版本不可用)
#     → 用 approval_mode 实现等价的"人在回路"模式
#
# 原笔记本知识点 (概念覆盖):
#   - HandoffBuilder: 动态路由到专家 Agent (用条件工作流模拟)
#   - RequestInfoExecutor: 暂停工作流等待人工输入 (用 approval_mode 模拟)
#   - 人工确认后继续执行
# ============================================================
@tool(approval_mode="always_require")
def book_flight_sensitive(
    destination: Annotated[str, "The destination city"],
    passenger_name: Annotated[str, "Passenger name"],
) -> str:
    """Book a flight — requires human approval before executing."""
    print(f"  [tool] book_flight_sensitive({destination!r}, {passenger_name!r}) called...")
    return (
        f"Flight booked to {destination} for {passenger_name}. "
        f"Confirmation #FLT-{hash(destination) % 10000:04d}"
    )


@tool(approval_mode="never_require")
def search_flights_auto(
    destination: Annotated[str, "The destination city"],
) -> str:
    """Search for available flights — runs automatically without approval."""
    print(f"  [tool] search_flights_auto({destination!r}) called...")
    return f"Found 3 flights to {destination}: prices from $350-$890"


async def demo_human_in_the_loop():
    """人在回路: 对比自动工具与需审批工具。"""
    print("=" * 60)
    print("Demo 5: Human-in-the-Loop — Approval-Based Tool Execution")
    print("=" * 60)

    # 创建带混合工具的 Agent: 查询自动执行，预订需审批
    agent = client.as_agent(
        name="TravelBookingAgent",
        instructions=(
            "You are a travel booking assistant. You can search for flights "
            "and book them. When the user asks to book, use book_flight_sensitive. "
            "When they ask to search, use search_flights_auto."
        ),
        tools=[search_flights_auto, book_flight_sensitive],
    )

    print("\n--- 工具审批模式对比 ---")
    print("  search_flights_auto:  approval_mode = never_require (自动执行)")
    print("  book_flight_sensitive: approval_mode = always_require (需人工确认)")
    print()

    # 测试 1: 搜索航班 (自动执行)
    print("--- 测试 1: 搜索航班 (自动执行，无需审批) ---")
    response = await agent.run("Search for flights to Tokyo")
    print(f"\n[Agent]: {response.text}\n")

    # 测试 2: 预订航班 (需审批 — 在交互式运行器中会暂停请求确认)
    print("--- 测试 2: 预订航班 (需人工审批) ---")
    print("  (在交互式运行器中，book_flight_sensitive 会暂停并请求用户确认)")
    print("  (用户确认后才真正执行预订，从而实现'人在回路')")
    print()

    # 打印工具审批配置
    print("--- 工具审批配置 ---")
    print(f"  {search_flights_auto.name:.<30} approval_mode = {search_flights_auto.approval_mode}")
    print(f"  {book_flight_sensitive.name:.<30} approval_mode = {book_flight_sensitive.approval_mode}")
    print()

    # --- 概念说明: 原笔记本的 RequestInfoExecutor 模式 ---
    print("--- 概念说明: 原笔记本的 Human-in-the-Loop 模式 ---")
    print("  原笔记本使用 RequestInfoExecutor 暂停工作流:")
    print("    1. Agent 检测到需要人工输入")
    print("    2. RequestInfoExecutor 暂停工作流，发出 RequestInfoEvent")
    print("    3. 应用程序收集人工输入 (console/UI)")
    print("    4. 通过 send_responses_streaming() 发送响应")
    print("    5. 工作流恢复执行")
    print("  当前版本用 approval_mode='always_require' 实现等价效果:")
    print("    敏感操作的工具调用会暂停，等待人工确认后才执行")
    print()

    # --- 概念说明: Handoff 模式 ---
    print("--- 概念说明: Handoff 编排模式 ---")
    print("  原笔记本使用 HandoffBuilder 实现动态 Agent 路由:")
    print("    客服 Agent → 评估请求 → 移交给专家 Agent (预订/退款/确认)")
    print("  当前版本用条件工作流 (Demo 3) 实现等价效果:")
    print("    availability_agent → 条件路由 → booking_agent / alternative_agent")
    print()


async def main():
    # Demo 1: 顺序编排 — 前台推荐 → 礼宾评审
    await demo_sequential()
    # Demo 2: 并发编排 — 三个专家 Agent 并行
    await demo_concurrent()
    # Demo 3: 条件工作流 — 根据可用性路由
    await demo_conditional_workflow()
    # Demo 4: 函数中间件 — 优先会员覆盖
    await demo_middleware()
    # Demo 5: 人在回路 — 审批模式
    await demo_human_in_the_loop()


if __name__ == "__main__":
    asyncio.run(main())
