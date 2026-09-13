# 第 7 章学习笔记：规划设计模式（Planning Design Pattern）

> 本章目标：掌握把复杂目标"拆开、结构化、再执行"的规划模式——目标定义与任务分解、结构化输出、多 agent 编排与迭代重规划。
> 核心一句话：**规划模式的本质是把"做什么（规划）"与"怎么做（执行）"解耦：先由一个规划 Agent 把高层目标分解成可校验的结构化子任务（带责任 Agent、优先级、依赖），再交给执行层按依赖顺序落地；计划不是一次性的，要能被结果和用户反馈触发重规划。**

---

## 1. 核心认知：规划 = 目标定义 + 任务分解 + 执行解耦

现实任务大多不可能一步完成，agent 需要一个**简洁、明确的总目标**来指导规划。教程用 `"Generate a 3-day travel itinerary."` 说明：目标写得越清楚，agent（和人）越能聚焦到正确结果（航班 + 酒店 + 活动）。

**任务分解（Task Decomposition）** 是把大目标拆成**面向目标的子任务**：

- 旅行例子：Flight Booking / Hotel Booking / Car Rental / Personalization。
- 每个子任务可交给**专门的 agent 或流程**；再由一个协调/下游 agent 把结果汇总成一份完整行程。
- 模块化的好处是**可增量增强**：后续可加"餐饮推荐""本地活动"等专家 agent，逐步打磨行程。

> 拓展：任务分解的收益（`07-python-agent-framework.ipynb` 明确列出）——**清晰**（每个子任务单一职责）、**可并行**（无依赖的子任务可并发）、**可靠**（失败被隔离在单个子任务内）、**可核算**（成本按子任务估算再汇总）。这正是第 8 章"多 agent 拆分决策"的前置理由。

> 拓展：**规划是"架构级"的模式，不是某个 API**。它回答的是"如何组织多步工作"，而非"怎么调模型"——所以它天然承接第 2 章的多 agent 协作、第 4 章的工具使用，并被第 14 章的 MAF 工作流工程化。

---

## 2. 结构化输出：让计划可被机器消费

规划的价值只有在**下游能解析**时才成立。教程用 **Pydantic 模型**定义计划 schema，让 LLM 返回结构化（JSON）而非自由文本：

```python
class TravelSubTask(BaseModel):
    task_id: int
    description: str
    assigned_agent: str          # 责任 agent（如 flight_agent / hotel_agent / activity_agent）
    priority: str                # high / medium / low
    dependencies: list[int] = []

class TravelPlan(BaseModel):
    destination: str
    trip_duration_days: int
    subtasks: list[TravelSubTask]
    total_estimated_budget_usd: int
    notes: str
```

- **`assigned_agent`** 把子任务路由到专家（教程用 `AgentEnum` 枚举限定可选 agent 集合）。
- **`priority` + `dependencies`** 让执行层知道先做谁、谁等谁。
- 结构化输出让规划结果**可校验、可路由、可汇总**，而不是一段需要再解析的自然语言。

> 拓展：**结构化输出是"规划"和"执行"之间的接口契约**。对照代码演进：README 最早用 `json.loads(response.output_text)` 手动解析字符串；正式示例（`07-python-agent-framework.ipynb`）直接 `options={"response_format": TravelPlan}` 拿到 `result.value` 对象；纯 Python 版用 `ChatOptions(response_format=TravelPlan)` + `TravelPlan.model_validate_json(...)` 校验。**趋势是把"让模型输出合法结构"从提示词技巧升级为 API 原生能力**，减少解析脆弱性（呼应第 4 章 `call_id` 协议化的思路）。

---

## 3. 规划 Agent + 多 agent 编排：语义路由 → 分配 → 汇总

一个典型的规划编排流程（`README` "Planning Agent with Multi-Agent Orchestration"）：

1. **接收请求**：规划 Agent 拿到用户消息（如 "I need a hotel plan for my trip."），结合含**可用 agent 清单**的 system prompt。
2. **生成结构化计划**：输出 `TravelPlan`，把请求分解成带 `assigned_agent` 的子任务（示例里分解出 flight/hotel/car/activities/destination_info 5 个子任务）。
3. **路由到对应 agent**：按子任务数量决定——**单任务**直接发给对应专家 agent；**多任务**通过 group chat manager 协调多 agent 协作。
4. **汇总结果**：规划 Agent 把各 agent 的产出整理成清晰计划交付用户。

> 拓展：规划 Agent 本质上是一个**语义路由器（semantic router）+ 任务调度器**——它自己不做业务，只决定"谁来做、做什么"。这带来一个重要设计原则：**规划层和执行层的工具集要分开**（规划 Agent 通常不需要业务工具，执行 Agent 才持有专家工具），这样规划更轻、更可控，也便于按角色做权限隔离（第 4、6 章的最小权限）。

> 拓展：`is_greeting` 字段是个容易被忽略的**分诊开关**——先判断是不是寒暄/闲聊，是则不走规划流程，直接由 DefaultAgent 回应。生产中类似的"前置分类"能省掉大量无意义的规划开销。

---

## 4. 计划执行：把结构化计划变成动作

规划完成后，执行层（教程里是 **Concierge 礼宾 Agent**）按计划落地（`07-python-agent-framework.py` 的 `demo_execute_plan`）：

- 把 `TravelPlan` 拼成执行 prompt（含目的地、天数、预算、每个子任务的优先级/责任 agent/依赖）。
- 礼宾 Agent 持有**专家工具**：`book_flight` / `reserve_hotel` / `book_activity`（标准 `@tool`，参数用 `Annotated[...]` 说明）。
- 指令要求它**按依赖顺序**逐个执行子任务，完成后汇总。

> 拓展：**注意这是"规划-执行"两段式，而不是一个 agent 从头包办**。规划 Agent 用 `get_response` + `response_format`（因为 `as_agent()` 不支持 `response_format`），执行 Agent 用 `as_agent()` + 工具——两种 API 各擅其长。分离的好处是各段可独立测试：规划对不对看 schema 是否合理，执行对不对看工具调用是否正确。

---

## 5. 迭代规划：反馈触发重规划

有些任务需要**来回与重规划**——某个子任务的结果会影响后续（如订航班时发现意外的数据格式，需要调整策略再订酒店）；用户反馈（如"我想坐更早的航班"）也会触发**部分重规划**。

实现方式（`demo_iterative_planning`）：把**当前计划（序列化为 JSON）+ 新需求**一起交给规划 Agent，让它输出更新后的 `TravelPlan`，并遵守约束——按新时长/预算调整子任务、保持优先级与依赖一致、更新总预算、在 `notes` 里记下改了什么。

> 拓展：**迭代规划是"计划是活的"这一认知的落地**。传统工作流是线性执行到底；规划的动态性体现在——把执行结果和用户反馈当成新的输入，重新过一遍规划。这也把规划与第 9 章（元认知/自反思）连接起来：**"重新规划"本质上是一种对当前策略的自我评估与修正**。教程同时点名 Magentic One——其编排器不只会规划，还会**跟踪任务进度并在需要时重规划**，是规模化版本。

---

## 6. 踩坑 & 易错点

- **目标写得含糊**：`"Generate a 3-day travel itinerary."` 若不给偏好/预算/约束，分解出的子任务会空泛；先明确总目标再分解。
- **只做分解、不做结构化**：一段自然语言"计划"下游没法可靠路由和汇总；要用 schema（`assigned_agent` / `priority` / `dependencies`）把它变成机器可读的契约。
- **忽略依赖与顺序**：子任务有先后（先定航班再定同日期酒店）；执行层必须尊重 `dependencies`，否则结果互相矛盾。
- **把规划 Agent 当执行 Agent 用**：让规划 Agent 又规划又干活，会让它工具过载、职责混乱；**规划层与执行层分离**。
- **计划一次定死**：现实约束会变、用户会改主意；不支持重规划的系统会交出过时答案。
- **重规划无收敛控制**：反馈循环要能停下来，否则会反复重规划（呼应第 5 章"自主性要配预算"）。
- **`response_format` 与 API 混用**：并非所有创建 agent 的方式都支持结构化输出（如某些 `as_agent()`），要按需选 `get_response` / `as_agent`，并对结果做 Pydantic 校验兜底。

---

## 7. 应用心法

- **先写"计划 schema"，再写 agent**：把目标分解成哪几类子任务、各由谁负责、优先级与依赖是什么——schema 定义清楚了，规划和执行都水到渠成。
- **规划层轻、执行层专**：规划 Agent 只做路由/分解（可不给业务工具），执行 Agent 各持窄域工具。
- **一切计划都可校验**：用 Pydantic 约束输出，解析失败时保留原始回复并报错（示例 Python 版就是这么兜底的）。
- **把重规划设计进流程**：预留"当前计划 + 新输入 → 更新计划"的入口，而不是当成事后补丁。
- **按子任务数选择编排**：单任务直连专家 agent，多任务才引入 group chat/协调器，避免过度编排。

---

## 8. 现实映射

- **前接第 1 章的"多步流程"信号**：这章正是那个信号的落地模式；**后接第 8 章多 agent**：`assigned_agent` + group chat manager 就是多 agent 协作的雏形，第 8 章会系统讲 group chat / hand-off / collaborative filtering。
- **与第 4 章的关系**：执行层的 `book_flight/reserve_hotel/book_activity` 是标准工具调用；规划负责"何时调哪个工具"的调度，工具调用负责"真正干活"。
- **与第 5、6 章的关系**：第 5 章的"迭代 + 记忆"是单 agent 的检索自纠错，这里的"迭代规划"是多步任务的自纠错；高风险子任务落到第 6 章的审批/分级。
- **代码文件对应**：`07-python-agent-framework.py`（`demo_planning_agent` 结构化规划 → `demo_execute_plan` 专家工具执行 → `demo_iterative_planning` 反馈重规划）；`README` 的语义路由示例对应 `AgentEnum` + `assigned_agent` 路由。
- **工业化出口**：Magentic One 的"编排器 + 进度跟踪 + 按需重规划"是本章模式在复杂任务上的规模化版本；第 14 章 MAF 的 sequential/concurrent/条件工作流则是它的工程实现。
