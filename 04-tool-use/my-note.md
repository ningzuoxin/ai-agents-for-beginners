# 第 4 章学习笔记：工具使用设计模式（Tool Use Design Pattern）

> 本章目标：吃透 agent 世界的"发动机"——function calling 的完整机制，从裸 API 到框架的两种实现路径，以及工具带来的安全问题怎么防。
> 核心一句话：**工具 = agent 的手脚；function calling 的本质是"LLM 只做选择题和填参数，代码负责执行"——模型返回的永远是调用意图（name + args）而不是结果，执行和回填由你（或框架）完成；权限最小化是安全答案。**

---

## 1. Function Calling 机制：一个必须刻进脑子的认知

这一章最核心的认知（很多人学了很久都没意识到）：

**LLM 从不执行任何东西。它只输出"我想调哪个函数 + 参数"。**

完整循环（教程用"查旧金山时间"走了一遍）：

```
用户消息 ──┐
工具 schema ─┴→ ① LLM ──→ 返回 function_call(name, arguments)   ← 注意：不是答案！
                      ↓
              ② 你的代码执行函数，拿到真实结果
                      ↓
              ③ 结果作为 function_call_output 追加进对话
                      ↓
              ④ 再次调 LLM ──→ 基于工具结果生成最终自然语言回答
```

- ①中模型对比用户请求和**每个函数的 description**，选出最合适的，返回 `ResponseFunctionToolCall(arguments='{"location":"San Francisco"}', call_id=..., name='get_current_time')`。
- ③④缺一不可：把 `response.output` 和 `function_call_output` 都 append 回 `messages`，模型才有完整上下文产出最终回答。
- 这就是第 2 章说的 agentic loop 的最小单元——**一轮工具调用 = 两次 LLM 调用夹一次代码执行**。

> 拓展：**为什么这个设计聪明**——LLM 的输出是"结构化意图"而非自由文本，把不可靠的文本解析变成可靠的协议约定（call_id 配对请求和结果）。安全边界也由此而来：**模型能"选"和"填参"，但"能做什么"始终由你注册的工具集决定**——它永远调不出你没给的东西。理解这点，就明白为什么工具的 description 写作（第 2 章说过 docstring = prompt 工程）直接决定调用准确率：模型选工具的唯一依据就是那段文字。

> 拓展：**协议演进要知道**。OpenAI 工具调用经历了 Chat Completions 的 `tool_calls`（2023）→ **Responses API**（2025，本教程用的 `/openai/v1/` 稳定端点、无需 `api_version`）的 `function_call` / `function_call_output` 项。趋势：Responses API 把 agent 循环原语（工具调用、状态引用）做成一等公民。旧教程/博客里的 `tool_calls` 写法仍能用，但新代码建议按本章的 Responses API 风格写。

---

## 2. 实现路径：裸 API vs 框架（同一件事的两档抽象）

教程给的对比就是第 2 章"SDK vs 平台"的工具版落地：

| | 裸 API（Responses API） | MAF（`@tool` 装饰器） | Agent Service |
|---|---|---|---|
| 你写什么 | schema JSON + 手动解析 tool_call + 手动回填 + 两次调用循环 | `@tool` 函数，框架全自动 | FunctionTool/ToolSet，服务端自动 |
| 循环在哪 | **你手写** | 你的进程，框架驱动 | 服务端 |
| 适合 | 理解原理、精细控制 | 生产默认选择 | 企业部署 |

- **MAF 版**六行搞定：`@tool` 装饰器自动把 Python 函数签名+docstring 序列化成 schema 发给 LLM，"back-and-forth communication"（教程原话）全由框架接管——第 1 节那整个循环你一行都不用写。
- **Agent Service 版**的增量价值（教程列了三条）：**自动工具调用**（解析/执行/回填全在服务端）、**Thread 托管状态**（不用自己管对话历史）、**开箱工具**（不用自己写）。

> 拓展（批判性阅读）：裸 API 那段代码的教学价值 > 实用价值——它让你看清框架帮你干了什么（schema 生成、call_id 配对、结果回填、循环控制）。**第一次学必须手写一遍裸的，之后再换框架**，否则 `@tool` 对你只是魔法。这也解释了教程为何先给 30 行裸代码再给 6 行框架代码。

### Agent Service 的工具两分法：Knowledge vs Action

教程把开箱工具分成两类，这个分类值得记住（判断"该给 agent 配什么"的第一步）：

- **Knowledge Tools（读世界）**：Bing Search（实时信息）、File Search、Azure AI Search（私有数据）——都是**检索**，无副作用，错了顶多答案不准。→ 详见第 5 章 RAG。
- **Action Tools（改世界）**：Function Calling、Code Interpreter、OpenAPI 工具、Azure Functions——**有副作用**（下单/发邮件/跑代码），错了要收拾残局。→ 人在回路、审批就该设在这类工具上（呼应第 2 章 `approval_mode`）。

> 拓展：这个两分法就是安全分级的依据：读类工具可放开（快速、自主），写类工具要收紧（审批/白名单/沙箱）。第 15 章浏览器 agent 的护栏"搜索可自主、提交表单需批准"是同一原则的浏览器版。

---

## 3. 安全：LLM 生成 SQL 的正确焦虑和正确答案

教程直面了一个经典担忧：LLM 动态生成 SQL → 注入风险、DROP/篡改风险。答案分三层：

1. **权限最小化**（核心答案）：给 app 的数据库账号配**只读**——PostgreSQL/Azure SQL 就给 SELECT 角色。LLM 再"想"删库，数据库层面也执行不了。
2. **环境隔离**：企业标准做法是数据先 ETL 进**只读数仓/副本**（schema 也为易用性优化过），agent 只连这个副本。
3. （教程没展开但同属这层）**校验与审批**：写操作走白名单 + 人工批准，输入参数在执行前做确定性校验。

> 拓展：这里藏着整个 agent 安全的总原则——**永远不要用 prompt 来做安全**（"请不要删库"写在 instructions 里等于没写）。prompt 是软约束，权限是硬约束；LLM 的行为要靠**架构兜底**：只读账号、沙箱执行、网络隔离、审批门。推而广之：Code Interpreter 必须跑在隔离沙箱（它能 `rm -rf`）；外部网页内容是 prompt 注入的载体（第 15 章的"页面内容视为不可信输入"同源）。一句话：**把 agent 当成一个聪明但不可完全信任的实习生——能力上放权，权限上收死。**

> 拓展：**Text-to-SQL 是这个模式最典型的应用**（教程代码里的 `fetch_sales_data_using_sqlite_query` 就是）。业界成熟做法比"裸给库"多两步：schema 语义化（表/字段注释写清楚，LLM 才能选对列）+ 生成 SQL 先过只读校验/EXPLAIN 再执行。本章笔记的实践篇可以直接从"给一个 SQLite 只读副本 + 一个查数函数"开始。

---

## 4. 踩坑 & 易错点

- **以为模型"执行"了工具**：模型只返回调用意图；忘了 ③④（回填结果再调一次）程序就只能打出"我调了函数"却没有答案。
- **工具描述含糊**：模型选工具的唯一依据是 description——"查数据"级别的描述 = 工具白给。参数描述同样重要（`"e.g. San Francisco"` 这种示例能显著减少参数格式错）。
- **工具塞太多**：几十个工具全塞给模型 → 选择困难、误调用率上升。窄域 agent 配 5-10 个精选工具比"万能 agent"配 50 个好用（第 8 章拆 agent 的动机之一）。
- **错误处理缺失**：工具抛异常直接炸循环——工具失败要把错误信息**作为结果回填**给模型，让它自己决定重试还是换路（六章件套里的 Error Handling & Validation 就是干这个的）。
- **拿 prompt 当安全边界**：见第 3 节，instructions 里写"禁止删库"防不住任何东西。
- **schema 和函数签名漂移**：手写 schema 时 JSON 里的参数名/类型和实际函数不一致 → 运行时才炸；用 `@tool` 自动生成就是为消灭这个漂移。

---

## 5. 选型心法

- **学习原理 / 极致控制**：裸 Responses API（本章前半的写法），适合理解协议和做框架无关的库。
- **生产默认**：MAF `@tool`——代码最少、loop 自动、可观测性接 Foundry traces。
- **企业托管 / 要开箱工具**：Agent Service ToolSet——服务端执行 + Thread 状态 + Bing/Search/Code Interpreter 白嫖。
- **工具分类先于写代码**：先列清单分 Knowledge/Action 两栏，读类放开写类收紧，再决定每类怎么实现。

## 6. 现实映射

- 本章机制 = 全课程代码的底座：第 5 章 RAG（File/AI Search 就是 Knowledge Tools 专题）、第 6 章（错误处理与可靠性是六章件套的展开）、第 7 章规划（多步工具链的调度策略）、第 8 章多 agent（工具集按角色拆分的前提）。
- `approval_mode`（第 2 章伏笔）在本章语义明确：Action Tools 的审批开关，第 16 章生产化时升级为完整人在回路确认流。
- Text-to-SQL 只读副本模式 → 企业里最常见的第一个落地场景（运营问数机器人）；第 15 章浏览器 agent 的"观察自主/提交审批"是同一原则的 UI 版。
- 教程章末的 smoke-test（`tests/lesson-04-smoke-tests.json`）验证的正是"agent 还会不会调工具"——工具调用链路是 agent 最容易悄悄坏掉的部分，部署后要常测。
