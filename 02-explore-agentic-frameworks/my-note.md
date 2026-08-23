# 个人总结

## 1. MAF vs Foundry Agent Service

1. MAF，SDK。核心概念，Client，Agent，Tools，AgentSession。
2. Foundry Agent Service，类比为一个 Web API 服务，客户端调用 Agent Service 完成各种实例对象的创建及动作的执行。核心概念，Agent，Thread，Message，Run。

---

# 第 2 章学习笔记：探索 Agent 框架（Explore AI Agent Frameworks）

> 本章目标：搞清楚 Agent 框架解决什么问题，以及——本章真正的核心——分清微软两套东西：**Microsoft Agent Framework (MAF)** 和 **Microsoft Foundry Agent Service** 到底谁是谁、怎么选。
> 核心一句话：**MAF 是 SDK（库，agentic loop 跑在你的代码里），Foundry Agent Service 是平台（agent 和会话状态托管在服务端）；路径就是教程那句"先用 MAF 迭代逻辑，要用 Agent Service 部署扩展"。**

---

## 1. Agent 框架 vs 传统 AI 集成：分水岭是自主性

教程开头用"个性化/自动化/体验"介绍传统 AI 框架——这些其实只是普通 AI 特性，不是分水岭。真正的区别在后半段：**Agent 框架给的是让 agent 自主干活的基建**：

- **多 agent 协作与协调**：创建能互相通信、分工的多个 agent。
- **任务自动化与管理**：多步工作流、任务委派、动态任务管理。
- **上下文理解与适应**：基于实时信息做决策，随环境变化调整。

> 拓展：把第 1 章的五要素对过来就清楚了——传统 AI 集成只用 LLM 的"生成"能力；Agent 框架把**循环（loop）+ 工具调用 + 状态管理**这三件事工程化，让你不用手写 while 循环、不用自己解析模型输出的工具调用 JSON。框架的本质价值 = **把 agentic loop 从"你手写"变成"框架提供"**，你只填三样：模型、指令、工具。

---

## 2. 本章核心：MAF vs Foundry Agent Service（SDK vs 平台）

两套东西名字像、都能建 agent，这是初学者第一大困惑源（教程自己也承认 "It does sound like there is overlap"）。一张表分清：

|             | **Microsoft Agent Framework (MAF)** | **Microsoft Foundry Agent Service**                                 |
| ----------- | ----------------------------------- | ------------------------------------------------------------------- |
| 本质        | **SDK / 库**（pip 装进你的应用）    | **平台 / 托管服务**（在 Foundry 里运行）                            |
| loop 在哪跑 | 你的进程里（`agent.run()`）         | 服务端（`create_and_process_run`）                                  |
| 会话状态    | 你的代码自己管                      | **Thread 对象存在服务端**                                           |
| 核心概念    | Agent、Tools、Azure Identity        | Agent（"智能微服务"）、Thread & Messages                            |
| 内置能力    | 工具调用、会话管理、keyless 认证    | 多模型直连（Llama/Mistral/Cohere）、Azure AI Search、Bing、代码执行 |
| 定位        | 快速构建、迭代 agent 逻辑           | 企业级安全部署、规模化                                              |

> 拓展：**类比 Web 开发**——MAF ≈ Express/Spring（嵌入你应用的框架库），Agent Service ≈ App Service/云托管（把应用托管出去换运维和企业能力）。不是二选一，是**开发期和部署期的两件套**，所以教程的最终建议是串联使用而非取舍。

> 拓展：**心智模型——客户端就是个薄 HTTP client**。Agent Service 的客户端 SDK（`azure.ai.projects`）不包含任何 agent 逻辑，它只负责告诉服务端"创建什么资源、触发什么动作"：`create_agent` / `create_thread` / `create_message` / `create_and_process_run` 全是 REST 调用，在服务端创建实体并执行。四个对象（agent/thread/message/run）都是**有 ID 的服务端资源**——可 list、可持久、Foundry 门户可见，和创建云资源（虚拟机、存储桶）是同一范式；`create_and_process_run` 本质是"提交长任务并等它跑完"，agent 干活时消息不断 append 进 thread。

> 拓展：**MAF 的"本机"跑的到底是什么**。MAF agent 也不是全本地——LLM 推理仍是远程 API 调用，真正在本机的是**循环控制 + 工具执行**。所以准确的问题是"loop 和工具在哪跑"，而不是"智能在哪"。这也带出两者最深 practical 差异——**工具能在哪执行**：MAF 的工具是你本机 Python 函数，可直接访问你的数据库/本地文件/内网服务，但服务进程得自己保活；Agent Service 的工具必须能被服务端执行（内置 code interpreter/Bing/Search，或 OpenAPI 指到的端点、Azure Functions 回调），受限但天然持久、可恢复。这就是"两段而非竞品"的根本原因。

> 拓展：**Thread 是理解 Agent Service 的钥匙**。MAF 里对话历史是你进程里的一个列表；Agent Service 里 `create_thread()` 之后，所有 message、中间状态都持久化在服务端——客户端变成无状态的，天然获得：断线续聊、跨请求恢复、审计轨迹。代价：状态不在你手里、多一跳网络延迟。这套 **agent / thread / run / message** 的 API 形状源自 OpenAI Assistants API——用过 Assistants API 的话会觉得眼熟。

> 拓展：**MAF 的来历**。它是微软 2025 年把 **Semantic Kernel**（企业级流程编排）和 **AutoGen**（多 agent 对话研究）合并统一的产物，Agent Service 则由 Ignite 2024 发布的 Azure AI Agent Service 演进、并入 Microsoft Foundry。知道这条时间线，看旧文章里三个名字混用时就不会乱。

---

## 3. 代码精读：一个最小 agent 长什么样

教程两段代码其实在演示同一件事的两种写法，值得精读的是几个模式：

**最小 agent = 模型 + 指令 + 工具**（第 1 章五要素的代码落地）：

```python
provider = FoundryChatClient(project_endpoint=..., model=..., credential=AzureCliCredential())
agent = provider.as_agent(name="travel_agent", instructions="...", tools=[book_flight])
response = await agent.run("I'd like to go to New York on January 1, 2025")
```

- **`@tool` 装饰器**：普通 Python 函数一键变工具，框架负责把签名转成模型能读的 schema、把模型的调用请求转回函数调用。
- **docstring 就是工具说明书**：`"""Book travel given location and date."""` 是 LLM 判断"何时调这个工具"的唯一依据——写得含糊，工具就永远不被调用或被错调。**给工具写 docstring 本质是 prompt 工程**。
- **`approval_mode="never_require"`**：工具审批开关的伏笔——默认策略可以要求人工批准工具调用（人在回路），这里显式关掉。对应第 8 章构建块里的 Human in the loop。
- **`AzureCliCredential`**：keyless 认证，`az login` 过就能跑，API 密钥不落代码（第 0 章环境准备的回收）。

> 拓展（批判性阅读）：教程"多 agent 协作"那段示例其实**只是手动串接**——`retrieval_result` 拼进字符串喂给下一个 agent 的 `run()`。这不算编排，真正的多 agent 工作流（sequential/concurrent、状态传递、条件路由）要等第 14 章 MAF 的 `AgentWorkflow` 才正式登场。读教程代码时留意这种"简化演示"和"生产写法"的差距。

---

## 4. 快速迭代三件套（压缩版）

教程给的三个迭代抓手，知道即可：

- **模块化组件**：AI/记忆连接器、函数调用插件、prompt 模板都是现成的，组装而非从零造。
- **协作工具**：按角色设计 agent，测试 refine 协作流程。
- **实时学习**：反馈闭环，从交互中调整行为（对应第 1 章 Learning agent）。

---

## 5. 踩坑 & 易错点

- **把 MAF 和 Agent Service 当竞品做选择题**：它们是开发链路的两段，教程原话"still confused → 先 MAF 后 Agent Service"。
- **docstring 敷衍**：工具描述是给 LLM 看的 API 文档，"查数据"这种描述会让工具形同虚设。
- **教程的多 agent 示例 ≠ 生产编排**：手动字符串拼接会丢结构、丢状态，别照抄进生产（等第 14 章）。
- **状态放哪没想清楚**：MAF 默认状态在内存里，服务重启对话就没了——要么自己持久化，要么上 Agent Service 的 Thread。
- **追新框架焦虑**：这章是 2024-2025 快速演进的产物，API 会变，**值得长期持有的是概念**（loop/thread/工具协议），不是某个 API 拼写。

---

## 6. 选型心法

- 学习/原型：MAF，几行代码起步，本仓库所有 notebook 都是这条线（`requirements.txt` 里的 `agent-framework`）。
- 企业部署：Agent Service（安全、合规、托管 Thread、内置 Search/Bing/代码执行），第 16 章完整走一遍。
- 不想绑 Azure：横向看 LangGraph（图编排）、CrewAI（角色组队）、OpenAI Agents SDK——概念全部通用，换了框架第 3 章的模式照样适用。
- 本地/离线：第 17 章 Foundry Local，同一套 MAF 代码换本地模型端点。

## 7. 现实映射

- 本章两段代码 = 本仓库每个 notebook 的固定开场（`FoundryChatClient` + `as_agent` + `run`），后面 15 章代码全部长在它上面。
- Thread/run/message 概念在第 5 章（RAG 的服务端会话）和第 16 章（部署后 agent 在 Foundry 门户可见可管）反复出现。
- `@tool` 审批模式 → 第 8 章人在回路；多 agent 串接 → 第 14 章 AgentWorkflow 的正规化；框架对比 → 第 2 章预习了第 16 章的部署决策。
