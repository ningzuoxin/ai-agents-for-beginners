# 第 17 章学习笔记：用 Foundry Local + Qwen 造本地 AI Agent

> 核心一句话：**把 agent 从云上搬回单机——用 SLM 当"调度员"、工具干重活，换来隐私 / 零 token 成本 / 离线可用；代价是放弃了前沿大模型，所以要让模型只 orchestrate、不扛知识。**

---

## 1. 最该记住的几个点

- **本地 agent 的三大动机**：隐私（代码/数据不出机）、成本（无按 token 账单）、离线（飞机上/安全环境/断网照跑）。代价是拿前沿云模型换本地 SLM——**接受约束、在约束内把 agent 做好**，而不是假装约束不存在。
- **SLM 的正确分工 = 模型 orchestrate，工具干重活**：SLM 擅长的是"决定调哪个工具、传什么参数"（bounded 决策），弱在广博知识、长程多跳推理。所以本地 agent 的赢法是——让它只会调度 `read_file` / `search_docs`，而非指望它背下你的代码库。这直接扬长避短。
- **Foundry Local 的杀手锏是 OpenAI 兼容端点**：它把模型在本地起成一个 `http://localhost:PORT/v1` 的 OpenAI-compatible 服务。于是云上 agent 代码**只改 `base_url` 就能搬下来**，其余不变。这才是"复用云代码"的真正原因。
- **为什么是 Qwen（而非随便一个 SLM）**：agent 必须产出可靠、格式正确的 tool call；很多 SLM 能聊天但工具调用残缺/不稳定。Qwen 是专为 function calling 训练的，能稳定吐出合法工具调用结构——这才是"本地聊天模型"变"本地 agent"的关键。
- **本地 RAG 全链路在机内**：本地 embedding 模型 → 本地向量库 Chroma（磁盘、进程内、无服务）→ 本地检索 → 本地 SLM。就是第 5 章 Agentic RAG，只是每个组件都跑本机。
- **本地 MCP 不是自动安全**：MCP 是传输协议不是云服务，可 `stdio` 起本地进程跑。但它以你的用户权限运行、能碰你碰得到的一切——照样要限范围（项目目录而非 home）、输出当不可信输入校验。
- **混合路由 = 第 16 章 model routing 的延伸**：把"本地机器"当成其中一个 model。敏感/离线/简单任务走本地 SLM；非敏感数据的硬推理走云；云不可用时降级到本地（质量降级而非直接挂）。
- **本地 SLM 的天花板（何时该回云）**：SLM 弱在长上下文、多跳推理、稀有/前沿知识——这正是它"只 orchestrate、不扛知识"的根本原因。所以混合路由的反面也要记住：遇到长文档摘要、跨多源推理、需要最新世界知识时，别硬撑本地，路由回云大模型。本地 agent 的取舍不是"云更好就全云"，而是"本地管调度与隐私、云管重推理"。

### 拓展：工具从何而来（接上一条"模型 orchestrate、工具干重活"）

**工具不是模型生成的,是你写进代码再注册给 agent 的函数。** SLM 只"决定调哪个、填什么参",真正干活的是你定义的普通 Python 函数——模型不"拥有"工具,只是被授权在每次请求里从你给的 tool 清单里挑。

三类来源：
- **① 手写函数**：本章 `get_weather` / `rag_search` 即如此,带 `@tool`/schema 描述后交给模型。
- **② 本地 MCP Server**：README 提到的 local MCP,以 `stdio` 起一个本机进程暴露 tools。
- **③ 现成 SDK 封装**：文件读写、搜索 API 等包成函数。

**本地场景的关键延伸**：tools 和模型**同机执行**——这是"本地 agent"相对"云 agent"的第二个含义(第一个是模型在本地):不仅模型在本地,工具执行也在本地,隐私/离线优势才成立。工具本质是普通代码,可访问本机文件、本地向量库,也可(可选)调外部 API。

**那 read_file 这种工具必须自己手写吗?——不一定,看用的什么层：**
- **裸 OpenAI SDK(本章 notebook 做法)**：要自己写。模型只认你传的 `tools` schema,所以 `read_file` 也需定义函数 + 写 JSON schema + 在循环里按模型返回的名字分发执行(`if name=="read_file": read_file(**args)`)。简单工具也逃不掉这层包裹——这是手搓 tool-calling 循环的代价。
- **Microsoft Agent Framework(前第14/16章)**：通用工具多现成(内置 Files / CodeInterpreter / Bing 等),`read_file` 类常直接挂;只有业务特有工具才需 `@tool` 手写。
- **MCP**：通用工具(如 file-system server)社区/官方已有现成 server,你"拿来起进程用"而非自己写函数——但 server 背后终归有人写了那个函数,只是不是你。
- **规律**：通用工具(read_file/写文件/搜网页)优先复用框架内置或 MCP 现成 server,别造轮子;业务特有工具(查你自己的系统)任何模式都躲不掉、必须手写——那才是 agent 真正的附加值。

## 2. 如何部署一个本地 SLM（核心步骤）

本地 SLM 不是凭空来的——它靠 **Microsoft Foundry Local** 这个本地运行时把模型下载、起服务、并暴露成 OpenAI 兼容端点。这才是"本地 agent"的地基，云上那步是 Foundry 托管、本机这步是你自己起：

1. **装 Foundry Local**（按 OS 文档，如 Windows `winget install Microsoft.FoundryLocal`）。
2. **下载并启动模型**（Foundry Local 自动选适合你硬件的 build：CPU / CUDA / NPU）：
   ```bash
   foundry model run qwen2.5-7b-instruct
   foundry service status
   ```
3. **在代码里接管**：`FoundryLocalManager` 负责下载/起服务并自动发现本地端点（不用硬编码端口），再把它当 OpenAI 兼容端点连上：
   ```python
   manager = FoundryLocalManager("qwen2.5-7b-instruct")   # 下载+起服务+发现端点
   client = OpenAI(base_url=manager.endpoint, api_key=manager.api_key)  # key 是本地占位，无云凭证
   ```

> 一句话：云上 agent 的模型由 Foundry 平台托管；本地 agent 的模型由 **Foundry Local 在你机器上起成 localhost 服务**——其余 agent 代码两者几乎一样。

### 部署好后，怎么用客户端连本地 SLM？（关键区分）

**`FoundryLocalManager` 不是推理客户端,它只负责"起模型 + 发现端点"。** 真正发请求干活的是 **OpenAI 兼容客户端**。两种连法:

- **Foundry Local 方案(本课原生)**：`FoundryLocalManager` 下载/起服务并自动告诉你端点,然后**用 `OpenAI` 客户端连**——它就是个便捷封装,帮你省去硬编码端口:
  ```python
  manager = FoundryLocalManager("qwen2.5-7b-instruct")   # 起服务 + 发现端点
  client = OpenAI(base_url=manager.endpoint, api_key=manager.api_key)  # 真正发请求的客户端
  # tools 不是传给客户端的,是传给每次 create(...) 的参数:手写函数 + 手写 JSON Schema 描述
  def read_file(path: str) -> str:
      with open(path, encoding="utf-8") as f: return f.read()
  tools = [{"type": "function", "function": {
      "name": "read_file", "description": "读取指定路径的文本内容",
      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
  resp = client.chat.completions.create(
      model="qwen2.5-7b-instruct", messages=[{"role": "user", "content": "读一下 main.py"}],
      tools=tools,                    # ← 工具清单随每次请求传;模型据其决定调哪个、填什么参
      # tool_choice="auto",          # 可选:auto=模型自决,或指定函数名强制调
  )
  # 模型若决定调用:resp.choices[0].message.tool_calls 含 name+arguments → 你代码执行后把结果塞回 messages 继续下一轮
  ```
- **Docker + Ollama 方案(本机推荐,见 docker-compose.yml)**：**根本不用 `FoundryLocalManager`**——Ollama 容器自己起服务,直接拿端点连:
  ```python
  client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")  # Ollama 默认端点
  # 同样手写函数 + tools 描述,调用时 tools=tools 随请求传(与上面完全一致,只换 base_url / model)
  def read_file(path: str) -> str:
      with open(path, encoding="utf-8") as f: return f.read()
  tools = [{"type": "function", "function": {
      "name": "read_file", "description": "读取指定路径的文本内容",
      "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
  resp = client.chat.completions.create(
      model="qwen2.5:7b", messages=[{"role": "user", "content": "读一下 main.py"}],
      tools=tools,
  )
  ```
  > 另两种来源:用 **Microsoft Agent Framework** 时 `@tool` 装饰器自动把函数签名转成 tools 描述、框架内部帮你传(底层仍是同一 `tools` 数组);用 **MCP** 时 tools 来自 MCP server,SDK 拉取清单转成同样的 `tools` 数组再传。

  **要不要手写 tools 的 JSON Schema?——只看你用裸 SDK 还是框架,与用哪个 runtime 无关:**
  | 方式 | 描述 tools 的形式 | 说明 |
  | --- | --- | --- |
  | `FoundryLocalManager` + `OpenAI`(本课 notebook) | **手写完整 JSON Schema** | Manager 只管起服务,不碰 tools |
  | `OpenAI` 直连 Ollama / llama.cpp / vLLM(裸 SDK) | **手写完整 JSON Schema** | 同裸 SDK 模式,都得自己写 |
  | **Microsoft Agent Framework** `@tool` | **写 `@tool` 注解即可** | 框架从函数签名自动生成 schema、内部传递 |
  | **MCP** | **不用你写** | server 提供 tool 清单,SDK 转成数组传给 `create(...)` |
  > 要点:`FoundryLocalManager` 不免去 schema 工作——它和"是否手写 tools 描述"无关,那只取决于裸 SDK 还是框架。换 runtime(Ollama→Foundry Local)只要 `tools` 数组结构不变,agent 逻辑完全复用。

**共性(再次印证"认端点、别绑 runtime")**：无论 Foundry Local / Ollama / llama.cpp / vLLM,agent 代码最终都 via **同一个 OpenAI 兼容客户端**(`openai` 库的 `OpenAI`,或直接打 `/v1/chat/completions`)。`FoundryLocalManager` 是 Foundry Local 专属的起服务封装,换了 runtime 就不需要它——你连的永远是 `base_url` 指向的那个 localhost 端点。**tools 跟着每次请求走**(非连上时注册一次永久有效),模型每次基于你传的 `tools` 清单决策、不持久记住工具;换 runtime 只要 `tools` 数组结构不变,agent 逻辑完全复用。

### 拓展：其他安装本地 LLM 的方式
**不是，它只是"最省心 + 和本课前面 Foundry 体系对接最顺"的那一个，绝不是本地推理的最优解或唯一解。** 真实生态里还有更主流/更轻量的替代，而且它们**都暴露 OpenAI 兼容端点**——所以换 runtime 不影响本章 agent 代码，只改 `base_url`：

| 运行时 | 定位 | 何时比 Foundry Local 更合适 |
| --- | --- | --- |
| **Ollama** | 社区最主流的"傻瓜式"封装（基于 llama.cpp），一行 `ollama run qwen2.5:7b` 即起，自带 REST API | 想最小阻力、不绑微软生态；Qwen 等模型开箱即用 |
| **llama.cpp** | 极致轻量的纯 C++ 引擎，量化先驱，CPU/树莓派都能跑 | 低资源设备、要精细控制量化与显存、嵌入式/边缘 |
| **vLLM** | 生产级高吞吐推理服务（需 GPU） | 本地但要并发扛多用户、做服务端级吞吐 |
| **Foundry Local** | 微软本地运行时，和 Foundry 门户/SDK 一脉相承 | 已在用 Foundry、想云↔边同套工具链、少折腾 |

**硬件要求差异**（核心看两件事：权重放哪 RAM/VRAM + 谁来计算 CPU/GPU/NPU）：

| 方式 | 最低内存 | 加速器 | 权重位置 | 跑 7B 模型典型门槛 |
| --- | --- | --- | --- | --- |
| **Foundry Local** | 8 GB RAM | CPU/GPU/NPU 自动选 | RAM（量化） | 8 GB 可跑、16 GB+ 舒服 |
| **Ollama** | 8 GB（Apple 统一内存更佳） | 优先 GPU，可纯 CPU | RAM / 统一内存 | 同上；Apple Silicon 最顺 |
| **llama.cpp** | 4–6 GB（量化极致） | CPU/CUDA/Metal/NPU 可分层卸载 | RAM 或 VRAM | 树莓派级老旧设备也能跑 |
| **vLLM** | 需独显，~16 GB 显存 | 必须 NVIDIA/AMD GPU | VRAM（显存） | 独显起步，量化（AWQ/GPTQ）可降到 ~6–8 GB |

> 规律：前三者是"本地能跑就行"（RAM 为主、加速器可选）；vLLM 是"本地但要扛并发"（显存为主、必须独显）。单机玩 agent 选前三者，本地搭服务端级 API 才上 vLLM。

**要点**：本章选 Foundry Local 是"教学连贯性"而非"技术必要"。理解这一点很重要——你学的是"agent 怎么连一个本地 OpenAI 兼容端点"，至于那个端点背后是 Foundry Local / Ollama / llama.cpp 无所谓。迁移到其他 runtime 时，Qwen 这类模型名照用，只需把 `base_url` 换成对应本地址（如 `http://localhost:11434/v1`）。

**各安装 LLM 方法简介：**

**① Foundry Local（微软）**
- 官网 / 文档：https://learn.microsoft.com/en-us/azure/foundry-local/  ｜  GitHub：https://github.com/microsoft/Foundry-Local
- 步骤：
  1. 按 OS 安装（Windows 示例：`winget install Microsoft.FoundryLocal`）。
  2. 拉起模型：`foundry model run qwen2.5-7b-instruct`（自动选 CPU/CUDA/NPU build）。
  3. 查服务：`foundry service status` 看本地端点。
  4. 代码里 `FoundryLocalManager("qwen2.5-7b-instruct")` 自动发现端点 → 接 `OpenAI(base_url=..., api_key=...)`（key 为本地占位，无云凭证）。

**② Ollama（社区最主流，傻瓜式）**
- 官网：https://ollama.com/  ｜  文档：https://docs.ollama.com/
- 步骤：
  1. 官网下安装包（Win / Mac / Linux），装完自带本地服务，默认端点 `http://localhost:11434`。
  2. 拉起模型：`ollama run qwen2.5:7b`（首次自动下载，随后常驻）。
  3. 代码里直接 `OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")` 接上即可，无需 `FoundryLocalManager`。
  4. （可选）`ollama pull` 预下载、`ollama list` 看已装模型。

**③ llama.cpp（极致轻量，纯 C++ 引擎）**
- 官网 / 源码：https://github.com/ggml-org/llama.cpp
- 步骤：
  1. 选途径：要么编译 `llama-server`，要么用封装版（如 `llama-cpp-python` 的 `python -m llama_cpp.server`）。
  2. 下载 GGUF 量化模型（如 Qwen2.5-7B 的 Q4_K_M.gguf，HuggingFace / 官网均有）。
  3. 起服务：`./llama-server -m qwen2.5-7b-q4_k_m.gguf --port 8080`（暴露 OpenAI 兼容端点）。
  4. 代码里 `OpenAI(base_url="http://localhost:8080/v1", api_key="sk-")` 接上。适合低资源 / 边缘设备，但步骤比 Ollama 手动。

**④ vLLM（生产级高吞吐，需 GPU）**
- 官网：https://vllm.ai/  ｜  文档：https://docs.vllm.ai/
- 步骤：
  1. 准备 Linux + NVIDIA/AMD GPU 环境，`pip install vllm`（或 Docker 镜像）。
  2. 起 OpenAI 兼容服务：`vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000`。
  3. 代码里 `OpenAI(base_url="http://localhost:8000/v1", api_key="token")` 接上。
  4. 适合本地但要并发扛多用户的服务端级场景；吞吐远高于前三者，但吃显存、且不在 Windows 原生友好。

> 共性：四种方式最终都给你一个 **OpenAI 兼容端点**，所以本章 agent 的 tool-calling 循环代码**一份通用**——真正要换的只有 `base_url` 与模型名（Qwen 在各家都可用）。选型看约束：省心/微软链→Foundry Local，最小阻力→Ollama，极端轻量/边缘→llama.cpp，高并发→vLLM。

**拓展：把本地 LLM 装进本地 Docker 是否可行？**
**完全可行，且是隔离/复现场景的推荐做法。** 本质是把上面的 runtime 跑在容器里，再把 OpenAI 兼容端点 `docker run -p` 暴露到 `localhost`，agent 代码一行不动（仍只换 `base_url`）。

- **为什么更好**：环境隔离（不污染本机、不手装 CUDA/Python 依赖）、可复现（`docker pull` 即同环境）、一键起服务。
- **各方式在 Docker 里的落地**：
  - **Ollama**：官方镜像 `ollama/ollama` → `docker run -d -p 11434:11434 ollama/ollama`，再 `docker exec` 内 `ollama run qwen2.5:7b`。最省心。
  - **vLLM**：官方镜像 `vllm/vllm-openai` → `docker run --gpus all -p 8000:8000 vllm/vllm-openai --model Qwen/Qwen2.5-7B-Instruct`。官方首推方式。
  - **llama.cpp**：官方服务镜像 `ghcr.io/ggml-org/llama.cpp`，挂载 GGUF 后起 `llama-server`。
  - **Foundry Local**：相对特殊——它是桌面级原生运行时，主打本机安装（winget/brew），非为容器设计；Docker 里跑需自己手搓镜像，社区不主流。故"Docker 跑 LLM"通常选前三者而非它。
- **GPU 前提**：纯 CPU 直接跑（慢）；要用宿主 GPU 需装 **NVIDIA Container Toolkit**（Windows 走 WSL2 + Docker Desktop + `nvidia-container-toolkit`），容器加 `--gpus all` 才能见显卡——这是 Docker + 本地 LLM 唯一的额外门槛。
- **衔接本章**：教程用 Foundry Local 原生装；若你本机用 Docker 跑 Ollama，agent 的 `OpenAI(base_url="http://localhost:11434/v1")` 代码一行不动，再次印证"认 OpenAI 兼容端点、别绑具体 runtime"。

**针对本机配置的推荐（实测环境：Win11 / 32 GB RAM / RTX 4060 8 GB 显存 / Docker 28 已装、但 nvidia runtime 未配置）**：
- **方式选 Ollama**：vLLM 需 ~16 GB 显存（你仅 8 GB，量化余量也小且配置繁琐）、llama.cpp 需手动挂 GGUF、Foundry Local 非容器设计——Ollama 官方镜像一键起、`qwen2.5:7b` 开箱即用，最省心。
- **模型选 Qwen2.5-7B 的 Q4 量化**（`qwen2.5:7b`）：8 GB 显存刚好装下（权重 ~4.5 GB + KV cache 留 2–3 GB 余量）；更小(3B)工具调用不稳、更大(14B+)放不下。7B 是"质量 vs 显存"最优平衡点，且 function calling 训练好、本地 agent 最稳。
- **必补一步**：当前 Docker 看不到 GPU，需装 **NVIDIA Container Toolkit**（WSL2 + Docker Desktop + `nvidia-container-toolkit`），否则只能 CPU 慢跑；装好后容器加 `--gpus all` 才能用上 4060 的 8 GB 显存。
- **落地命令**：`docker run -d --gpus all -p 11434:11434 -v ollama:/root/.ollama ollama/ollama` → `docker exec -it ollama ollama run qwen2.5:7b` → agent 接 `OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")`。
- **一键配置**：同级目录已提供可直接用的 `docker-compose.yml`（Ollama + Qwen2.5-7B，含 GPU 预留与模型持久化卷），用法 `docker compose up -d` → `docker compose exec ollama ollama run qwen2.5:7b`。

## 3. 关键事实速查

- **硬件底线**：~8 GB RAM 真实可用下限，16 GB+ 舒服；GPU/NPU 加速但不必，无加速器时 Foundry Local 自动选 CPU build。
- **沙箱是必须的，哪怕本地**：工具一律限定在项目根目录内（`_safe_path` 解析后校验 `PROJECT_ROOT in parents`），否则一个能读任意路径的工具 = 以你权限横扫全盘。
- **tool-calling 循环是手搓的**：本章不用 agent_framework，而是用 OpenAI SDK 标准的 `tools=` schema + 自己写的 `run_agent` 循环（模型要工具→本地执行→把结果喂回→反复直到出最终答案，`max_iterations` 兜底）。这正是因为它兼容 OpenAI 接口才这么简单。
- **循环为什么是多轮的（本地 agent ≠ 一次性聊天）**：模型一次回复往往只产出"调工具"而非"最终答案"——它说"我要调 read_file"，你执行完把结果塞回 `messages` 再问，模型才可能给出真正回答；复杂任务会反复好几轮。所以 `run_agent` 必须循环，不能 `create()` 一次就完。这正是 §56–72 注释里"塞回 messages 继续下一轮"的含义。
- **量化（quantization）是本地 LLM 可行的基石**：硬件表里的 Q4 / AWQ / GPTQ 都是量化——把模型权重从 16 位浮点压成 4 位整数等，体积与显存占用骤降（7B 全精度 ~14GB → Q4 ~4.5GB），代价是精度略损。这就是为什么 7B 能塞进你 8GB 显存：靠量化，不是靠小模型。

## 4. 与第 16 章的对照（部署弧的两端）

| 维度 | 第 16 章（上云） | 第 17 章（下机） |
| --- | --- | --- |
| 目标 | 规模化、可靠、省钱的生产服务 | 隐私、零成本、离线可用 |
| 模型 | 云上大模型（可路由大小） | 本机 SLM（Qwen） |
| 客户端 | FoundryChatClient / OpenAIChatCompletionClient | OpenAIChatCompletionClient 或裸 OpenAI SDK，仅 `base_url` 不同 |
| 状态/可观测 | 外置 thread store、OTel、评估门 | 单机内，重点在沙箱与本地 RAG |

> 细节、完整 notebook 代码、Knowledge Check 答案 → 回看原 README（`17-creating-local-ai-agents/README.md`）与笔记本（`17-local-agent-foundry-local.ipynb`）。
