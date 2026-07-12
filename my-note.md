## prompt

```
以这份笔记中的示例代码为基础，结合前面章节的纯python版的代码示例，重新生成当前章节的纯python版本的示例代码。要求：
1. 必须保证覆盖原示例代码中的所有知识点。
2. 代码风格和前述几章的代码风格保持一致。
```


```
以当前目录下每份笔记中的示例代码为基础，结合前面章节的纯python版的代码示例，重新生成当前章节的纯python版本的示例代码。要求：
1. 必须保证覆盖原示例代码中的所有知识点。
2. 代码风格和前述几章的代码风格保持一致。
```

---

# 笔记 → 纯 Python 代码示例 改写指导说明（以下为补充的规范）

> 用途：当要求把某章 `.ipynb` 笔记里的示例代码改写为「可直接运行的纯 Python 脚本」时，
> 依据本说明统一风格、格式与适配规则。上方 `## prompt` 为你原始写好的触发指令，已原样保留。
>
> **适用范围**：主要针对使用 `agent_framework` 的笔记（绝大多数章节）。非 `agent_framework` 的课（见 §9）只改客户端/异步入口，保留原 SDK。

---

## 1. 总原则

1. **知识点全覆盖**：原笔记每个 cell 演示的概念、模式、工具、Agent 都必须保留；若某 API 新版不可用，用等价方式**模拟实现**并在注释说明「原笔记用 X，本版用 Y 模拟，知识点不变」。
2. **可直接运行**：不依赖 Jupyter，`python xxx.py` 直接执行；入口用 `asyncio.run(main())`，所有 `await` 包在 `async def` 里。
3. **风格一致**：沿用下方规定的结构、命名、中文注释、客户端与 API 适配方式。
4. **忠实优先**：尽量 1:1 还原笔记；仅当原 API 在新版（agent-framework >= 1.8.0）删除时才适配。

---

## 2. 文件命名与位置

- 一个 `.ipynb` 对应一个同名 `.py`，沿用原笔记 stem（`09-python-agent-framework.ipynb` → `09-python-agent-framework.py`）。
- 一课含多个笔记本则各自生成、互不合并（如第 10 课的 `10-python-agent-framework.py` 与 `10-expense_claim-demo.py`）。
- 一律放在原笔记同级 `code_samples/` 目录。
- **只转 Python 笔记，忽略同课的 `.cs` / dotnet 版**（见 §9）。

---

## 3. 文件结构（自上而下模板）

```python
"""
Lesson XX - <章节标题> (纯 Python 版)

核心主题: 一句话概括 + 以"<场景>"串联:
  Demo 1: <场景/知识点>
  Demo 2: ...

对应原笔记本:
  - <xxx>.ipynb → Demo 1

API 变更说明（agent-framework >= 1.8.0）:
  - <已删除 API> → <替代>
已知限制 / 注意事项:
  - <如 gpt-4o 不把 base64 当图片等>
"""

import os, sys, asyncio
# 按需: base64 / json / time / typing(Annotated, ...) / collections.abc

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from agent_framework import tool, AgentResponseUpdate, WorkflowBuilder
from agent_framework.openai import OpenAIChatCompletionClient

if hasattr(sys.stdout, "reconfigure"):        # Windows GBK → UTF-8，避免 ✓/✗ 报错
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()
client = OpenAIChatCompletionClient(
    model=os.getenv("GITHUB_MODEL_ID"),
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url=os.getenv("GITHUB_ENDPOINT"),
)


# ============================================================
# Demo 1: <标题>  | 对应知识点: ...
# ============================================================
async def demo_1():
    print("=" * 60)
    print("Demo 1: <标题>")
    print("=" * 60)
    ...
    # 其余 Demo 2 / Demo 3 ... 按相同结构定义（开头打印 =*60 横幅）


async def main():
    await demo_1()
    await demo_2()


if __name__ == "__main__":
    asyncio.run(main())
```

---

## 4. 客户端与环境适配（关键）

### 4.1 模型接入：Azure Foundry → GitHub Models

| 原笔记（已废弃） | 纯 Python 版（现行） |
|---|---|
| `from agent_framework.foundry import FoundryChatClient` | `from agent_framework.openai import OpenAIChatCompletionClient` |
| `FoundryChatClient(project_endpoint=..., model=..., credential=DefaultAzureCredential())` | `OpenAIChatCompletionClient(model=GITHUB_MODEL_ID, api_key=GITHUB_TOKEN, base_url=GITHUB_ENDPOINT)` |
| `DefaultAzureCredential` / `AzureCliCredential` | 不再需要，token 从 `.env` 读取 |
| `AZURE_AI_PROJECT_ENDPOINT` / `AZURE_AI_MODEL_DEPLOYMENT_NAME` | `GITHUB_MODEL_ID` / `GITHUB_TOKEN` / `GITHUB_ENDPOINT` |

- 环境变量统一读这三个，**不要硬编码模型名**（除非是 Demo 内明确对比用的小模型，如 `gpt-4o-mini`，可硬编码并注明）。
- `.env` 由用户自备：`GITHUB_MODEL_ID=...`、`GITHUB_TOKEN=...`、`GITHUB_ENDPOINT=https://models.inference.ai.azure.com`。

### 4.2 Agent 创建：`create_agent`(await) → `as_agent`(无需 await)

```python
agent = client.as_agent(name=..., instructions=..., tools=[...])   # 无 await
```

### 4.3 已删除 API 的等价替代

| 原笔记 API | 替代方式 |
|---|---|
| `AzureAIProjectAgentProvider` | `OpenAIChatCompletionClient` + GitHub Models（见 4.1） |
| `HandoffBuilder`（动态路由） | `WorkflowBuilder` + `add_edge(condition=...)`（条件）或 `add_chain(...)`（顺序）模拟 |
| `RequestInfoExecutor` / 人在回路 | `@tool(approval_mode="always_require")` |
| 多模态图（经工具结果传视觉模型） | GitHub Models 免费版不支持 → 改传文本，或注释说明限制 |
| `ChatOptions(max_tokens=...)` | 保留，用于成本/Token 预算 Demo |

### 4.4 `WorkflowBuilder` 输出参数（修正措辞）

- `output_from` **不是必填**。它决定工作流最终返回哪些 Agent 的输出：
  - 多 Agent 需要汇总**所有**末端输出时：`output_from="all"`；
  - 只需**单个末端 Agent** 输出时：传该 agent（或 `[agent]`，如第 14 课条件工作流的 `output_from=[display_result]`）即可，**可不传**。
- 对照同目录既有 `.py` 取实际签名，勿盲加 `output_from="all"`。

### 4.5 进阶编排（详细写法以 §8 的 14 课范本为准）

- 顺序/并发/条件：`WorkflowBuilder` + `add_edge` / `add_fan_out_edges` / `add_chain`，条件用 `condition=<函数>`。
- 结构化输出：`AgentExecutor(client.as_agent(...), id=...)` 包装 + `AgentExecutorRequest` / `AgentExecutorResponse` / `Message` / `WorkflowContext`。
- 自定义执行器 `@executor(id="...")` + `await ctx.yield_output(...)`；中间件 `@function_middleware` 拦截工具调用。

---

## 5. 代码风格与约定

1. **全中文注释**；每块代码前用 `=====` 分隔框，标注「对应知识点 / 原笔记本 / 现行 API」。
2. **工具**：`@tool(approval_mode="never_require")`（需审批用 `"always_require"`）；参数 `Annotated[str, "说明"]`；函数内加 `print(f"  [tool] <name>({arg!r}) called...")`。
3. **Agent 指令**：英文原指令多行拼接，仅注释用中文。
4. **流式输出**：抽成 `stream_workflow` 辅助函数，按 Agent 分段打印，避免每处重复：
   ```python
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
                   print(f"\n{'=' * 50}\n[{author}]:\n{'=' * 50}")
                   last_author = author
               print(update.text, end="", flush=True)
       print()
   ```
5. **非流式取结果**：`result = await workflow.run(request); outputs = result.get_outputs()`，再 `outputs[0].text`。
   - ⚠️ **坑**：`workflow.run(..., stream=True)` 返回的是异步生成器，对其 `async for` 遍历后**不能再 `.get_outputs()`**；非流式必须 `await`（拿到结果对象）再调 `get_outputs()`。参考文件曾有 `events.get_outputs()` 误用流变量名的 bug，照抄会 `NameError`。
6. **Demo 函数**：每个 `async def demo_xxx():` 开头打印 `=`*60 横幅；`main()` 顺序 `await` 调用。
7. **UTF-8**：顶部保留 `sys.stdout.reconfigure(encoding="utf-8")`。
8. **数据模型**：`pydantic.BaseModel` + `Field(..., description=...)`；列表返回用 `list[X]`。
9. **资源路径**：读同目录文件（如 `receipt.jpg`）用 `os.path.join(os.path.dirname(__file__), "receipt.jpg")`，勿用相对路径。
10. **不引入新依赖**；可选依赖（如 `graphviz`）缺失时 `try/except ImportError` 优雅降级。

---

## 6. 知识点覆盖检查清单

改写后逐条核对原笔记 cell：

- [ ] Agent 创建与 `instructions`
- [ ] `@tool` 定义与调用（含 `approval_mode`）
- [ ] 单 Agent `agent.run(...)` 与多轮对话
- [ ] 多 Agent 编排：顺序 / 并发 / 条件路由
- [ ] 工作流 `WorkflowBuilder` + `add_edge` / `add_fan_out_edges`
- [ ] 流式 vs 非流式输出
- [ ] `pydantic` 结构化输出
- [ ] 错误处理 / 主备回退（如 09 课 fallback）
- [ ] 自反思 / 自评估（独立评估 Agent 打分）
- [ ] 人在回路（`approval_mode="always_require"`）
- [ ] 中间件（`@function_middleware`）
- [ ] 成本管理（`max_tokens` / 模型选择 / 缓存）
- [ ] 可观测性（计时、日志、`WorkflowViz`）
- [ ] 多模态/图片输入（若涉及，按 §7 限制或改文本）

---

## 7. 已知限制（写进文件头 docstring）

- GitHub Models 免费版**不支持在工具结果里传图片**：原笔记把图片当 base64 文本传给 `gpt-4o` 时，模型不会当图像解析（原笔记 Note 已明确）。本版忠实保留该行为，或改为「先 OCR 提取文本 → 传文本」并注释知识点不变。
- `gpt-4o` 多出现在「注意事项」举例，代码内模型由 `GITHUB_MODEL_ID` 指定，**不要硬编码 gpt-4/gpt-4o**。
- 大 base64 图片可能超上下文窗口；生产环境建议更小图片或更大上下文模型。

---

## 8. 运行前置与验证（兜底「可直接运行」）

**前置**：
- Python 3.12+；`pip install -r requirements.txt`（agent-framework >= 1.8.0）。
- 本目录准备好 `.env`（见 §4.1）。

**生成后必须验证**：
1. 语法校验：`python -m py_compile xxx.py`（确认无语法错误）。
2. **最好实跑一次**：`python xxx.py`，确认能连模型、各 Demo 正常输出、无运行时异常（如 `NameError`、缺少 `get_outputs` 等 API 误用）。
3. 对照 §6 清单逐条确认知识点都在。

---

## 9. 适用范围与例外

- **只转 Python / `.ipynb`**，忽略同课的 `.cs` / dotnet 笔记本。
- **非 `agent_framework` 的课**（如 11 MCP/A2A、13 memory、15 browser-use、18 security）用的不是 `FoundryChatClient`，而是 `mcp` / `a2a-sdk` / `browser-use` 等 SDK。此类笔记：
  - **保留原 SDK 与导入**，只把模型客户端改为 `OpenAIChatCompletionClient` + GitHub Models（若原笔记用 Azure/OpenAI，按 §4.1 映射）；
  - 把 Jupyter cell 改写成 `async def demo_*` + `asyncio.run(main())` 的纯脚本结构；
  - 知识点覆盖与风格约定（§5/§6）仍适用。
- **API 版本对照**：agent-framework 版本间 API 会变。生成前先对照同目录或 §10 参考样例里的**当前可用写法**校验签名（如 `as_agent`、`get_outputs`、`output_from`），不要凭记忆写；对不确定的 API 先 `python -c "import agent_framework; help(...)"` 或查已完成的 `.py`。

---

## 10. 参考样例（已完成，风格以它们为准）

- `08-multi-agent/code_samples/workflows-agent-framework/python/01...basic.py` — 顺序工作流 + 非流式取结果
- `09-metacognition/code_samples/09-python-agent-framework.py` — 主备工具回退 + 自反思 + 自评估
- `10-ai-agents-production/code_samples/10-python-agent-framework.py` — 可观测性/评估/成本 + 报销工作流
- `10-ai-agents-production/code_samples/10-expense_claim-demo.py` — 报销笔记 1:1 忠实转换
- `14-microsoft-agent-framework/code-samples/14-python-agent-framework.py` — 顺序/并发/条件/中间件/人在回路 全覆盖范本（⚠️ 注意其内部 `events.get_outputs()` 误用流变量名的 bug，照用前需改为 `result.get_outputs()`）
