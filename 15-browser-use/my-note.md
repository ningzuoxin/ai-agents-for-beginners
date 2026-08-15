# 第 15 章学习笔记：用 AI 驱动浏览器自动化（Browser-Use + Playwright + CDP）

> 本章目标：构建一个能自己打开 Airbnb、搜 Stockholm、用视觉模型读价格、结构化抽取并找出最便宜房源的浏览器自动化 agent。
> 核心一句话：**用 CDP 把 Playwright 和 Browser-Use 接到同一个 Chrome 上，Browser-Use（视觉 LLM 决策）做动态导航、Playwright 做精确控制，Pydantic 抽取结构化数据，Python 做业务逻辑。**

---

## 1. 核心概念

### CDP（Chrome DevTools Protocol）
- 程序化控制 Chromium 系浏览器（Chrome、Edge、Brave 等）的调试协议。能力：开页面、点击、截图、读 DOM、监听网络。不止 Chrome，所有 Chromium 内核都支持。

### Chrome
- 谷歌浏览器，基于 Chromium 内核。章节里 Playwright 连的是以 CDP 模式启动的 Chrome（默认端口 `9222`）。
- 拓展：Edge、Brave、Opera、Vivaldi 等也用 Chromium 内核，同样支持 CDP；Firefox（Gecko）、Safari（WebKit）不是 Chromium 内核，不支持 CDP。

### Playwright
- 跨语言浏览器自动化框架（Python/Node/.NET/Java），与 Selenium 同类。通过 CDP 控制浏览器，做"打开、点击、填表、截图"等确定性操作。它**不会自己看懂页面**，你写代码告诉它点哪个。GitHub: https://github.com/microsoft/playwright

### Browser-Use
- **开源第三方项目**（非微软），建立在 Playwright 之上的 **AI 代理层**。给自然语言任务 → 内部用 Playwright 操作浏览器 + 视觉 LLM 看页面决定下一步 → 循环至完成。
- 层次：任务 → Browser-Use（决策）→ Playwright（执行）→ CDP → Chromium。
- 作用/为什么需要：Playwright 只懂"你写代码点哪个"，面对动态页面、弹窗、布局变化就得手写脆弱的选择器；Browser-Use 用 LLM 看页面自己决定下一步，把自然语言任务变成自动执行的浏览器操作。一句话区别：仅用 Playwright 需事先知道网页结构、手写"点哪个选择器"的代码；用 Browser-Use 只需自然语言描述任务，由 LLM 看页面自己决定怎么操作，无需了解具体 DOM 结构。GitHub: https://github.com/browser-use/browser-use

### 视觉模型
- 多模态大模型，可接受截图理解页面。原理：把截图当输入 token 与文本一起送进模型（非传统 OCR 逐字识别，而是端到端"看懂"布局与语义）；章节用 `browser_use.ChatAzureOpenAI`（Azure 上的视觉模型，如 gpt-5-mini/gpt-4o），关键是模型要支持视觉。常用视觉模型：GPT-4o / GPT-4o-mini（OpenAI）、Azure OpenAI 同款部署、Claude 3.5+（Anthropic）、Gemini 1.5+（Google）、Qwen-VL（阿里）。
- 拓展：视觉模型怎么"看懂"截图：图像先被切成小块（patch），经视觉编码器（如 ViT）转成一串图像 token，与文本 token 拼在一起送进同一 Transformer；模型在统一空间里对齐"图块"和"文字"，从而能回答"截图里最便宜的房源是哪个""这个按钮上写的什么"等问题，而非只做字符识别。

### Computer Use Agent（CUA）
- 让 AI 像人一样操作浏览器/电脑的代理。本章 Airbnb agent 就是小型 CUA。CUA 是"类别/思路"，Browser-Use 是其中一种实现路线。
- 两种"看页面"路线：
  - **Browser-Use 路线**（本章）：DOM + 截图，AI 语义化"选元素点"。
  - **CUA 像素路线**（如 OpenAI Computer Use）：AI 只看截图，输出鼠标/键盘坐标操作。

### Agent vs Actor
- 动态布局 / 找元素 / 复杂流程 → 用 **Agent**（灵活但慢）。
- 已知结构 / 需 timing 控制 / 高精度 → 用 **Actor**（Playwright 直接控制，快而准）。

### 安全护栏
浏览器 agent 跑在真实网站上，边界比纯 API 脚本更严：
- 限定浏览域/沙箱；观察与动作分离（搜索/读取可自主，**提交表单/下单/改设置需用户显式批准**）。
- 密钥不入 prompt/日志，认证交用户；页面内容视为不可信输入（防 prompt 注入）。
- 风险步骤前用代码确定性校验（URL、价格、动作摘要）；设预算与停止条件（动作数/时间/标签上限）。

---

## 2. 实践流程

### 架构连接方式
章节用的是 **CDP 共享会话**模式，三件套都连到同一个 Chrome：
1. 以 CDP（远程调试）启动 Chrome → 默认 `http://localhost:9222`；
2. Playwright 用 `chromium.connect_over_cdp(cdp_url)` 连上；
3. Browser-Use 用 `Browser(cdp_url=cdp_url, keep_alive=True)` 连上同一个 Chrome。
> `keep_alive=True` 很重要：默认 Agent 跑完浏览器会关掉，设为 True 才能继续拿页面做抽取。

为什么连同一个 Chrome：Browser-Use 用自然语言导航到目标页面 → 把同一个 Chrome 里的页面对象交还 Playwright/代码做精确抽取与逻辑处理 → 三者共连一个可见 Chrome，全程每步操作肉眼可见。

### 执行顺序（main 里的真实步骤）
1. `start_chrome_with_cdp(port=9222)` 启动带 CDP 的 Chrome；
2. Playwright `connect_over_cdp` 连上（便于自定义 Playwright 操作）；
3. 创建 `AirbnbSearchAgent(llm=llm, cdp_url=cdp_url)`；
4. `agent.search_stockholm()`：
   - **Step 1（Agent 导航）**：`Agent(task=..., use_vision=True)` 打开 Airbnb、关弹窗、搜 "Stockholm, Sweden"；
   - **Step 2（Vision 抽取）**：拿当前 page，用 `page.extract_content(prompt=..., structured_output=SearchResult, llm=llm)` 让视觉模型读价格并填进 Pydantic 模型；
5. Python 比较 `listings` 找最便宜、算均价与价格区间。

### 关键代码模式
```python
# 1) Browser-Use 的 LLM 用自家的 ChatAzureOpenAI，自动读环境变量
from browser_use import Agent, Browser, ChatAzureOpenAI
llm = ChatAzureOpenAI(model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"))

# 2) 用 CDP 接到同一个 Chrome（keep_alive 保活）
browser = Browser(cdp_url="http://localhost:9222", keep_alive=True)

# 3) Agent 用视觉导航
agent = Agent(task="...", llm=llm, browser=browser, use_vision=True)
await agent.run()

# 4) 视觉抽取结构化数据
page = (await browser.get_pages())[0]
result = await page.extract_content(
    prompt=extraction_prompt,
    structured_output=SearchResult,   # Pydantic 模型
    llm=llm,
)
```

### Pydantic 结构化抽取
- 定义 `AirbnbListing`（title / price_per_night / currency / rating / url）和 `SearchResult`（location / total_listings_found / listings / cheapest_listing / average_price / price_range）。
- 价值：让视觉模型输出**类型安全、可校验**的 Python 对象，而不是自由文本，便于后续业务逻辑（比较/排序）。
- 抽取 prompt 要点：**只要看得到的房源**、只取数字价格、顺带算最低价/均价/区间、带上可点击 URL。

---

## 3. 踩坑 & 易错点
- **模型要支持视觉**：`use_vision=True` 才启用截图分析；用的是 Azure 上的视觉部署，不是普通文本模型。
- **`keep_alive=True`**：不设置，Agent 跑完 Chrome 直接关，后续 `get_pages()` 拿不到页面。
- **抽取 prompt 要约束**：明确"只要房源卡片、只要看得到的价格、排除 Experiences"，否则模型容易混入无关内容或返回 None。
- **异步**：整个流程是 `async`，入口 `asyncio.run(main())`，所有 `await` 包在 `async def` 里。
- **温度设低**（如 0.3）：自动化要稳定可复现，别用高温度聊天设定。

---

## 4. 选型心法（什么时候用这套）
- 有稳定 API 的同款任务 → 优先用 API（更快、好测、好安全）。
- 任务依赖"页面上看到的东西"、站点没 API、或页面常变导致选择器易碎 → 用浏览器 agent。
- 混合策略：先用 Agent 探索/应对动态 UI，结构稳定后切回 Playwright 直接控制，灵活与精确兼得。

## 5. 现实映射
本章的 agent 是微软 **Project Opal**（Microsoft 365 Copilot 中的企业级 CUA）的迷你本地版——在隔离 Cloud PC 上异步后台运行，体现前序章节的人在回路、可信安全、规划/元认知、可复用 Skills 等概念。
