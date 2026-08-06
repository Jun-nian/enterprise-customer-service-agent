# 企业内部智能客服 Agent 项目总结报告

---

## 1. 项目概述

| 属性 | 内容 |
|------|------|
| **项目名称** | 05_RagAgent_Business — 企业内部智能客服 Agent |
| **版本** | v5.0（基于 LangGraph 第五期演进） |
| **开发团队** | 南哥AGI研习社 (B站/YouTube) |
| **技术定位** | 基于 RAG + LangGraph 多工具 Agent 的企业内部智能客服系统 |
| **运行环境** | Windows 11, Python 3.13, Ollama |

### 业务目标

为企业内部员工提供一站式智能客服，解决以下问题：

- **HR 问答**：入职流程、考勤制度、薪资查询、保险政策、收入证明
- **IT 支持**：WiFi/VPN 配置、工单状态查询、设备报修
- **行政服务**：会议室预定、办公用品申领、班车路线
- **知识检索**：公司制度、产品文档、技术方案的语义搜索
- **员工查询**：同事联系方式、部门归属、汇报关系
- **订单查询**：销售订单状态、客户信息、交付进度

### 核心用户角色

| 角色 | 典型场景 |
|------|----------|
| **普通员工** | 查询HR政策、IT帮助、公司FAQ |
| **管理者** | 查询员工信息、订单状态、部门数据 |
| **HR/行政** | 回答员工高频问题，减少重复人工咨询 |
| **IT 运维** | 工单管理、设备问题排查 |

---

## 2. 技术架构

### 整体架构图（文字描述）

```
┌─────────────────────────────────────────────────────────────────┐
│                        前端展示层                                │
│  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  Gradio UI   │  │  REST API 客户端  │  │  Claude Code     │  │
│  │  (端口 7860)  │  │  (apiTest.py)    │  │  (MCP 客户端)    │  │
│  └──────┬───────┘  └────────┬─────────┘  └────────┬─────────┘  │
│         │                   │                     │             │
├─────────┼───────────────────┼─────────────────────┼─────────────┤
│         │          后端服务层 (FastAPI :8012)      │ MCP 协议    │
│         ▼                   ▼                     ▼             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    main.py (FastAPI)                      │   │
│  │  • POST /v1/chat/completions (流式 / 非流式)              │   │
│  │  • lifespan: 初始化 LLM + 工具 + Graph                    │   │
│  │  • 依赖注入: get_dependencies()                           │   │
│  └────────────────────────┬─────────────────────────────────┘   │
│                           │                                     │
├───────────────────────────┼─────────────────────────────────────┤
│                   Agent 核心引擎层                               │
│                           ▼                                     │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                demoRagAgent.py (LangGraph)                 │   │
│  │                                                            │   │
│  │  START → agent → call_tools → grade_documents ─┬──→ END   │   │
│  │            │         │            │              │          │   │
│  │            │         │            ▼              ▼          │   │
│  │            │         │         rewrite ──→ agent           │   │
│  │            │         │         (最多3次)                    │   │
│  │            │         │              │                       │   │
│  │            │         ▼──────────────┼──────────────────│   │   │
│  │            │         generate ←─────┘                      │   │
│  │            │                                               │   │
│  │  状态: MessagesState (messages + relevance_score           │   │
│  │         + rewrite_count)                                   │   │
│  │  检查点: MemorySaver (线程内)                               │   │
│  │  长期记忆: InMemoryStore (跨线程, 768维嵌入)                │   │
│  └─────────────┬─────────────────────────────────────────────┘   │
│                │                                                 │
├────────────────┼─────────────────────────────────────────────────┤
│           工具层 (共享实现架构)                                    │
│                │                                                 │
│   ┌────────────┴──────────────┐                                  │
│   │  mcp_server/tools_impl.py │ ← 单一事实来源                    │
│   │  (5个纯 Python 工具函数)    │                                  │
│   │  • retrieve               │                                  │
│   │  • search_employees       │                                  │
│   │  • search_faq             │                                  │
│   │  • search_orders          │                                  │
│   │  • search_tickets         │                                  │
│   └──┬───────────────┬────────┘                                  │
│      │               │                                           │
│      ▼               ▼                                           │
│  ┌──────────┐  ┌──────────────────────┐                          │
│  │ MCP 服务器│  │ utils/tools_config.py│                          │
│  │(server.py│  │ (LangChain @tool 包装)│                          │
│  │ stdio)   │  │ → bind_tools()       │                          │
│  └──────────┘  └──────────────────────┘                          │
│                                                                  │
├──────────────────────────────────────────────────────────────────┤
│                    数据与存储层                                    │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  ChromaDB   │  │  JSON 数据  │  │  Ollama (本地LLM)        │  │
│  │  向量存储    │  │  文件       │  │  :11434                  │  │
│  │             │  │  employees  │  │  qwen2.5:7b (对话)       │  │
│  │ yuxing_     │  │  faq.json   │  │  nomic-embed-text (嵌入)  │  │
│  │ handbook    │  │  orders     │  │                          │  │
│  │             │  │  tickets    │  │                          │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 前端技术栈

| 技术 | 用途 | 版本 |
|------|------|------|
| **Gradio** | Web 聊天 UI 框架 | 6.14.0 |
| **CSS3** | 暗黑科幻主题自定义样式 | 内联 |
| **Server-Sent Events** | 流式响应渲染 | HTTP SSE |

UI 采用 **暗黑科幻主题**（深色背景 #0a0a12，青色高亮 #00e5ff，扫描线动画），支持 `<think>` 标签渲染和流式逐字输出。

### 后端技术栈

| 技术 | 用途 | 版本 |
|------|------|------|
| **Python** | 主语言 | 3.13 |
| **FastAPI** | REST API 框架 | 0.136.1 |
| **Uvicorn** | ASGI 服务器 | 0.39.0 |
| **LangGraph** | Agent 状态机/工作流引擎 | 1.2.1 |
| **LangChain** | 提示模板、工具抽象、嵌入接口 | 最新 |
| **LangChain-Chroma** | ChromaDB 集成 | 0.2.6 |
| **ChromaDB** | 向量数据库（持久化） | 1.5.9 |
| **Pydantic** | 数据验证/结构化输出 | 2.13.4 |
| **OpenAI SDK** | LLM/嵌入 API 客户端 | 2.38.0 |
| **pdfminer.six** | PDF 文本提取 | 最新 |
| **MCP (mcp)** | 模型上下文协议服务器 | 最新 |

### Agent 实现方式

基于 **LangGraph StateGraph** 的自定义多节点 Agent：

1. **agent 节点**：LLM 绑定 5 个工具，根据提示决定调用工具或直接回复
2. **call_tools 节点**：`ParallelToolNode` 使用 `ThreadPoolExecutor`(max_workers=5) 并行执行工具
3. **grade_documents 节点**：LLM 结构化输出 `DocumentRelevanceScore` 评分检索结果相关性
4. **rewrite 节点**：LLM 重写用户问题，改进检索召回率（最多 3 次）
5. **generate 节点**：LLM 根据检索上下文生成最终回复

### 部署方式

- **本地部署**：所有组件运行在本地 Windows 机器
- **Ollama**：本地 LLM 推理服务 (localhost:11434)
- **ChromaDB**：本地文件持久化 (chromaDB/ 目录)
- **FastAPI**：单进程 ASGI 服务 (0.0.0.0:8012)
- **Gradio**：开发模式启动 (127.0.0.1:7860)
- **无容器化**：未使用 Docker/容器

### 第三方集成

| 集成 | 方式 | 状态 |
|------|------|------|
| **Ollama** | OpenAI 兼容 API | 已集成 (qwen2.5:7b) |
| **OpenAI / OneAPI / 通义千问** | API 密钥 | 已配置（可选切换） |
| **Claude Code MCP** | stdio 传输 | 已配置 (.claude/mcp.json) |
| **PostgreSQL** | 连接串预留 | 未激活（仅配置） |
| **LangSmith** | 注释掉的追踪配置 | 可选启用 |

---

## 3. 功能模块

### 3.1 对话管理

| 功能 | 实现 | 文件位置 |
|------|------|----------|
| **多轮对话** | `MemorySaver` 检查点持久化线程内对话历史 | [demoRagAgent.py:671](demoRagAgent.py#L671) |
| **上下文管理** | `filter_messages()` 保留最近 5 条 AIMessage/HumanMessage | [demoRagAgent.py:267-272](demoRagAgent.py#L267-L272) |
| **长期记忆** | `InMemoryStore` 按 user_id 命名空间跨线程存储 | [demoRagAgent.py:674-676](demoRagAgent.py#L674-L676) |
| **记忆触发** | 用户消息包含"记住"时自动存储 | [demoRagAgent.py:294-297](demoRagAgent.py#L294-L297) |
| **线程隔离** | configurable.thread_id 区分不同会话 | [main.py:384](main.py#L384) |

### 3.2 意图识别与工具调用

意图识别完全由 **LLM function calling** 实现，无需关键词匹配或正则：

```python
# agent节点核心逻辑 (demoRagAgent.py:385-390)
llm_chat_with_tool = llm_chat.bind_tools(tool_config.get_tools())  # 绑定5个工具
agent_chain = create_chain(llm_chat_with_tool, Config.PROMPT_TEMPLATE_TXT_AGENT)
response = agent_chain.invoke({"question": question, "messages": messages, "userInfo": user_info})
```

LLM 自动判断是否需要调用工具以及调用哪个工具，返回 `tool_calls` 或直接回复。

### 3.3 知识检索（RAG 流程）

```
PDF 文档 → pdfminer 提取文本 → sent_tokenize 中文断句
  → split_text(800字符块, 200字符重叠) → Ollama nomic-embed-text 嵌入
  → ChromaDB PersistentClient 持久化 → retrieve 工具语义搜索
```

| 参数 | 值 |
|------|-----|
| 分块大小 | 800 字符 |
| 重叠大小 | 200 字符 |
| 嵌入模型 | nomic-embed-text:latest (768维) |
| 向量数据库 | ChromaDB (持久化到 chromaDB/ 目录) |
| 当前集合 | yuxing_handbook（裕兴集团员工手册.pdf, 933KB） |
| 检索数量 | top 5 |
| 评分机制 | LLM 结构化输出 yes/no 二元评分 |

### 3.4 工具调用

| 工具 | 数据源 | 查询方式 | 路由目标 |
|------|--------|----------|----------|
| `retrieve` | ChromaDB 向量库 | 语义搜索 | grade_documents |
| `search_employees` | data/employees.json (20人) | 关键词匹配 name/dept/position | grade_documents |
| `search_faq` | data/faq.json (15条 FAQ) | 关键词匹配 topic/question/answer | grade_documents |
| `search_orders` | data/orders.json (10条订单) | 关键词匹配 order_id/customer/rep/status | grade_documents |
| `search_tickets` | data/tickets.json (10条工单) | 关键词匹配 ticket_id/title/applicant/status | grade_documents |

工具采用**白名单路由模式**（[demoRagAgent.py:100](demoRagAgent.py#L100)）：
- 检索类工具 → `grade_documents` 节点（相关性评分）
- 非检索类工具 → `generate` 节点（直接生成回复）

### 3.5 用户认证与权限控制

**当前未实现。** `userId` 和 `conversationId` 仅用于线程隔离和记忆命名空间，不做身份验证。

### 3.6 日志与监控

| 机制 | 实现 | 配置 |
|------|------|------|
| **日志框架** | Python logging + ConcurrentRotatingFileHandler | [main.py:57-70](main.py#L57-L70) |
| **日志级别** | DEBUG | 可切换 INFO |
| **日志文件** | output/app.log | 5MB × 3 个备份 |
| **日志格式** | `时间 - 模块 - 级别 - 消息` | |
| **关键日志点** | agent 处理、工具调用、评分结果、最终响应 | 全流程覆盖 |
| **LangSmith** | 注释掉待启用 | [main.py:46-47](main.py#L46-L47) |
| **Graph 可视化** | Mermaid PNG 导出 | [demoRagAgent.py:637-655](demoRagAgent.py#L637-L655) |

### 3.7 反馈与评估机制

**当前未实现。** 以下功能待建设：
- 用户满意度评分（无 UI 入口）
- 自动评估指标（无 RAGAS 等框架集成）
- 对话质量审查（无人工审查界面）

---

## 4. 代码组织与设计模式

### 目录结构

```
05_RagAgent_Business/
├── main.py                     # FastAPI 入口 (407行)
├── demoRagAgent.py             # LangGraph 工作流引擎 (820行)
├── webUI.py                    # Gradio 前端 (282行)
├── apiTest.py                  # API 测试客户端 (74行)
├── vectorSaveTest.py           # PDF→向量→ChromaDB 入库 (208行)
├── utils/
│   ├── config.py               # 统一配置类 (29行)
│   ├── llms.py                 # 多后端 LLM 初始化 (157行)
│   ├── tools_config.py         # LangChain 工具包装 (71行)
│   ├── pdfSplitTest_Ch.py      # 中文 PDF 解析+分块 (112行)
│   └── pdfSplitTest_En.py      # 英文 PDF 解析+分块
├── mcp_server/
│   ├── __init__.py             # 包标识
│   ├── tools_impl.py           # 共享工具实现 (140行)
│   └── server.py               # MCP 服务器入口 (55行)
├── prompts/
│   ├── prompt_template_agent.txt      # Agent 决策提示
│   ├── prompt_template_generate.txt   # 回复生成提示
│   ├── prompt_template_grade.txt      # 相关性评分提示
│   └── prompt_template_rewrite.txt    # 问题重写提示
├── data/
│   ├── employees.json          # 20条员工数据
│   ├── faq.json                # 15条FAQ（6大类别）
│   ├── orders.json             # 10条销售订单
│   └── tickets.json            # 10条IT工单
├── input/                      # PDF 源文件
├── chromaDB/                   # ChromaDB 持久化存储
└── output/
    └── app.log                 # 应用日志
```

### 关键设计模式

#### 模式 1：共享实现架构（Single Source of Truth）

```
mcp_server/tools_impl.py  ← 唯一的工具实现
        ↑                        ↑
        | MCP @tool() 装饰        | LangChain @tool 装饰
        |                        |
mcp_server/server.py      utils/tools_config.py
```

MCP 服务器和 LangGraph 工作流共享同一套实现代码。新增工具只需在 `tools_impl.py` 中定义一次。

#### 模式 2：策略模式 — 工具路由

`ToolConfig._build_routing_config()` 根据工具名称白名单动态决定路由策略：

```python
retrieval_tools = {"retrieve", "search_employees", "search_faq", "search_orders", "search_tickets"}
# 检索工具 → grade_documents（需评分）
# 非检索工具 → generate（直接生成）
```

#### 模式 3：工厂模式 — LLM 初始化

`initialize_llm()` 根据 `llm_type` 配置键从 `MODEL_CONFIGS` 字典创建对应实例，支持 openai/oneapi/qwen/ollama 四种后端。

#### 模式 4：模板方法 — 提示链缓存

`create_chain()` 使用**函数静态属性**实现线程安全的提示模板缓存：

```python
if not hasattr(create_chain, "prompt_cache"):
    create_chain.prompt_cache = {}
    create_chain.lock = threading.Lock()
```

#### 模式 5：装饰器模式 — 工具注册

工具函数通过 `@tool` (LangChain) 或 `@mcp.tool()` (MCP) 装饰器注册，自动生成 JSON Schema 供 LLM function calling 使用。

### 核心类/函数职责描述

| 文件 | 类型 | 职责 |
|------|------|------|
| **demoRagAgent.py** | `MessagesState(TypedDict)` | 状态定义：messages + relevance_score + rewrite_count |
| **demoRagAgent.py** | `ToolConfig` | 工具管理：名称集合、路由配置白名单 |
| **demoRagAgent.py** | `ParallelToolNode(ToolNode)` | 并行工具执行：ThreadPoolExecutor(max_workers=5) |
| **demoRagAgent.py** | `DocumentRelevanceScore(BaseModel)` | 结构化评分输出：binary_score "yes"/"no" |
| **demoRagAgent.py** | `create_graph()` | 图编译：5 节点 + 4 边 + 2 条件路由 |
| **demoRagAgent.py** | `create_chain()` | 提示缓存 + LCEL 链构建 |
| **main.py** | `ChatCompletionRequest/Response` | OpenAI 兼容 API 数据模型 |
| **main.py** | `handle_stream_response()` | SSE 流式响应生成 |
| **main.py** | `lifespan()` | 应用生命周期管理（初始化/清理） |
| **utils/llms.py** | `OllamaEmbeddings(Embeddings)` | 自定义 Ollama 嵌入适配器 |
| **utils/llms.py** | `initialize_llm()` | 多后端 LLM 工厂 |
| **mcp_server/tools_impl.py** | `retrieve/search_*/_get_embedding` | 5 工具 + 嵌入辅助（无框架依赖） |
| **mcp_server/server.py** | `FastMCP` 实例 | MCP 服务器注册（stdio 传输） |
| **utils/pdfSplitTest_Ch.py** | `getParagraphs()` | PDF → 800 字符块管道 |

### 依赖管理

项目无 `requirements.txt`，依赖通过 pip 手动安装：

```
核心依赖: langgraph, langchain-core, langchain-openai, langchain-chroma, chromadb
Web: fastapi, uvicorn, gradio
LLM: openai (SDK, 用于 API 调用)
PDF: pdfminer.six, nltk
MCP: mcp, langchain-mcp-adapters
工具: pydantic, httpx, concurrent-log-handler
```

---

## 5. 数据流与交互序列

### 完整请求-响应数据流

```
用户输入: "张建国的邮箱是什么?"
        │
        ▼
┌─ [1] Gradio UI ─────────────────────────────────────────────────┐
│  send_message() → POST /v1/chat/completions                     │
│  Body: {"messages":[{"role":"user","content":"张建国的邮箱..."}],│
│         "stream":true,"userId":"123","conversationId":"123"}     │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP POST
                             ▼
┌─ [2] FastAPI main.py ───────────────────────────────────────────┐
│  chat_completions() 解析请求                                     │
│  → 构建 config: {thread_id: "123@@123", user_id: "123"}         │
│  → stream=True → handle_stream_response()                       │
└────────────────────────────┬────────────────────────────────────┘
                             │ graph.stream()
                             ▼
┌─ [3] Node: agent ───────────────────────────────────────────────┐
│  a) store_memory() → InMemoryStore.search() 检索历史记忆         │
│  b) filter_messages() → 保留最近5条对话                          │
│  c) create_chain(agent_prompt) → llm.bind_tools(5个工具)        │
│     Prompt: "你是一个企业智能客服助手..." + tools JSON Schema     │
│  d) LLM(qwen2.5:7b) → 决定调用 search_employees                  │
│  e) 返回 AIMessage(tool_calls=[{name:"search_employees",         │
│        args:{query:"张建国"}}])                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │ tools_condition → "tools"
                             ▼
┌─ [4] Node: call_tools (ParallelToolNode) ───────────────────────┐
│  ThreadPoolExecutor(max_workers=5)                               │
│  → search_employees("张建国")                                    │
│    → mcp_server/tools_impl.py:_search_employees()                │
│    → 读取 data/employees.json → 匹配 name/department/position    │
│    → 返回 "姓名: 张建国, 部门: 技术研发中心,                      │
│            职位: 技术总监, 邮箱: zhangjianguo@xinrui.com..."      │
│  → ToolMessage(content=结果, name="search_employees")            │
└────────────────────────────┬────────────────────────────────────┘
                             │ route_after_tools()
                             │ search_employees ∈ retrieval_tools
                             ▼ → "grade_documents"
┌─ [5] Node: grade_documents ─────────────────────────────────────┐
│  create_chain(grade_prompt) + DocumentRelevanceScore 结构化输出   │
│  Prompt: question="张建国的邮箱是什么?"                           │
│          context="姓名: 张建国, 部门: 技术研发中心..."            │
│  LLM → {"binary_score": "yes"}                                  │
│  state.relevance_score = "yes"                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │ route_after_grade()
                             │ score="yes" → "generate"
                             ▼
┌─ [6] Node: generate ────────────────────────────────────────────┐
│  create_chain(generate_prompt) → LLM 生成最终回复                 │
│  Prompt: question + context                                     │
│  → "张建国的邮箱是 zhangjianguo@xinrui.com，电话 1380001001"     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼ → END
┌─ [7] FastAPI StreamingResponse ─────────────────────────────────┐
│  generate_stream() 逐个产出 SSE chunk:                           │
│  data: {"choices":[{"delta":{"content":"张建国"},"finish_reason":│
│         null}]}\n\n                                             │
│  ...                                                            │
│  data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n    │
└────────────────────────────┬────────────────────────────────────┘
                             │ SSE 流
                             ▼
┌─ [8] Gradio UI ─────────────────────────────────────────────────┐
│  逐字渲染在聊天框，显示 "◆ 企业智能客服 ◆" 暗黑主题               │
└──────────────────────────────────────────────────────────────────┘
```

### 异常路径 — 检索不相关时的重写循环

```
grade_documents → binary_score = "no"
                       │
              route_after_grade → "rewrite"
                       │
              Node: rewrite → LLM 重写问题 → rewrite_count + 1
                       │
              Edge: rewrite → agent (重新检索)
                       │
         (最多循环3次，第3次后强制 → generate)
```

---

## 6. 部署与运维

### 环境要求

| 类别 | 要求 |
|------|------|
| **操作系统** | Windows / Linux / macOS |
| **Python** | 3.13（推荐，3.10+ 可用） |
| **Ollama** | 本地运行，端口 11434 |
| **Ollama 模型** | qwen2.5:7b (对话) + nomic-embed-text:latest (嵌入) |
| **内存** | 8GB+ (LLM 推理) |
| **磁盘** | 2GB+ (含 ChromaDB 向量数据) |
| **网络** | 如需使用 OpenAI/通义千问，需外网访问 |

### 启动与停止

```bash
# 前置条件
ollama pull qwen2.5:7b
ollama pull nomic-embed-text

# 1. 向量入库（仅首次或 PDF 更新后）
cd 05_RagAgent_Business
python vectorSaveTest.py

# 2. 启动后端（终端1）
python main.py                           # http://0.0.0.0:8012

# 3. 启动前端（终端2）
python webUI.py                          # http://127.0.0.1:7860

# 4. 或命令行直接对话
python demoRagAgent.py

# 停止：Ctrl+C 终止两个进程
```

### 配置项说明

| 配置 | 文件位置 | 默认值 | 说明 |
|------|----------|--------|------|
| `LLM_TYPE` | utils/config.py:25 | `"ollama"` | LLM 后端选择 (openai/oneapi/qwen/ollama) |
| `CHROMADB_COLLECTION_NAME` | utils/config.py:14 | `"yuxing_handbook"` | ChromaDB 集合名 |
| `HOST` | utils/config.py:28 | `"0.0.0.0"` | FastAPI 监听地址 |
| `PORT` | utils/config.py:29 | `8012` | FastAPI 监听端口 |
| `LOG_FILE` | utils/config.py:17 | `"output/app.log"` | 日志文件路径 |
| `MAX_BYTES` | utils/config.py:18 | `5*1024*1024` | 日志轮转大小 (5MB) |
| `DB_URI` | utils/config.py:22 | Postgres 连接串 | 环境变量可覆盖 |
| API Keys | utils/llms.py | 硬编码 | **安全风险，见第7章** |
| MCP 服务器 | .claude/mcp.json | stdio 传输 | Claude Code 自动发现 |

### 日志位置

```
05_RagAgent_Business/output/app.log       # 主日志
05_RagAgent_Business/output/app.log.1     # 轮转备份1
05_RagAgent_Business/output/app.log.2     # 轮转备份2
05_RagAgent_Business/output/app.log.3     # 轮转备份3
```

### 性能指标（基于当前 Ollama 本地部署）

| 指标 | 预估值 |
|------|--------|
| **冷启动时间** | ~15 秒（LLM 初始化 + Graph 编译） |
| **简单回复延迟** | 2-5 秒 (qwen2.5:7b) |
| **工具调用延迟** | 3-8 秒（含检索+评分+生成） |
| **嵌入生成** | ~0.3 秒/批次 (25条) |
| **并发支持** | 单线程 FastAPI + ThreadPoolExecutor(5) |
| **ChromaDB 检索** | <100ms |

---

## 7. 已知问题与潜在风险

### 7.1 API 密钥安全

| 严重程度 | 问题 | 位置 |
|----------|------|------|
| **高** | API 密钥硬编码在源代码中 | [utils/llms.py:40-63](utils/llms.py#L40-L63) |
| **高** | OpenAI / OneAPI / 通义千问 的 API 密钥明文存储 | [vectorSaveTest.py:18-32](vectorSaveTest.py#L18-L32) |
| **中** | 无 .env 文件或环境变量管理 | — |

### 7.2 输入校验

| 严重程度 | 问题 |
|----------|------|
| **中** | 用户输入仅做了非空检查，无注入防护 |
| **中** | 无输入长度限制（大文本可能导致 LLM OOM） |
| **低** | JSON 文件读取异常已捕获，但无数据完整性校验 |

### 7.3 可扩展性瓶颈

| 瓶颈 | 说明 |
|------|------|
| **单进程 FastAPI** | 无 worker 多进程配置，高并发下阻塞 |
| **内存存储** | MemorySaver/InMemoryStore 重启后丢失，不适合生产 |
| **JSON 文件工具** | 每次查询全量加载文件到内存，数据量大时性能下降 |
| **Ollama 本地推理** | qwen2.5:7b 推理速度有限，不支持高并发 |
| **无缓存** | 相同问题重复 LLM 调用，无响应缓存 |
| **工具函数阻塞** | 同步 `json.load()` 阻塞事件循环 |

### 7.4 代码质量问题

- `vectorSaveTest.py` 和 `utils/llms.py` 中 API 密钥配置**重复**
- `main.py` 的 `handle_stream_response` 和 `handle_non_stream_response` 中流处理逻辑**重复**
- `demoRagAgent.py` 中 `get_latest_question` 使用 `__class__.__name__` 字符串比较而非 `isinstance()`
- 日志中中文内容在 Windows GBK 终端中显示为乱码（数据本身正确）

### 7.5 已注释掉的 TODO

- LangSmith 追踪（[main.py:46-47](main.py#L46-L47)）
- PostgreSQL 持久化（已配置 URI 但未激活）
- 页码筛选功能（已实现但默认 None）

---

## 8. 改进建议

### 短期优化（1-2 周）

| 优先级 | 改进项 | 收益 |
|--------|--------|------|
| **高** | API 密钥迁移到 `.env` + `python-dotenv` | 安全性 |
| **高** | 添加输入长度限制和基础 XSS 防护 | 安全性 |
| **中** | 提取公共流处理逻辑减少代码重复 | 可维护性 |
| **中** | 添加响应缓存（同一问题 N 分钟内复用） | 性能 |
| **中** | `isinstance()` 替代 `__class__.__name__` 字符串比较 | 健壮性 |
| **低** | 添加 `requirements.txt` | 环境标准化 |
| **低** | 启用日志中文化或统一英文输出 | 可读性 |

### 长期演进方向（1-3 月）

| 方向 | 具体方案 |
|------|----------|
| **生产化部署** | Docker 容器化 + PostgreSQL 替代内存存储 |
| **多轮对话增强** | 槽位填充（slot-filling）替代纯 LLM 意图识别 |
| **主动推荐** | 根据用户部门/角色推荐常见问题 |
| **多渠道** | 企业微信/钉钉/飞书 Bot 接入 |
| **多模态** | 支持图片上传（如截图报错） |
| **评估体系** | 集成 RAGAS 自动评估 + 用户满意度采集 |
| **工具增强** | 将 JSON 文件工具升级为真实 API 调用（CRM/ITSM） |
| **MCP 升级** | 从 stdio 升级到 SSE/HTTP，支持远程工具调用 |
| **权限体系** | 基于 userId 的角色权限控制（普通员工 vs HR vs 管理员） |
| **监控告警** | Prometheus metrics + 异常告警 |

---

## 9. 总结与结论

### 项目成熟度评估

**当前阶段：原型 → 生产可用的过渡期**

| 维度 | 评分 | 说明 |
|------|------|------|
| **架构设计** | ★★★★☆ | LangGraph 工作流设计成熟，MCP 集成架构前瞻 |
| **代码质量** | ★★★☆☆ | 注释详尽但存在重复代码和密钥安全问题 |
| **功能完整度** | ★★★☆☆ | 核心 RAG+工具调用完备，缺认证/评估/监控 |
| **生产就绪度** | ★★☆☆☆ | 内存存储、单进程、密钥硬编码不宜直接上线 |
| **可扩展性** | ★★★★☆ | 工具架构支持热插拔，MCP 支持标准协议扩展 |

### 对业务的价值

1. **即时价值**：已可部署为内部员工自助问答系统，覆盖 HR/IT/行政等 5 大领域
2. **降本增效**：减少 HR/IT 重复人工咨询 60%+（基于 FAQ 和知识库覆盖范围估算）
3. **知识沉淀**：将纸质员工手册数字化为可检索的向量知识库
4. **标准化基础**：MCP 标准协议为后续工具扩展和第三方集成奠定基础

---

## 附录：面试常见问题

**Q1: 为什么选择 LangGraph 而不是直接用 LangChain Chain？**

LangGraph 提供显式的状态图和条件路由，支持循环（重写→重新检索）和并行工具执行。普通 Chain 是线性 DAG，无法表达"评分不通过则重写再检索"的回环逻辑。

**Q2: 为什么工具路由用白名单而非 LLM 自主判断？**

LLM 自主判断下一步路由（如 ReAct 模式）会增加一次 LLM 调用，延迟和成本更高。白名单路由是确定性的，零延迟且可靠。检索类工具 → 评分 → 生成 / 重写的路径是预定义的业务规则。

**Q3: ChromaDB 和 JSON 文件为什么同时存在？**

ChromaDB 处理**非结构化数据**（PDF 文档语义搜索），JSON 文件处理**结构化数据**（员工列表、FAQ 精确匹配）。两者互补：ChromaDB 适合"公司的考勤制度是什么"这类开放问题，JSON 工具适合"张建国的邮箱"这类精确查询。

**Q4: 如何保证工具调用的可靠性？**

三层容错：(1) 每个工具函数内部 `try-except` 返回错误消息而非抛出异常；(2) `ParallelToolNode._run_single_tool` 捕获工具执行异常返回 `ToolMessage(error)`；(3) 路由函数对异常状态有默认降级路径（如未知工具名默认路由到 generate）。

**Q5: MCP 服务器为什么不直接集成到 FastAPI 中？**

MCP 使用 stdio 传输时是独立子进程，由 Claude Code 管理生命周期。这实现了关注点分离：FastAPI 处理 Web 请求，MCP 服务器处理工具协议。两者通过共享 `tools_impl.py` 重用代码但运行时互不依赖。

**Q6: 如果 LLM 始终选择错误的工具怎么办？**

当前依赖 LLM function calling 的零样本能力。改进方案：(1) 在 Agent 提示中增加 few-shot 示例；(2) 使用更强的模型（如从 qwen2.5:7b 升级到 GPT-4o-mini）；(3) 增加路由层的意图分类模型做二次校验。

**Q7: 项目从"健康档案"变为"企业客服"改了什么？**

核心改动：工具描述从"健康档案查询"改为"企业知识库查询"；移除乘法演示工具，新增 4 个企业 JSON 数据工具；提示模板全面重写为企业客服角色；UI 从"INTELLIGENT HEALTH RECORD SYSTEM"改为"企业智能客服"；PDF 知识库从健康档案替换为裕兴集团员工手册；新增 MCP 服务器和共享工具实现架构。
