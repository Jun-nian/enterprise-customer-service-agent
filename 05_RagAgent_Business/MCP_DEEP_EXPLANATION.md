# MCP（Model Context Protocol）在 05_RagAgent_Business 中的深度解析

---

## 一、原理上如何理解

### 1.1 一句话本质

**MCP 是一个标准化的"工具说明书协议"**——它定义了一套通用格式，让任何 LLM 客户端（Claude Code、VSCode 扩展、自研应用）都能**自动发现、理解和调用**你写好的工具函数，而不需要人工配置映射。

### 1.2 类比：USB 协议

```
USB 设备插入 → 主机自动识别 → 加载驱动 → 可用

MCP 服务器启动 → 客户端自动发现 → 解析工具签名 → LLM 可调用
```

传统方式：每个 AI 应用都要单独写一遍"工具 X 是什么，参数 Y 是什么类型"。  
MCP 方式：工具定义一次，通过标准协议暴露，所有兼容客户端自动获得调用能力。

### 1.3 三层抽象

```
┌──────────────────────────────────────────────┐
│  应用层   │ 具体函数实现                        │
│           │ tools_impl.py: search_employees() │
├──────────────────────────────────────────────┤
│  表示层   │ JSON Schema 描述工具签名            │
│           │ {name:"search_employees",         │
│           │  inputSchema:{query:string},       │
│           │  description:"查询公司员工信息..."}  │
├──────────────────────────────────────────────┤
│  传输层   │ stdio / SSE / Streamable HTTP     │
│           │ 进程标准输入输出流                   │
└──────────────────────────────────────────────┘
```

在本项目中：

| 层 | 位置 | 具体实现 |
|----|------|----------|
| 应用层 | [mcp_server/tools_impl.py](05_RagAgent_Business/mcp_server/tools_impl.py) | 5 个 Python 函数（纯逻辑，零框架依赖） |
| 表示层 | [mcp_server/server.py](05_RagAgent_Business/mcp_server/server.py) | 通过 `@mcp.tool()` 装饰器自动生成 JSON Schema |
| 传输层 | [.claude/mcp.json](.claude/mcp.json) | stdio 子进程通信 |

---

## 二、逻辑上如何理解

### 2.1 架构关键问题：为什么同一个工具要实现两套？

项目中有两条**平行的调用路径**，但共享同一套实现：

```
                     tools_impl.py
                     （唯一事实来源）
                    /                \
                   /                  \
     @mcp.tool() 装饰               @tool 装饰 (LangChain)
     server.py                    tools_config.py
          |                              |
     FastMCP + stdio               LangGraph bind_tools()
          |                              |
     Claude Code 调用              LangGraph 工作流调用
   （外部 AI 客户端）              （内部 Agent 引擎）
```

**这不是重复实现，而是"同一种能力，两种接入协议"：**

| 维度 | MCP 路径 | LangChain 路径 |
|------|----------|----------------|
| 调用方 | Claude Code / VSCode 等外部客户端 | 本项目的 LangGraph Agent |
| 通信 | stdio 子进程 (跨进程) | Python import (进程内) |
| 工具描述 | MCP JSON Schema | LangChain BaseTool JSON Schema |
| 启动 | Claude Code 自动拉起子进程 | FastAPI lifespan 内初始化 |

### 2.2 为什么不合二为一？

**根本原因：进程边界不同。**

- LangGraph Agent 运行在 FastAPI 进程内，通过 Python 对象直接交互
- Claude Code 是一个**独立进程**，无法访问 FastAPI 内存中的 Python 对象

MCP 的作用就是在这两个独立进程之间架设一座**标准化桥梁**。

### 2.3 FastMCP 自动做了哪些事？

当你在 `server.py` 中写：

```python
@mcp.tool()
def search_employees_tool(query: str) -> str:
    return search_employees(query)
```

`@mcp.tool()` 装饰器自动完成以下工作：

1. **提取函数名** → `search_employees_tool`
2. **推导类型签名** → `{query: string} → string`
3. **提取 docstring** → 工具描述文本
4. **生成 JSON Schema** → 符合 MCP 协议的 `Tool` 对象
5. **注册到 FastMCP** → 加入 `list_tools()` 响应列表
6. **路由分发** → 收到 `tools/call` 请求时自动匹配并执行

---

## 三、举例数值计算说明

### 3.1 场景：用户通过 Claude Code 查询"张建国的邮箱"

#### Step 1：MCP 服务器启动（自动）

Claude Code 读取 `.claude/mcp.json`，启动子进程：

```
进程树：
Claude Code (PID: 1000)
  └── python server.py (PID: 1020)  ← MCP 子进程，通过 stdin/stdout 通信
```

内存占用估算：

| 组件 | 内存 |
|------|------|
| 子进程基础 | ~20 MB (Python 解释器) |
| tools_impl.py 导入 | ~2 MB (json, os, openai 模块) |
| ChromaDB 客户端 (lazy) | 0 MB (首次调用 `retrieve` 时才连接) |
| **MCP 服务器总计** | **~22 MB** |

#### Step 2：工具发现（JSON-RPC 交换）

Claude Code 通过 stdin 发送 JSON-RPC：

```json
// Claude Code → MCP Server (stdin)
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}
```

MCP 服务器通过 stdout 返回：

```json
// MCP Server → Claude Code (stdout)
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {
        "name": "search_employees_tool",
        "description": "查询公司员工信息。支持按姓名、部门名称、职位关键词搜索...",
        "inputSchema": {
          "type": "object",
          "properties": {
            "query": {"type": "string"}
          },
          "required": ["query"]
        }
      },
      // ... 其余 4 个工具
    ]
  }
}
```

**数值指标：** 5 个工具，JSON 约 3KB，传输耗时 <1ms（本地 stdin/stdout）。

#### Step 3：工具调用（端到端链路）

```
用户输入: "张建国的邮箱是什么？"
                │
Claude Code     │  LLM 分析输入 → 匹配工具 search_employees_tool
                │  发送 JSON-RPC tools/call:
                │  {
                │    "method": "tools/call",
                │    "params": {
                │      "name": "search_employees_tool",
                │      "arguments": {"query": "张建国"}
                │    }
                │  }
                │
MCP Server      │  server.py → tools_impl.search_employees("张建国")
(stdin/stdout)  │  → 打开 data/employees.json (20条记录, ~3KB)
                │  → 遍历匹配 name/department/position (20次字符串比较)
                │  → 返回结果:
                │  "姓名: 张建国, 部门: 技术研发中心,
                │   职位: 技术总监, 邮箱: zhangjianguo@xinrui.com"
                │
                │  响应耗时:
                │  JSON解析:   ~0.01ms
                │  文件IO:     ~0.5ms  (3KB JSON)
                │  循环匹配:   ~0.05ms (20条 × 3字段)
                │  返回序列化: ~0.01ms
                │  ─────────────────
                │  工具执行:   ~0.6ms
                │  ─────────────────
                │  MCP 往返总耗时: ~1ms (本地stdio)
                │
Claude Code     │  收到结果 → 格式化回复 → "张建国的邮箱是 zhangjianguo@xinrui.com"
```

**对比：如果没有 MCP**

如果 Claude Code 想调用同样的工具但没有 MCP，需要：

1. 人工手写工具描述（容易出错）
2. 维护另一份工具实现副本（代码重复）
3. 手动配置 RPC 通信（非标准协议）

**数值对比：**

| 方式 | 代码重复 | 配置工作量 | 工具同步 |
|------|----------|------------|----------|
| MCP | 0 行重复 | 3 行 JSON 配置 | 自动（共享 tools_impl.py） |
| 无 MCP | ~140 行重复 | 需自定义协议 | 手动维护双份 |

---

## 四、有什么意义

### 4.1 对本项目的意义

```
                   在没有 MCP 之前
                   ================
                   
   tools_config.py (140行 JSON 处理逻辑)
          ↑
          │ Python import
          │
   LangGraph Agent ──── 对用户消息响应
   
   外部 AI 客户端 (Claude Code) ──── ✗ 无法调用这些工具
                                       （不知道工具存在，不知道如何调用）


                   有了 MCP 之后
                   =============
                   
              tools_impl.py (140行核心逻辑，唯一来源)
              /                \
   tools_config.py          server.py
   (LangChain @tool)        (FastMCP @mcp.tool)
        |                        |
   LangGraph Agent          Claude Code
   (内部使用)               (外部使用)
   
   新增一个工具只需修改 tools_impl.py → 两个路径自动生效
```

### 4.2 战略意义

| 意义 | 说明 |
|------|------|
| **协议标准化** | MCP 是 Anthropic 主导的开放标准，类似 USB 之于外设。一次编写，多客户端使用 |
| **生态互通** | 同样的工具可被 Claude Code、VSCode Copilot、Cursor、自研 Agent 同时调用 |
| **厂商解耦** | 工具实现不绑定任何框架（tools_impl.py 不依赖 LangChain 也不依赖 MCP SDK） |
| **渐进增强** | 不影响现有工作流，MCP 作为一个并列通道而非替代现有代码 |
| **团队协作** | 不同开发者可独立开发 MCP 工具服务器，通过标准协议集成 |

### 4.3 实际场景类比

```
MCP 之于 AI 工具   ≈   REST API 之于 Web 服务

REST API: 前端调后端有标准格式 (GET/POST + JSON)
MCP:      AI客户端调工具有标准格式 (tools/list + tools/call)
```

---

## 五、在整个流程中该步骤处于什么位置，作用是什么

### 5.1 MCP 在完整数据流中的位置

```
用户输入 (自然语言)
       │
       ▼
┌──────────────────────────────────────────────────────┐
│ [A] 前端层  (Gradio :7860 / Claude Code UI)          │
│   用户消息被发送到对应后端                              │
└────────┬─────────────────────────┬───────────────────┘
         │ 路径1: Gradio → FastAPI  │ 路径2: Claude Code → MCP
         ▼                         ▼
┌──────────────────┐   ┌──────────────────────────────┐
│ [B] FastAPI      │   │  ★ [M] MCP Server ★          │
│ main.py          │   │  mcp_server/server.py         │
│                  │   │                               │
│ POST /v1/chat/   │   │  注册5个 MCP Tool             │
│ completions      │   │  (stdio 子进程)               │
│      │           │   │        │                      │
│      ▼           │   │        ▼                      │
│ LangGraph Graph  │   │  tools_impl.py                │
│      │           │   │  (与下方共享实现)              │
└──────┼───────────┘   └──────────┬───────────────────┘
       │                          │
       ▼                          ▼
┌──────────────────────────────────────────────────────┐
│ [C] tools_impl.py  ★ 共享实现层 ★                     │
│  retrieve / search_employees / search_faq /          │
│  search_orders / search_tickets                      │
│       │                                              │
│       ▼                                              │
│ [D] 数据层                                           │
│  ChromaDB / employees.json / faq.json /              │
│  orders.json / tickets.json                          │
└──────────────────────────────────────────────────────┘
```

### 5.2 路径对比

| 步骤 | 路径 1：LangGraph Agent 内部调用 | 路径 2：MCP 外部调用 |
|------|----------------------------------|-----------------------|
| **触发方** | 用户通过 Gradio UI 发消息 | 用户通过 Claude Code 对话 |
| **入口** | FastAPI `lifespan()` 初始化 Graph | `.claude/mcp.json` 拉起子进程 |
| **工具加载** | `get_tools()` → `@tool` 装饰 → `bind_tools()` | `@mcp.tool()` 装饰 → `list_tools` 响应 |
| **工具发现** | LangChain 自动从 `@tool` 生成 function calling schema | MCP JSON-RPC `tools/list` 返回 JSON Schema |
| **工具执行** | `ParallelToolNode` 线程池并行执行 | MCP `tools/call` → Python 函数同步执行 |
| **结果处理** | `grade_documents` → `generate` → SSE 流 | Claude Code LLM 自行处理结果 |
| **状态管理** | `MessagesState` + `MemorySaver` + `InMemoryStore` | 无状态（Claude Code 管理对话） |

### 5.3 MCP 的核心作用

```
MCP 的作用可以三句话概括：

1. 工具定义标准化层
   让"这款工具有什么功能、接受什么参数、返回什么结果"
   的元信息有了通用格式。

2. 工具通信协议层
   让"AI 客户端如何发现工具、如何调用工具、如何获取结果"
   有了标准流程（JSON-RPC over stdio/SSE/HTTP）。

3. 工具实现解耦层
   让工具实现（tools_impl.py）与任何特定框架（LangChain、Claude Code）
   解耦，通过薄薄的适配层即可接入任意兼容协议的系统。
```

### 5.4 一句话总结

> **MCP 位于"用户输入"和"工具数据源"之间，作为一条独立于 LangGraph 主流程的旁路通道：它不对已有的 agent→tools→grade→generate 流程做任何修改，而是开了一扇标准化的门，让外部 AI 客户端也能"看到并调用"相同的工具集。**
