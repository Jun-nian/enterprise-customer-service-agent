<div align="center">

# 🚀 LangGraph ChatBot — 智能客服 Agent

基于 **LangGraph** 的多版本智能客服对话系统，从基础聊天机器人到企业级 RAG 智能客服，涵盖 **多轮记忆 · 工具调用 · 动态路由 · 知识检索 · MCP 协议 · 强化学习** 全链路能力。

</div>

---

## 📑 目录

- [项目介绍](#-项目介绍)
- [功能特性](#-功能特性)
- [项目架构](#-项目架构)
- [项目结构](#-项目结构)
- [快速开始](#-快速开始)
- [分版本使用指南](#-分版本使用指南)
- [配置说明](#-配置说明)
- [MCP 集成](#-mcp-集成)
- [评测体系](#-评测体系)
- [技术栈](#-技术栈)
- [版本演进](#-版本演进)
- [License](#-license)

---

## 📖 项目介绍

本项目是 LangGraph 学习与实践的完整系列，包含 **7 个渐进式版本**，从最简单的聊天机器人逐步演进到具备 RAG 检索、多工具调用、MCP 协议和强化学习的企业级智能客服 Agent。

每个版本都基于前一个版本迭代，适合：
- 🎓 **学习者**：按版本顺序学习 LangGraph 的核心概念（状态图、记忆、工具调用、路由）
- 🏢 **开发者**：直接使用 06/07 版本作为企业智能客服的起点

---

## ✨ 功能特性

| 特性 | 说明 |
|------|------|
| 💬 **多轮对话** | 短期记忆（线程内）+ 长期记忆（跨线程持久化） |
| 🔧 **工具调用** | LLM 自动决策 + 并行工具执行（`ThreadPoolExecutor`） |
| 🔀 **动态路由** | 检索类工具 → 相关性评分 → 生成/重写 的智能路由 |
| 📚 **RAG 知识检索** | PDF → 向量化 → ChromaDB 语义搜索 |
| 📊 **多数据源工具** | 员工 / FAQ / 订单 / 工单 四类 JSON 数据查询 |
| 🐘 **PostgreSQL 持久化** | 会话状态与长期记忆的数据库存储 |
| 🔌 **MCP 协议** | 工具通过 Model Context Protocol 对外暴露（Claude Code 可直接调用） |
| 🎨 **Web UI** | Gradio 暗黑科幻主题 + 流式 SSE 响应 |
| 🤖 **强化学习** | ε-greedy bandit 自动选择最优检索策略（07 版） |
| 📈 **评测体系** | hit@k + LLM-judge 自动化评测框架（07 版） |

---

## 🏗️ 项目架构

### 核心工作流（以 07 企业客服版为例）

```
                ┌─────────────────────────────────────────────┐
                │              START                          │
                └──────────────────────┬──────────────────────┘
                                       ▼
                ┌─────────────────────────────────────────────┐
                │      agent（LLM 意图识别 + 工具决策）          │
                └──────────────────────┬──────────────────────┘
                          ┌────────────┴────────────┐
                          ▼                         ▼
             需要调用工具                   直接回复用户
                          │                         │
                          ▼                         ▼
        ┌─────────────────────────┐          ┌───────────┐
        │  call_tools（并行执行）   │          │  generate │
        └─────────────────────────┘          └─────┬─────┘
                          │                        │
                          ▼                        │
        ┌─────────────────────────┐                │
        │  grade_documents（评分）  │                │
        └────────────┬────────────┘                │
             ┌───────┴───────┐                    │
             ▼               ▼                    │
         相关性「yes」    相关性「no」              │
             │               │                    │
             ▼               ▼                    │
        ┌─────────┐   ┌──────────┐               │
        │ generate│◄──│  rewrite  │（最多3次）     │
        └────┬────┘   └──────────┘               │
             │                                   │
             └───────────────┬───────────────────┘
                             ▼
                          END
```

### 5 大核心节点

| 节点 | 职责 |
|------|------|
| **agent** | 分析用户问题，决定是否调用工具及调用哪个工具 |
| **call_tools** | 通过线程池并行执行工具调用 |
| **grade_documents** | LLM 结构化输出评估检索内容相关性（yes/no） |
| **rewrite** | 重写用户问题以改进检索召回率（最多 3 次） |
| **generate** | 根据检索上下文生成最终回复 |

---

## 📂 项目结构

```
LangGraphChatBot/
├── 01_ChatBot/                  # ① 基础聊天机器人（FastAPI + Gradio）
├── 02_ChatBot/                  # ② 记忆功能（短期 + 长期记忆 + 对话历史管理）
├── 03_ChatBotWithPostgres/      # ③ PostgreSQL 持久化存储
├── 04_RagAgent/                 # ④ RAG 智能客服（工具调用 + 动态路由）
├── 05_RagAgent_Business/        # ⑤ 企业客服版（本地 Ollama）
├── 06_RagAgent_Business/        # ⑥ 企业客服版（环境变量配置，公开版）
├── 07_RagAgent_Business/        # ⑦ 企业客服版（.env 配置 + RL + 评测）
├── 08_RagAgent_Business/        # ⑧ 企业客服版（规划范式 + 安全护栏 + 评测基准 + 多Agent）
│   ├── main.py                  # FastAPI 服务入口
│   ├── demoRagAgent.py          # LangGraph 工作流引擎
│   ├── webUI.py                 # Gradio 聊天界面
│   ├── vectorSaveTest.py        # PDF → ChromaDB 入库工具
│   ├── eval_rag.py              # 评测 CLI 入口
│   ├── evals/                   # 评测框架（dataset/judge/metrics/runner/trace）
│   ├── rl/                      # 强化学习（bandit + search_agent）
│   ├── mcp_server/              # MCP 服务器（工具实现单一事实来源）
│   ├── utils/                   # 配置、LLM 初始化、PDF 解析
│   ├── prompts/                 # 提示模板（agent/generate/grade/rewrite/judge）
│   ├── data/                    # JSON 模拟数据（employees/faq/orders/tickets）
│   └── input/                   # 知识库 PDF 文件
├── pictures/                    # 架构图等资源
└── README.md
```

> 每个版本的详细结构见各子目录内的 `PROJECT_SUMMARY.md` / `README.md`。

---

## 🚀 快速开始

### 环境要求

| 依赖 | 要求 |
|------|------|
| **Python** | 3.10+（推荐 3.13） |
| **Ollama**（可选） | 本地 LLM 推理服务，端口 11434 |
| **模型** | `qwen2.5:7b`（对话）+ `nomic-embed-text`（嵌入） |

### 1. 克隆项目

```bash
git clone https://github.com/Jun-nian/enterprise-customer-service-agent.git
cd enterprise-customer-service-agent
```

### 2. 安装依赖

```bash
# 推荐使用虚拟环境
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 安装核心依赖（以 06 版 requirements.txt 为例）
pip install -r 06_RagAgent_Business/requirements.txt

# 或按版本逐个安装（旧版本 01-04）
pip install langgraph==0.2.74
pip install langchain-openai==0.3.6
pip install fastapi==0.115.8
pip install uvicorn==0.34.0
pip install gradio==5.18.0
```

### 3. 拉取本地模型（可选）

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### 4. 配置环境变量（07 版）

```bash
cd 07_RagAgent_Business
cp .env.example .env
# 编辑 .env，填入你的 LLM API 密钥
# 使用 Ollama 本地模式则无需修改
```

---

## 📦 分版本使用指南

### 版本 ① 基础聊天机器人（01_ChatBot）

```bash
cd 01_ChatBot
python main.py        # 启动 FastAPI 服务（端口 8012）
python webUI.py       # 启动 Gradio 界面（端口 7860）
```

### 版本 ② 记忆功能（02_ChatBot）

```bash
cd 02_ChatBot
python demoWithMemory.py   # 短期/长期记忆 + 对话历史管理演示
python main.py             # 启动 API 服务
python webUI.py            # 启动 Web 界面
```

### 版本 ③ PostgreSQL 持久化（03_ChatBotWithPostgres）

```bash
cd 03_ChatBotWithPostgres
docker-compose up -d       # 启动 PostgreSQL（含 pgvector）

# 安装额外依赖
pip install langgraph-checkpoint-postgres psycopg psycopg-pool

python demoWithMemory.py   # 记忆演示
python main.py             # 启动 API 服务
python webUI.py            # 启动 Web 界面
```

### 版本 ④ RAG 智能客服（04_RagAgent）

```bash
cd 04_RagAgent

# 1. 知识灌库（将 PDF 向量化存入 ChromaDB）
python vectorSaveTest.py

# 2. 命令行直接对话
python demoRagAgent.py

# 3. 启动 API + Web 服务
python main.py
python webUI.py
```

### 版本 ⑤⑥ 企业智能客服（05/06_RagAgent_Business）

```bash
cd 06_RagAgent_Business

# 1. 知识入库（首次或更新 PDF 后）
python vectorSaveTest.py

# 2. 启动后端 API（终端 1）
python main.py                # http://0.0.0.0:8012

# 3. 启动 Web 界面（终端 2）
python webUI.py               # http://127.0.0.1:7860

# 4. 或命令行直接对话
python demoRagAgent.py
```

> **05 vs 06 区别**：05 是本地 Ollama 版（配置硬编码）；06 是公开版（配置改由环境变量读取），两者核心代码相同。

### 版本 ⑦ 企业智能客服 + RL（07_RagAgent_Business）

```bash
cd 07_RagAgent_Business

# 1. 配置 .env
cp .env.example .env

# 2. 知识入库
python vectorSaveTest.py

# 3. 启动 API + Web 服务
python main.py
python webUI.py

# 4. 运行评测
python eval_rag.py --limit 20            # hit@k + LLM-judge 评测
python eval_rag.py --no-judge            # 跳过 LLM 评判，仅检索命中率
python eval_rag.py --strategy auto       # RL bandit 在线学习评测
```

> 💡 07 版新增的 `ENABLE_SEARCH_AGENT` 开关默认关闭，开启后图结构才会注入 RL 搜索节点，不影响原有行为。

### 版本 ⑧ 企业智能客服 + 规划 + 护栏（08_RagAgent_Business，最新）

```bash
cd 08_RagAgent_Business

# 1. 配置 .env（含 08 新增能力开关）
cp .env.example .env

# 2. 启动 API + Web 服务
python main.py
python webUI.py

# 3. 能力开关（可选，按需开启）
#    ENABLE_PLANNING=true    # 规划范式（Planner/Executor/Critic）
#    ENABLE_GUARDRAILS=true  # 安全护栏（7类攻击检测 + PII脱敏）
#    ENABLE_RELIABILITY=true # 工具可靠性（超时/重试/熔断）

# 4. 评测
python eval_rag.py --agent-eval            # Agent 行为层评测
python eval_rag.py --agent-eval --compare  # 多策略基准对比（default vs plan-execute）
python eval_attack.py                      # 输入护栏攻击测试集评测
python evals/memory_eval.py                # 记忆检索质量评估
```

> 💡 08 版所有新增能力默认关闭，开启后仍保留简单查询快路径，不影响 07 既有行为。

---

## ⚙️ 配置说明

### 环境变量（07 版 .env）

```ini
# LLM 类型: deepseek / openai / oneapi / qwen / ollama
LLM_TYPE=ollama

# ---------- DeepSeek ----------
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
DEEPSEEK_API_KEY=sk-your-key
DEEPSEEK_CHAT_MODEL=deepseek-chat

# ---------- OpenAI ----------
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_API_KEY=sk-your-key

# ---------- 通义千问（嵌入）----------
EMBEDDING_PROVIDER=qwen
QWEN_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_API_KEY=sk-your-key
QWEN_EMBEDDING_MODEL=text-embedding-v1

# ---------- RL 搜索 Agent ----------
ENABLE_SEARCH_AGENT=false        # true 时启用 RL 搜索策略优化
```

### 支持的 LLM 后端

| 后端 | 用途 | 配置位置 |
|------|------|----------|
| **Ollama** | 本地聊天 + 嵌入 | 默认，零配置 |
| **DeepSeek** | 聊天（无嵌入，自动回退 Ollama） | 07 版 `.env` |
| **OpenAI** | 聊天 + 嵌入 | `.env` / `llms.py` |
| **通义千问** | 聊天 + 嵌入 | `.env` / `llms.py` |
| **OneAPI** | 聚合多模型 | `.env` / `llms.py` |

---

## 🔌 MCP 集成

06/07 版的企业工具同时通过 **MCP（Model Context Protocol）** 对外暴露，Claude Code / VSCode 等 MCP 客户端可直接调用。

### 在 Claude Code 中配置

```json
{
  "mcpServers": {
    "enterprise-customer-service": {
      "command": "python",
      "args": ["mcp_server/server.py"],
      "cwd": "/path/to/07_RagAgent_Business"
    }
  }
}
```

配置后 Claude Code 自动发现 5 个企业工具：

| 工具 | 功能 | 数据源 |
|------|------|--------|
| `retrieve` | 企业知识库语义搜索 | ChromaDB |
| `search_employees` | 员工信息查询 | data/employees.json |
| `search_faq` | 公司 FAQ 查询 | data/faq.json |
| `search_orders` | 销售订单查询 | data/orders.json |
| `search_tickets` | IT 工单查询 | data/tickets.json |

> 工具实现采用**单一事实来源**架构：`mcp_server/tools_impl.py` 是唯一实现，LangGraph 工作流与 MCP 服务器共享同一套代码。

---

## 📈 评测体系

07 版内置完整的 RAG 评测框架（`evals/` + `eval_rag.py`）：

| 指标 | 说明 |
|------|------|
| **hit@k** | 检索工具输出是否包含金标准子串 |
| **LLM-judge** | LLM 评判回答是否覆盖关键事实（`prompt_template_judge.txt`） |
| **策略分布** | RL 评测中各检索策略的选择统计 |

### 评测命令

```bash
python eval_rag.py --limit 20          # 评测前 20 个用例
python eval_rag.py --no-judge          # 跳过 LLM 评判（更快）
python eval_rag.py --strategy default  # 固定 default 策略（基线）
python eval_rag.py --strategy auto     # RL bandit 在线学习
python eval_rag.py --cases search_faq  # 只评测 FAQ 用例
```

---

## 🛠️ 技术栈

| 类别 | 技术 |
|------|------|
| **Agent 框架** | LangGraph |
| **LLM 生态** | LangChain, LangChain-Chroma, OpenAI SDK |
| **大模型** | Ollama / DeepSeek / OpenAI / 通义千问 / OneAPI |
| **向量数据库** | ChromaDB |
| **关系数据库** | PostgreSQL（+ pgvector） |
| **后端框架** | FastAPI + Uvicorn |
| **前端框架** | Gradio |
| **工具协议** | MCP (Model Context Protocol) |
| **强化学习** | ε-greedy 多臂老虎机（bandit） |
| **文档解析** | pdfminer.six, NLTK |

---

## 📜 版本演进

| 版本 | 目录 | 核心能力 | 状态 |
|------|------|----------|------|
| ① | `01_ChatBot` | 基础聊天机器人 + Web UI | ✅ 完整 |
| ② | `02_ChatBot` | + 短期/长期记忆 + 对话历史管理 | ✅ 完整 |
| ③ | `03_ChatBotWithPostgres` | + PostgreSQL 持久化存储 | ✅ 完整 |
| ④ | `04_RagAgent` | + RAG 知识检索 + 工具调用 + 动态路由 | ✅ 完整 |
| ⑤ | `05_RagAgent_Business` | 企业客服版（本地 Ollama） | ✅ 完整 |
| ⑥ | `06_RagAgent_Business` | 企业客服版（环境变量配置） | ✅ 完整 |
| ⑦ | `07_RagAgent_Business` | + RL 检索优化 + 评测体系 | ✅ 完整 |
| ⑧ | `08_RagAgent_Business` | + 规划范式 + 安全护栏 + 评测基准 + 多Agent（最新） | ✅ 最新 |

---

## 🛡️ 08 版亮点（08_RagAgent_Business）

08 版在 07 基础上升级为 **面试有竞争力的 Agent 作品**，新增四大能力：

### 1. 🧠 显式规划范式（P0-A）

将固定 5 节点 DAG 升级为 **Plan-and-Execute + 动态重规划**：

```
用户请求 → planner（复杂度判定）
  ├─ 简单查询 → agent（快路径，零额外成本）
  └─ 复杂查询 → executor（逐步执行）→ critic（评估）
                 ↑                        │
                 └── 失败重规划（≤2次） ←─┘
```

- `planning/planner.py`：结构化输出多步计划 + 规则预判复杂度
- `planning/executor.py`：按计划逐步调用工具，观察回填
- `planning/critic.py`：评估结果，决定继续/完成/重规划

### 2. 🛡️ 安全护栏层（P0-C）

复用 **vivo 7 类提示词攻击范式**（越狱/指令注入/角色扮演/数据窃取/混淆规避/有害内容/拒绝服务）：

- 输入侧：`guardrails/input_guard.py` 攻击检测，命中即拒答
- 输出侧：`guardrails/output_guard.py` PII 脱敏（手机号/邮箱/身份证）
- 越权校验：`guardrails/authz.py` 敏感工具角色鉴权
- 审计日志：`guardrails/audit.py` 脱敏记录，不落全量敏感数据

**攻击测试集实测：召回率 100%、误报率 0%**（`eval_attack.py`）

### 3. 📊 Agent 行为层评测基准（P0-B）

在 hit@k 之上补齐 Agent 行为层指标：

| 指标 | 说明 |
|------|------|
| 任务成功率 | LLM-judge 判定回答覆盖金标准 |
| 工具调用准确率 | 期望工具是否被正确调用 |
| 多步完成率 | complex 用例多步执行完成比例 |
| 无效 LLM 调用率 | 未调用工具却空转回答 |
| 步数/延迟/Token | 成本指标（可观测层） |

5 类评测集（单工具/多工具/跨源/无解/对抗）+ 多策略基准对比表导出。

### 4. 🔌 多智能体协作（P1-C）

`multiagent/roles.py`：Planner/Executor/Critic 角色化 + 领域子 Agent 路由
（HR/IT/业务/知识库），各子 Agent 持有独立工具子集。

### 5. ⚙️ 工具可靠性（P1-B）

`reliability/tool_wrapper.py`：超时 / 指数退避重试 / 失败降级 / 熔断器。

### 开启方式（全部默认关闭，兼容 07 行为）

```bash
# 在 08_RagAgent_Business/.env 中配置
ENABLE_PLANNING=true        # 规划范式
ENABLE_GUARDRAILS=true      # 安全护栏
ENABLE_RELIABILITY=true     # 工具可靠性

# 评测
python eval_rag.py --agent-eval           # Agent 行为层评测
python eval_rag.py --agent-eval --compare # 多策略基准对比
python eval_attack.py                     # 攻击护栏评测
python evals/memory_eval.py               # 记忆检索质量评估
```

---

## 📄 License

[MIT](LICENSE)

---

<div align="center">
Made with ❤️ by <a href="https://github.com/Jun-nian">Jun-nian</a>
</div>
