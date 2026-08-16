# 08_RagAgent_Business — 企业智能客服 Agent（增强版）

在 **07_RagAgent_Business** 基础上升级的 **面试级 Agent 作品**，新增：

| 能力 | 模块 | 说明 |
|------|------|------|
| 🧠 **规划范式** (P0-A) | `planning/` | Plan-and-Execute + 动态重规划（Planner/Executor/Critic） |
| 🛡️ **安全护栏** (P0-C) | `guardrails/` | 7 类攻击检测 + PII 脱敏 + 越权校验 + 审计（复用 vivo 范式） |
| 📊 **评测基准** (P0-B) | `evals/` | Agent 行为层指标 + 5 类评测集 + 多策略基准对比 |
| 🔌 **多智能体** (P1-C) | `multiagent/` | 领域子 Agent 路由 + 角色化协作 |
| ⚙️ **工具可靠性** (P1-B) | `reliability/` | 超时 / 重试 / 降级 / 熔断 |
| 👁️ **可观测性** (P1-A) | `evals/observability.py` | LLM 调用级 token/延迟/重试采集 |

## 架构

```
START → guard_input(输入护栏)
  → planner(复杂度判定)
    ├─ 简单 → agent(快路径) → call_tools → grade_documents → generate
    └─ 复杂 → executor(逐步执行) → call_tools → critic(评估)
         ↑                                      │
         └──────── 失败重规划(≤2次) ←────────────┘
  → guard_output(PII脱敏) → END
```

## 快速开始

```bash
cp .env.example .env           # 配置 LLM + 能力开关
python main.py                 # FastAPI 服务 (端口 8012)
python webUI.py                # Gradio 界面 (端口 7860)
python demoRagAgent.py         # 命令行对话
```

## 能力开关（默认关闭，兼容 07）

| 开关 | 功能 |
|------|------|
| `ENABLE_PLANNING=true` | 规划范式：复杂查询多步执行 |
| `ENABLE_GUARDRAILS=true` | 安全护栏：输入检测 + PII 脱敏 |
| `ENABLE_RELIABILITY=true` | 工具可靠性：超时/重试/熔断 |

## 评测

```bash
python eval_rag.py --agent-eval             # Agent 行为层评测
python eval_rag.py --agent-eval --compare   # 多策略基准对比 → output/agent_benchmark.md
python eval_attack.py                       # 攻击护栏评测（召回率/误报率）
python evals/memory_eval.py                 # 记忆检索质量评估
```

## 目录结构

```
08_RagAgent_Business/
├── main.py                 # FastAPI 服务入口
├── demoRagAgent.py         # LangGraph 工作流引擎（8 节点：guard_input/planner/agent/call_tools/grade/rewrite/generate/guard_output）
├── eval_rag.py             # 评测 CLI（检索层 + Agent 行为层 + 基准对比）
├── eval_attack.py          # 攻击护栏评测 CLI
├── planning/               # 规划范式：planner/executor/critic
├── guardrails/             # 安全护栏：input/output/authz/audit + 7类攻击范式
│   └── tests/              # 攻击测试用例集
├── reliability/            # 工具可靠性：tool_wrapper
├── multiagent/             # 多智能体：roles/orchestrator
├── evals/                  # 评测框架：trace/metrics/judge/agent_eval/memory_eval
├── mcp_server/             # MCP 服务器（工具单一事实来源）
├── utils/                  # 配置、LLM 初始化、PDF 解析
├── prompts/                # 提示模板（含 planner）
└── data/ input/            # 数据与知识库
```
