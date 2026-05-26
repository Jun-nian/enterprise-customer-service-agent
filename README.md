# 企业智能客服 Agent

基于 **LangGraph + ChromaDB + FastAPI + Gradio** 的企业内部智能客服系统，支持 RAG 知识检索、多工具调用和 MCP 标准协议。

## 架构

```
START → agent (LLM决策) → call_tools (并行执行) → grade_documents (相关性评分)
                │              │                          │
                ▼              ▼                          ▼
               END         generate ←─────────────────────┘
                               ↑
                          rewrite (重写问题, 最多3次)
```

**5 个节点**: agent / call_tools / grade_documents / rewrite / generate  
**5 个工具**: 知识库检索 / 员工查询 / FAQ 查询 / 订单查询 / 工单查询  
**MCP 协议**: 工具同时通过 MCP stdio 对外暴露

## 功能

- 企业知识库语义搜索 (ChromaDB + nomic-embed-text)
- 员工信息查询 (按姓名/部门/职位)
- 公司 FAQ 查询 (入职/考勤/报销/IT/行政/福利)
- 销售订单查询 (按客户/订单号/状态)
- IT 工单查询 (按编号/标题/申请人/状态)
- 流式 SSE 响应 + Gradio 暗黑主题 UI
- MCP 协议支持 (Claude Code / VSCode 可直接调用工具)

## 快速开始

### 环境要求

- Python 3.10+
- Ollama (推荐，零配置本地运行)

### 安装

```bash
git clone https://github.com/YOUR_USERNAME/enterprise-customer-service-agent.git
cd enterprise-customer-service-agent
pip install -r requirements.txt
```

### 拉取模型

```bash
ollama pull qwen2.5:7b
ollama pull nomic-embed-text
```

### 配置

```bash
cp .env.example .env
# 如果使用 Ollama 本地模式，无需修改 .env
# 如果使用 OpenAI/通义千问，编辑 .env 填入 API 密钥
```

### 导入知识库

```bash
# 将你的 PDF 文档放入 input/ 目录
# 修改 vectorSaveTest.py 中的 INPUT_PDF 路径
python vectorSaveTest.py
```

### 启动

```bash
# 终端1: 后端 API 服务
python main.py                    # http://localhost:8012

# 终端2: Web 聊天界面
python webUI.py                   # http://127.0.0.1:7860

# 或命令行直接对话
python demoRagAgent.py
```

## 项目结构

```
├── main.py                  # FastAPI 服务入口
├── demoRagAgent.py          # LangGraph 工作流引擎
├── webUI.py                 # Gradio 聊天 UI
├── vectorSaveTest.py        # PDF → ChromaDB 入库工具
├── apiTest.py               # API 测试脚本
├── utils/
│   ├── config.py            # 统一配置
│   ├── llms.py              # LLM 初始化 (OpenAI/OneAPI/Qwen/Ollama)
│   ├── tools_config.py      # LangChain 工具包装
│   ├── pdfSplitTest_Ch.py   # 中文 PDF 解析
│   └── pdfSplitTest_En.py   # 英文 PDF 解析
├── mcp_server/
│   ├── tools_impl.py        # 工具核心实现 (单一事实来源)
│   └── server.py            # MCP 服务器 (FastMCP + stdio)
├── prompts/                 # 提示模板 (agent/generate/grade/rewrite)
├── data/                    # JSON 模拟数据
└── input/                   # PDF 知识库文件
```

## MCP 配置

在 Claude Code 或其他 MCP 客户端的配置中添加：

```json
{
  "mcpServers": {
    "enterprise-customer-service": {
      "command": "python",
      "args": ["mcp_server/server.py"],
      "cwd": "/path/to/enterprise-customer-service-agent"
    }
  }
}
```

配置后 Claude Code 会自动发现 5 个企业工具。

## 技术栈

| 组件 | 技术 |
|------|------|
| Agent 框架 | LangGraph 1.2 |
| LLM | Ollama (qwen2.5:7b) / OpenAI / 通义千问 |
| 嵌入模型 | nomic-embed-text / text-embedding-3-small |
| 向量数据库 | ChromaDB |
| API 框架 | FastAPI |
| 前端 | Gradio 6.x |
| 工具协议 | MCP (Model Context Protocol) |

## License

MIT
