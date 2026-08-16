# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """统一的配置类，集中管理所有常量"""
    # prompt文件路径
    PROMPT_TEMPLATE_TXT_AGENT = "prompts/prompt_template_agent.txt"
    PROMPT_TEMPLATE_TXT_GRADE = "prompts/prompt_template_grade.txt"
    PROMPT_TEMPLATE_TXT_REWRITE = "prompts/prompt_template_rewrite.txt"
    PROMPT_TEMPLATE_TXT_GENERATE = "prompts/prompt_template_generate.txt"
    PROMPT_TEMPLATE_TXT_JUDGE = "prompts/prompt_template_judge.txt"
    PROMPT_TEMPLATE_TXT_PLANNER = "prompts/prompt_template_planner.txt"

    # Chroma 数据库配置
    CHROMADB_DIRECTORY = "chromaDB"
    CHROMADB_COLLECTION_NAME = "yuxing_handbook"

    # 日志持久化存储
    LOG_FILE = "output/app.log"
    MAX_BYTES=5*1024*1024,
    BACKUP_COUNT=3

    # 数据库 URI，默认值
    DB_URI = os.getenv("DB_URI", "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable")

    # deepseek/openai/oneapi/qwen/ollama
    LLM_TYPE = os.getenv("LLM_TYPE", "ollama")

    # RL/搜索Agent 开关（默认关闭：图结构与行为与原来完全一致）
    ENABLE_SEARCH_AGENT = os.getenv("ENABLE_SEARCH_AGENT", "false") == "true"
    # bandit 状态持久化文件
    BANDIT_STATE_FILE = os.getenv("BANDIT_STATE_FILE", "output/bandit_state.json")

    # P0-A 规划范式开关（默认关闭：保持原 5 节点快路径；开启后复杂任务走 Plan-and-Execute）
    ENABLE_PLANNING = os.getenv("ENABLE_PLANNING", "false") == "true"
    # 动态重规划上限（防无限 replan 循环）
    PLAN_MAX_REPLAN = int(os.getenv("PLAN_MAX_REPLAN", "2"))

    # P0-C 安全护栏开关（默认关闭：不影响原行为；开启后启用输入检测/输出脱敏/越权校验/审计）
    ENABLE_GUARDRAILS = os.getenv("ENABLE_GUARDRAILS", "false") == "true"
    # 护栏模式：block=命中即拒答 / flag=命中降级为受限回答
    GUARD_MODE = os.getenv("GUARD_MODE", "block")
    # 审计开关（随护栏开启默认启用）
    ENABLE_AUDIT = os.getenv("ENABLE_AUDIT", "true") == "true"

    # P1-B 工具可靠性开关（默认关闭：保持原调用路径；开启后注入 超时/重试/降级/熔断）
    ENABLE_RELIABILITY = os.getenv("ENABLE_RELIABILITY", "false") == "true"
    # 工具单次调用超时（秒）
    TOOL_TIMEOUT_SECONDS = float(os.getenv("TOOL_TIMEOUT_SECONDS", "20"))
    # 工具最大重试次数
    TOOL_MAX_RETRIES = int(os.getenv("TOOL_MAX_RETRIES", "2"))

    # API服务地址和端口
    HOST = "0.0.0.0"
    PORT = 8012
