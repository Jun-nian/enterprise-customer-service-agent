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

    # API服务地址和端口
    HOST = "0.0.0.0"
    PORT = 8012
