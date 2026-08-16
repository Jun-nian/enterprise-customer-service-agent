import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from openai import OpenAI
import logging
from typing import List

# 加载 .env 文件
load_dotenv()

# 设置日志模版
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 自定义兼容 Ollama 的 Embeddings 类
class OllamaEmbeddings(Embeddings):
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            resp = self.client.embeddings.create(input=text, model=self.model)
            embeddings.append(resp.data[0].embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(input=text, model=self.model)
        return resp.data[0].embedding


# 模型配置 — 从环境变量读取
MODEL_CONFIGS = {
    "deepseek": {
        "base_url": os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "chat_model": os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat"),
    },
    "openai": {
        "base_url": os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "chat_model": os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        "embedding_model": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    },
    "oneapi": {
        "base_url": os.getenv("ONEAPI_API_BASE", ""),
        "api_key": os.getenv("ONEAPI_API_KEY", ""),
        "chat_model": os.getenv("ONEAPI_CHAT_MODEL", "qwen-max"),
        "embedding_model": os.getenv("ONEAPI_EMBEDDING_MODEL", "text-embedding-v1"),
    },
    "qwen": {
        "base_url": os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "api_key": os.getenv("QWEN_API_KEY", ""),
        "chat_model": os.getenv("QWEN_CHAT_MODEL", "qwen-max"),
        "embedding_model": os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v1"),
    },
    "ollama": {
        "base_url": os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1"),
        "api_key": "ollama",
        "chat_model": os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b"),
        "embedding_model": os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest"),
    },
}

# 嵌入维度映射 (用于 InMemoryStore)
EMBEDDING_DIMS = {
    "nomic-embed-text": 768,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "text-embedding-v1": 1536,
}

# 默认配置
DEFAULT_LLM_TYPE = "ollama"
DEFAULT_TEMPERATURE = 0.7

# 嵌入独立配置 — 允许与聊天 LLM 使用不同后端
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", None)


def _get_embedding_config(llm_type: str) -> dict:
    """获取嵌入模型配置，优先使用独立 EMBEDDING_PROVIDER"""
    if EMBEDDING_PROVIDER and EMBEDDING_PROVIDER in MODEL_CONFIGS:
        provider = EMBEDDING_PROVIDER
    else:
        provider = llm_type

    config = MODEL_CONFIGS[provider]
    if provider == "ollama":
        return {
            "provider": "ollama",
            "base_url": config["base_url"],
            "api_key": config["api_key"],
            "model": config["embedding_model"],
        }
    elif provider in ("openai", "oneapi", "qwen"):
        return {
            "provider": provider,
            "base_url": config["base_url"],
            "api_key": config["api_key"],
            "model": config.get("embedding_model", "text-embedding-3-small"),
        }
    else:
        # deepseek 等无嵌入模型的平台，回退到 Ollama
        ollama_config = MODEL_CONFIGS["ollama"]
        logger.info(f"{llm_type} 无嵌入模型，回退到 Ollama 嵌入")
        return {
            "provider": "ollama",
            "base_url": ollama_config["base_url"],
            "api_key": ollama_config["api_key"],
            "model": ollama_config["embedding_model"],
        }


class LLMInitializationError(Exception):
    """自定义异常类用于LLM初始化错误"""
    pass


def initialize_llm(llm_type: str = DEFAULT_LLM_TYPE) -> tuple[ChatOpenAI, Embeddings]:
    """
    初始化LLM实例

    Args:
        llm_type (str): LLM类型，可选值为 'deepseek', 'openai', 'oneapi', 'qwen', 'ollama'

    Returns:
        ChatOpenAI: 初始化后的LLM实例
        Embeddings: 嵌入模型实例

    Raises:
        LLMInitializationError: 当LLM初始化失败时抛出
    """
    try:
        if llm_type not in MODEL_CONFIGS:
            raise ValueError(f"不支持的LLM类型: {llm_type}. 可用的类型: {list(MODEL_CONFIGS.keys())}")

        config = MODEL_CONFIGS[llm_type]

        # 创建聊天 LLM
        llm_chat = ChatOpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["chat_model"],
            temperature=DEFAULT_TEMPERATURE,
            timeout=120,
            max_retries=2
        )

        # 创建嵌入模型（统一使用 OllamaEmbeddings = 原始 OpenAI client，兼容所有 OpenAI 风格 API）
        emb_config = _get_embedding_config(llm_type)

        llm_embedding = OllamaEmbeddings(
            base_url=emb_config["base_url"],
            api_key=emb_config["api_key"],
            model=emb_config["model"]
        )

        logger.info(f"成功初始化 {llm_type} LLM (聊天: {config['chat_model']}, 嵌入: {emb_config['provider']}/{emb_config['model']})")
        return llm_chat, llm_embedding

    except ValueError as ve:
        logger.error(f"LLM配置错误: {str(ve)}")
        raise LLMInitializationError(f"LLM配置错误: {str(ve)}")
    except Exception as e:
        logger.error(f"初始化LLM失败: {str(e)}")
        raise LLMInitializationError(f"初始化LLM失败: {str(e)}")


def get_llm(llm_type: str = DEFAULT_LLM_TYPE) -> tuple[ChatOpenAI, Embeddings]:
    """
    获取LLM实例的封装函数，提供默认值和错误处理

    Args:
        llm_type (str): LLM类型

    Returns:
        tuple[ChatOpenAI, Embeddings]: LLM实例和嵌入模型实例
    """
    try:
        return initialize_llm(llm_type)
    except LLMInitializationError as e:
        logger.warning(f"使用默认配置重试: {str(e)}")
        if llm_type != DEFAULT_LLM_TYPE:
            return initialize_llm(DEFAULT_LLM_TYPE)
        raise


def get_embedding_dims(embedding: Embeddings) -> int:
    """获取嵌入模型的向量维度"""
    sample = embedding.embed_query("test")
    return len(sample)
