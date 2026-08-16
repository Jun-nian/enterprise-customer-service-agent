# mcp_server/tools_impl.py
# 企业智能客服工具实现（纯 Python，无 LangChain 依赖）
# 作为 MCP server 和 LangGraph tools_config.py 的共享实现层

import json
import os

# 路径常量
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMADB_DIR = os.path.join(BASE_DIR, "chromaDB")
COLLECTION_NAME = "yuxing_handbook"

# 嵌入客户端 — 从环境变量读取，支持多后端
from dotenv import load_dotenv
load_dotenv()

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", os.getenv("LLM_TYPE", "ollama"))
if EMBEDDING_PROVIDER == "deepseek":
    EMBEDDING_PROVIDER = "ollama"  # DeepSeek 无嵌入模型，回退到 Ollama

if EMBEDDING_PROVIDER == "ollama":
    EMBEDDING_BASE_URL = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1")
    EMBEDDING_API_KEY = "ollama"
    EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")
elif EMBEDDING_PROVIDER == "openai":
    EMBEDDING_BASE_URL = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    EMBEDDING_API_KEY = os.getenv("OPENAI_API_KEY", "")
    EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
elif EMBEDDING_PROVIDER == "qwen":
    EMBEDDING_BASE_URL = os.getenv("QWEN_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    EMBEDDING_API_KEY = os.getenv("QWEN_API_KEY", "")
    EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v1")
elif EMBEDDING_PROVIDER == "oneapi":
    EMBEDDING_BASE_URL = os.getenv("ONEAPI_API_BASE", "")
    EMBEDDING_API_KEY = os.getenv("ONEAPI_API_KEY", "")
    EMBEDDING_MODEL = os.getenv("ONEAPI_EMBEDDING_MODEL", "text-embedding-v1")
else:
    EMBEDDING_BASE_URL = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1")
    EMBEDDING_API_KEY = "ollama"
    EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest")

from openai import OpenAI
_embedding_client = OpenAI(base_url=EMBEDDING_BASE_URL, api_key=EMBEDDING_API_KEY)


def _get_embedding(text: str) -> list:
    """调用嵌入模型生成向量"""
    resp = _embedding_client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return resp.data[0].embedding


def retrieve(query: str, n_results: int = 5) -> str:
    """搜索企业知识库（ChromaDB），返回与查询相关的文档内容。

    Args:
        query: 查询文本
        n_results: 返回的文档数量（top_k 策略可调大以提高召回）

    Returns:
        str: 检索到的文档拼接文本
    """
    try:
        import chromadb
        client = chromadb.PersistentClient(path=CHROMADB_DIR)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        emb = _get_embedding(query)
        results = collection.query(query_embeddings=[emb], n_results=n_results)
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            return "\n\n---\n\n".join(docs)
        return "未找到相关知识库信息，请尝试使用其他关键词搜索。"
    except Exception as e:
        return f"知识库查询失败: {str(e)}"


def search_employees(query: str, limit: int = 5) -> str:
    """查询公司员工信息，支持按姓名、部门、职位进行搜索。

    Args:
        query: 查询关键词
        limit: 返回结果条数上限

    Returns:
        str: 匹配的员工信息文本
    """
    filepath = os.path.join(DATA_DIR, "employees.json")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            employees = json.load(f)
        results = []
        query_lower = query.lower()
        for emp in employees:
            if (query_lower in emp.get("name", "").lower()
                or query_lower in emp.get("department", "").lower()
                or query_lower in emp.get("position", "").lower()):
                results.append(
                    f"姓名: {emp['name']}, 部门: {emp['department']}, "
                    f"职位: {emp['position']}, 上级: {emp['manager']}, "
                    f"邮箱: {emp['email']}, 电话: {emp['phone']}"
                )
        if results:
            return "\n".join(results[:limit])
        return f"未找到与'{query}'匹配的员工信息"
    except Exception as e:
        return f"员工查询失败: {str(e)}"


def search_faq(query: str, limit: int = 3) -> str:
    """查询公司常见问题(FAQ)，涵盖入职、考勤、报销、IT支持、HR政策、行政服务、员工福利等类别。

    Args:
        query: 查询关键词
        limit: 返回结果条数上限

    Returns:
        str: 匹配的FAQ文本
    """
    filepath = os.path.join(DATA_DIR, "faq.json")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            faqs = json.load(f)
        results = []
        query_lower = query.lower()
        for faq in faqs:
            if (query_lower in faq.get("question", "").lower()
                or query_lower in faq.get("topic", "").lower()
                or query_lower in faq.get("answer", "").lower()):
                results.append(f"【{faq['topic']}】{faq['question']}\n回答: {faq['answer']}")
        if results:
            return "\n\n".join(results[:limit])
        return f"未找到与'{query}'匹配的FAQ信息"
    except Exception as e:
        return f"FAQ查询失败: {str(e)}"


def search_orders(query: str, limit: int = 5) -> str:
    """查询公司销售订单信息，支持按客户名称、订单编号、销售代表、订单状态进行搜索。

    Args:
        query: 查询关键词
        limit: 返回结果条数上限

    Returns:
        str: 匹配的订单详情文本
    """
    filepath = os.path.join(DATA_DIR, "orders.json")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            orders = json.load(f)
        results = []
        query_lower = query.lower()
        for order in orders:
            if (query_lower in order.get("order_id", "").lower()
                or query_lower in order.get("customer", "").lower()
                or query_lower in order.get("sales_rep", "").lower()
                or query_lower in order.get("status", "").lower()):
                results.append(
                    f"订单编号: {order['order_id']}, 客户: {order['customer']}, "
                    f"产品: {order['product']}, 金额: {order['amount']}, "
                    f"状态: {order['status']}, 销售代表: {order['sales_rep']}, "
                    f"创建日期: {order['created']}, 交付日期: {order.get('delivered', '待定')}"
                )
        if results:
            return "\n\n".join(results[:limit])
        return f"未找到与'{query}'匹配的订单信息"
    except Exception as e:
        return f"订单查询失败: {str(e)}"


def search_tickets(query: str, limit: int = 5) -> str:
    """查询公司IT支持工单信息，支持按工单编号、标题、申请人、处理状态进行搜索。

    Args:
        query: 查询关键词
        limit: 返回结果条数上限

    Returns:
        str: 匹配的工单信息文本
    """
    filepath = os.path.join(DATA_DIR, "tickets.json")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tickets = json.load(f)
        results = []
        query_lower = query.lower()
        for ticket in tickets:
            if (query_lower in ticket.get("ticket_id", "").lower()
                or query_lower in ticket.get("title", "").lower()
                or query_lower in ticket.get("applicant", "").lower()
                or query_lower in ticket.get("status", "").lower()
                or query_lower in ticket.get("assignee", "").lower()):
                results.append(
                    f"工单编号: {ticket['ticket_id']}, 标题: {ticket['title']}, "
                    f"状态: {ticket['status']}, 优先级: {ticket['priority']}, "
                    f"处理人: {ticket['assignee']}, 申请人: {ticket['applicant']}, "
                    f"创建时间: {ticket['created']}, 解决时间: {ticket.get('resolved', '未解决')}\n"
                    f"描述: {ticket['description']}"
                )
        if results:
            return "\n\n".join(results[:limit])
        return f"未找到与'{query}'匹配的工单信息"
    except Exception as e:
        return f"工单查询失败: {str(e)}"
