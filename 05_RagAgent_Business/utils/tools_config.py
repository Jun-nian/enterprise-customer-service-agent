from langchain_chroma import Chroma
from langchain_core.tools import create_retriever_tool
from langchain_core.tools import tool
from .config import Config

# 从 mcp_server/tools_impl.py 导入共享工具实现（单一事实来源）
import sys
import os
_mcp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mcp_server")
if _mcp_dir not in sys.path:
    sys.path.insert(0, _mcp_dir)

from tools_impl import (
    search_employees as _search_employees,
    search_faq as _search_faq,
    search_orders as _search_orders,
    search_tickets as _search_tickets,
)


def get_tools(llm_embedding):
    """
    创建并返回企业智能客服工具列表。
    工具核心实现位于 mcp_server/tools_impl.py，此处仅做 LangChain @tool 装饰器包装。

    Args:
        llm_embedding: 嵌入模型实例，用于初始化向量存储。

    Returns:
        list: 工具列表。
    """
    # 创建Chroma向量存储实例
    vectorstore = Chroma(
        persist_directory=Config.CHROMADB_DIRECTORY,
        collection_name=Config.CHROMADB_COLLECTION_NAME,
        embedding_function=llm_embedding,
    )
    # 将向量存储转换为检索器
    retriever = vectorstore.as_retriever()
    # 创建企业知识库检索工具
    retriever_tool = create_retriever_tool(
        retriever,
        name="retrieve",
        description="这是企业知识库查询工具。搜索并返回与用户问题相关的企业知识库信息，包括公司制度、产品文档、技术方案等内容。"
    )

    # 以下工具从 mcp_server/tools_impl.py 导入核心实现，仅包装 LangChain @tool 装饰器

    @tool
    def search_employees(query: str) -> str:
        """查询公司员工信息，支持按姓名、部门、职位进行搜索。输入员工姓名、部门名称或职位关键词，返回匹配的员工信息。"""
        return _search_employees(query)

    @tool
    def search_faq(query: str) -> str:
        """查询公司常见问题(FAQ)，涵盖入职、考勤、报销、IT支持、HR政策、行政服务、员工福利等类别。输入问题关键词，返回匹配的FAQ条目。"""
        return _search_faq(query)

    @tool
    def search_orders(query: str) -> str:
        """查询公司销售订单信息，支持按客户名称、订单编号、销售代表、订单状态进行搜索。输入关键词，返回匹配的订单详情。"""
        return _search_orders(query)

    @tool
    def search_tickets(query: str) -> str:
        """查询公司IT支持工单信息，支持按工单编号、标题、申请人、处理状态进行搜索。输入关键词，返回匹配的工单详情。"""
        return _search_tickets(query)

    # 返回企业智能客服工具列表
    return [retriever_tool, search_employees, search_faq, search_orders, search_tickets]
