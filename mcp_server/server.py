# mcp_server/server.py
# 企业智能客服 MCP 服务器入口
# 使用 FastMCP + stdio 传输，供 Claude Code 等 MCP 客户端调用

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP
from tools_impl import retrieve, search_employees, search_faq, search_orders, search_tickets

mcp = FastMCP(
    "enterprise-customer-service",
    instructions="企业智能客服 MCP 服务器 - 提供员工查询、FAQ搜索、订单查询、工单查询、知识库检索等工具"
)


@mcp.tool()
def retrieve_tool(query: str) -> str:
    """搜索企业知识库（ChromaDB）。用于查询公司制度、产品文档、技术方案、员工手册等内容。
    当用户问题涉及公司政策、流程规定、产品信息时使用此工具。"""
    return retrieve(query)


@mcp.tool()
def search_employees_tool(query: str) -> str:
    """查询公司员工信息。支持按姓名、部门名称、职位关键词搜索。
    输入员工姓名、部门或职位，返回匹配员工的联系方式、部门、上级等信息。"""
    return search_employees(query)


@mcp.tool()
def search_faq_tool(query: str) -> str:
    """查询公司常见问题(FAQ)。涵盖入职、考勤、报销、IT支持、HR政策、行政服务、员工福利等类别。
    输入问题关键词，返回匹配的FAQ条目和答案。"""
    return search_faq(query)


@mcp.tool()
def search_orders_tool(query: str) -> str:
    """查询公司销售订单信息。支持按客户名称、订单编号、销售代表、订单状态搜索。
    输入关键词，返回匹配订单的详细信息。"""
    return search_orders(query)


@mcp.tool()
def search_tickets_tool(query: str) -> str:
    """查询公司IT支持工单信息。支持按工单编号、标题、申请人、处理状态、处理人搜索。
    输入关键词，返回匹配工单的详细信息和处理进度。"""
    return search_tickets(query)


if __name__ == "__main__":
    mcp.run(transport="stdio")
