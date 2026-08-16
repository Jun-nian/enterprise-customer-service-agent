# multiagent/orchestrator.py
# P1-C 多智能体协作编排器：Router → 领域子 Agent → Planner/Executor/Critic
# 提供一个独立于主工作流的可演示入口，展示多智能体如何分工协作。
#
# 协作流程：
#   1. RoleRouter 根据用户意图路由到领域子 Agent（HR/IT/业务/知识库）
#   2. 领域子 Agent 限定工具子集（减少越权面 + 提升专注度）
#   3. Planner/Executor/Critic 三元组完成规划-执行-评估闭环
#
# 该编排器作为 08 版"多智能体协作"的独立能力展示；主工作流的规划节点
# 已通过 planning/ 模块实现等价角色化（见 demoRagAgent.create_graph）。

import logging
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from multiagent.roles import (
    route_to_domain, get_domain_tools, build_agent_context,
    ROLE_PLANNER, ROLE_EXECUTOR, ROLE_CRITIC,
)

logger = logging.getLogger(__name__)


def domain_agent_invoke(query: str, llm_chat, all_tools) -> dict:
    """领域子 Agent 执行：路由领域 + 限定工具 + 角色化调用

    Args:
        query: 用户问题
        llm_chat: 聊天模型
        all_tools: 全部工具列表

    Returns:
        dict: {domain_agent, allowed_tools, role_trace, tool_result}
    """
    # 1. Router 路由领域
    ctx = build_agent_context(query)
    domain_agent = ctx.domain_agent
    logger.info(f"[orchestrator] Router 路由 → {domain_agent}")

    # 2. 领域子 Agent 限定工具子集
    domain_tools = get_domain_tools(domain_agent, all_tools)
    tool_names = [t.name for t in domain_tools]
    logger.info(f"[orchestrator] {domain_agent} 工具子集: {tool_names}")

    # 3. 角色化执行：Executor 用领域工具调用
    ctx.enter(ROLE_EXECUTOR.name)
    result_parts = []
    # 若路由到知识库：直接检索
    if domain_agent == "knowledge_agent":
        from demoRagAgent import create_chain
        from utils.config import Config
        chain = create_chain(llm_chat, Config.PROMPT_TEMPLATE_TXT_AGENT)
        resp = chain.invoke({"question": query, "messages": "", "userInfo": ""})
        result_parts.append(str(resp.content if hasattr(resp, "content") else resp))
    else:
        # 对领域工具逐一尝试（演示角色化工具调用）
        for t in domain_tools:
            try:
                r = t.invoke({"query": query})
                if r and not str(r).startswith("未找到"):
                    result_parts.append(f"[{t.name}] {r}")
            except Exception as e:
                logger.warning(f"[orchestrator] 工具 {t.name} 调用失败: {e}")

    ctx.enter(ROLE_CRITIC.name)
    # Critic 评估：是否有结果
    ok = bool(result_parts)
    ctx.enter(ROLE_PLANNER.name)

    return {
        "domain_agent": domain_agent,
        "allowed_tools": tool_names,
        "role_trace": ctx.role_trace,
        "tool_result": "\n\n".join(result_parts) if result_parts else "未检索到相关信息",
        "success": ok,
    }


def run_multiagent_demo(query: str, llm_chat, all_tools) -> dict:
    """多智能体协作演示入口"""
    return domain_agent_invoke(query, llm_chat, all_tools)
