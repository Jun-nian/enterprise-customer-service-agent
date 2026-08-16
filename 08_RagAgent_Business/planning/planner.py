# planning/planner.py
# P0-A 规划范式：Planner 节点
# 将用户问题分解为结构化计划（步骤列表），并判定复杂度：
#   - simple  → 走原快路径（agent 单轮工具决策）
#   - complex → 走 Plan-and-Execute（executor 逐步执行 + critic 评估）
#
# 设计目标（面试话术钩子）：
#   "把固定 DAG 升级为可自主规划/重规划的 Agent：简单查询零成本快路径，
#    复杂跨源查询自动拆解为多步计划并动态重规划。"

import logging
import sys
import os
from typing import List, Optional

from pydantic import BaseModel, Field

# 延迟导入 demoRagAgent 以复用 create_chain / get_latest_question / Config，
# 同时避免 planning → demoRagAgent → planning 的循环导入
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)


class PlanStep(BaseModel):
    """规划中的单个执行步骤"""
    tool: str = Field(description="要调用的工具名，如 retrieve/search_employees/search_faq/search_orders/search_tickets")
    query: str = Field(description="传给工具的查询关键词")
    goal: str = Field(description="本步骤要解决的目标（一句话，供 critic 评估）")
    n_results: Optional[int] = Field(default=None, description="retrieve 工具的返回条数（可选）")
    limit: Optional[int] = Field(default=None, description="结构化工具的结果条数上限（可选）")


class AgentPlan(BaseModel):
    """Planner 的结构化输出"""
    complexity: str = Field(description="复杂度：simple=单工具/直接回答，complex=需要多步/多工具协作")
    steps: List[PlanStep] = Field(default_factory=list, description="执行步骤列表（simple 时可为空）")


# 判定"是否可能为复杂查询"的轻量规则（先于 LLM 调用，省成本）
# 命中这些信号即直接走 Plan-and-Execute，避免不必要的 LLM 复杂度判断
_COMPLEXITY_HINTS = [
    "并且", "同时", "对比", "比较", "分别", "以及",
    "所有", "全部", "汇总", "统计", "列出", "多少", "几", "哪些",
]


def _rule_based_complexity(question: str) -> Optional[str]:
    """规则预判复杂度：命中强信号返回 'complex'，否则返回 None（交给 LLM 判断）"""
    q = question.strip()
    # 列举/汇总/对比型问题：倾向于多步
    if any(h in q for h in _COMPLEXITY_HINTS):
        return "complex"
    # 单实体精确查询（姓名/编号/工单号）：明确简单，直接快路径
    import re
    if re.search(r"(员工|姓名|工号|订单号|工单号|FAQ|制度)", q):
        return "simple"
    return None


def planner_node(state: dict, llm_chat) -> dict:
    """Planner 节点：产出结构化计划，返回 plan + plan_mode + plan_index

    Args:
        state: 当前状态（含 messages、plan_attempts）
        llm_chat: 聊天模型

    Returns:
        dict: 状态更新（plan / plan_mode / plan_index / plan_attempts）
    """
    from demoRagAgent import get_latest_question, create_chain
    from utils.config import Config

    question = get_latest_question(state) or ""
    logger.info(f"[planner] 用户问题: {question}")

    # 重规划上限控制（避免无限 replan）
    attempts = int(state.get("plan_attempts", 0) or 0)
    if attempts >= Config.PLAN_MAX_REPLAN:
        logger.warning(f"[planner] 已达重规划上限({Config.PLAN_MAX_REPLAN})，走快路径兜底")
        return {
            "plan_mode": "simple",
            "plan": [],
            "plan_index": 0,
            "plan_attempts": attempts,
        }

    # 1) 规则预判（零成本）
    rule = _rule_based_complexity(question)
    if rule == "simple":
        logger.info("[planner] 规则判定: simple → 快路径")
        return {
            "plan_mode": "simple",
            "plan": [],
            "plan_index": 0,
            "plan_attempts": attempts,
        }

    # 2) LLM 结构化规划
    try:
        from planning.planner import AgentPlan
        plan_chain = create_chain(llm_chat, Config.PROMPT_TEMPLATE_TXT_PLANNER, AgentPlan)
        result = plan_chain.invoke({"question": question})

        if rule == "complex" or result.complexity.lower() == "complex":
            steps = [s.model_dump() for s in result.steps]
            logger.info(f"[planner] LLM 规划: complex, {len(steps)} 步 → {steps}")
            return {
                "plan_mode": "complex",
                "plan": steps,
                "plan_index": 0,
                "plan_attempts": attempts,
            }
        logger.info("[planner] LLM 判定: simple → 快路径")
        return {
            "plan_mode": "simple",
            "plan": [],
            "plan_index": 0,
            "plan_attempts": attempts,
        }
    except Exception as e:
        logger.error(f"[planner] 规划失败，走快路径兜底: {e}")
        return {
            "plan_mode": "simple",
            "plan": [],
            "plan_index": 0,
            "plan_attempts": attempts,
        }


def route_after_planner(state: dict) -> str:
    """Planner 后路由：simple → agent（快路径），complex → executor（多步执行）"""
    mode = state.get("plan_mode", "simple")
    return "agent" if mode == "simple" else "executor"
