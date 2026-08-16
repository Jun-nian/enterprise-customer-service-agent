# planning/executor.py
# P0-A 规划范式：Executor 节点
# 按 planner 产出的计划逐步执行：每步把 plan 步骤转为一次工具调用（tool_call），
# 由既有的 call_tools 节点执行并回填 observation 到消息历史。

import logging
import os
import sys
import uuid

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


def executor_node(state: dict, tool_config) -> dict:
    """Executor 节点：执行计划中的当前步骤

    从 state.plan[plan_index] 取出当前步骤，构造带 tool_calls 的 AIMessage，
    交给 call_tools 节点执行；同时推进 plan_index，并把观察结果（observation）
    累计到 plan 状态。

    Args:
        state: 当前状态（含 plan / plan_index / plan 观察）
        tool_config: 工具配置（用于校验工具名 + 汇总工具输出到 observation）

    Returns:
        dict: 状态更新（messages + plan_index + plan_observations）
    """
    plan = state.get("plan") or []
    idx = int(state.get("plan_index") or 0)

    # 防御：计划为空或已执行完 → 空消息，路由到 critic 判定结束
    if idx >= len(plan):
        logger.warning(f"[executor] 计划已执行完 (index={idx}/{len(plan)})")
        return {"plan_index": idx}

    step = plan[idx]
    tool_name = step.get("tool", "")
    allowed = tool_config.get_tool_names() if hasattr(tool_config, "get_tool_names") else set()
    if tool_name not in allowed:
        logger.warning(f"[executor] 步骤工具 '{tool_name}' 不在可用集合，跳过")
        return {
            "messages": [AIMessage(content=f"计划步骤引用了不可用工具: {tool_name}")],
            "plan_index": idx + 1,
        }

    # 构造 tool_call 参数（透传 plan 中的可选参数）
    args = {"query": step.get("query", "")}
    if step.get("n_results"):
        args["n_results"] = step["n_results"]
    if step.get("limit"):
        args["limit"] = step["limit"]

    tool_call = {
        "name": tool_name,
        "args": args,
        "id": f"call_plan_{uuid.uuid4().hex[:8]}",
    }
    logger.info(f"[executor] 执行计划步骤[{idx}]: {tool_name}({args}) 目标: {step.get('goal','')}")
    ai_msg = AIMessage(content="", tool_calls=[tool_call])

    return {
        "messages": [ai_msg],
        "plan_index": idx + 1,
    }
