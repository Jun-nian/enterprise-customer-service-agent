# planning/critic.py
# P0-A 规划范式：Critic 节点
# 评估 executor 步骤的执行结果：
#   - 计划全部执行完 → generate（汇总生成最终回答）
#   - 步骤失败且未达重规划上限 → 回 planner 重规划（动态 replan）
#   - 还有剩余步骤且成功 → executor（继续下一步）
#
# 默认使用规则评估（零 LLM 成本）：检查工具输出是否含错误标记。
# 可选 LLM 评估（Config.CRITIC_USE_LLM=true）对复杂目标做语义满足度判断。

import logging
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger(__name__)

# 工具输出中的错误标记（与 tools_impl.py 的失败文案对齐）
_ERROR_MARKERS = [
    "Error", "失败", "错误", "exception", "traceback",
    "未找到", "无法", "未知工具", "not found",
]
# 明确视为"成功但无结果"的标记（不算失败，但不算有效证据）
_EMPTY_MARKERS = ["未找到相关", "未找到与", "未找到匹配"]


def _extract_last_tool_result(state: dict) -> tuple:
    """从消息历史末尾提取最近一次工具执行结果

    Returns:
        (tool_name, tool_output, is_tool_msg)
    """
    messages = state.get("messages") or []
    for msg in reversed(messages):
        name = getattr(msg, "name", None)
        if name:
            return name, str(getattr(msg, "content", "")), True
    return "", "", False


def _is_failure(tool_name: str, output: str) -> bool:
    """判断工具输出是否表示失败"""
    text = output.strip()
    if not text:
        return True  # 空输出视为失败（无证据）
    # 明确错误标记
    for m in _ERROR_MARKERS:
        if m in text[:80] and not any(e in text[:80] for e in _EMPTY_MARKERS):
            return True
    return False


def _is_empty_result(tool_name: str, output: str) -> bool:
    """判断是否为空结果（未命中，但非系统错误）"""
    return any(e in output for e in _EMPTY_MARKERS)


def critic_node(state: dict, llm_chat=None, tool_config=None) -> dict:
    """Critic 节点：评估计划执行进度，返回下一步决策

    Args:
        state: 当前状态（含 plan / plan_index / plan_observations / plan_attempts）
        llm_chat: 可选 LLM（用于语义满足度评估，默认规则评估）
        tool_config: 可选工具配置

    Returns:
        dict: 状态更新（plan_observations + next + plan_attempts）
    """
    from utils.config import Config

    plan = state.get("plan") or []
    idx = int(state.get("plan_index") or 0)
    attempts = int(state.get("plan_attempts") or 0)
    observations = list(state.get("plan_observations") or [])

    tool_name, tool_output, is_tool = _extract_last_tool_result(state)

    # 累计当前步骤的观察结果
    if is_tool:
        obs_entry = {
            "step_index": max(0, idx - 1),
            "tool": tool_name,
            "output": tool_output[:2000],          # 截断防审计膨胀
            "ok": not _is_failure(tool_name, tool_output),
            "empty": _is_empty_result(tool_name, tool_output),
        }
        # 防重复追加（同一 tool_call 多次触发 critic）
        if not observations or observations[-1].get("tool") != tool_name or \
           observations[-1].get("output") != tool_output:
            observations.append(obs_entry)
        logger.info(f"[critic] 步骤观察: {obs_entry['tool']} ok={obs_entry['ok']} empty={obs_entry['empty']}")

    all_done = idx >= len(plan)
    failed = not is_tool or (observations and not observations[-1].get("ok", True))

    # 决策
    if all_done:
        logger.info(f"[critic] 计划执行完毕 ({idx}/{len(plan)}) → generate")
        return {"plan_observations": observations, "next": "generate", "plan_attempts": attempts}
    if failed and attempts < Config.PLAN_MAX_REPLAN:
        logger.info(f"[critic] 步骤失败且未达重规划上限 ({attempts}/{Config.PLAN_MAX_REPLAN}) → replan")
        return {"plan_observations": observations, "next": "replan", "plan_attempts": attempts + 1}
    if failed:
        logger.warning(f"[critic] 步骤失败但已达重规划上限 → 降级 generate")
        return {"plan_observations": observations, "next": "generate", "plan_attempts": attempts}
    logger.info(f"[critic] 步骤成功，继续下一步 ({idx}/{len(plan)}) → executor")
    return {"plan_observations": observations, "next": "executor", "plan_attempts": attempts}


def route_after_critic(state: dict) -> str:
    """Critic 后路由：executor / generate / planner（重规划）"""
    nxt = state.get("next", "generate")
    # 规范化 next 值
    if nxt == "replan":
        return "planner"
    if nxt in ("executor", "generate", "planner"):
        return nxt
    return "generate"
