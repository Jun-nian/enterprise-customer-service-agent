# rl/search_agent.py
# search_agent 节点逻辑：bandit 选检索策略，产出带 tool_calls 的 AIMessage
# 三种策略：
#   - default:    复用 rewrite prompt 重写查询 → 单次检索
#   - multi_query:原查询 + rewrite 变体两个 tool_calls（ParallelToolNode 天然并行，结果合并）
#   - top_k:      retrieve 调 n_results=10（提高召回，误召由 grade 抑制）
# 奖励回填由 grade_documents 节点调用 update_bandit 完成（1 轮延迟）

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_core.messages import AIMessage

from demoRagAgent import create_chain, get_latest_question
from utils.config import Config

# 检索类工具：需要 grade 评分的工具集合（与 demoRagAgent.ToolConfig 保持一致）
RETRIEVAL_TOOLS = {"retrieve", "search_employees", "search_faq", "search_orders", "search_tickets"}


def _rewrite_query(llm_chat, question: str) -> str:
    """复用 rewrite prompt 重写查询，返回改进后的查询"""
    try:
        chain = create_chain(llm_chat, Config.PROMPT_TEMPLATE_TXT_REWRITE)
        resp = chain.invoke({"question": question})
        return str(resp.content if hasattr(resp, "content") else resp).strip()
    except Exception as e:
        print(f"[search_agent] 重写失败，使用原查询: {e}")
        return question


def search_agent(state: dict, llm_chat, tool_config, bandit, force_strategy: str = None) -> dict:
    """search_agent 节点：选择策略并产出带 tool_calls 的 AIMessage

    该节点不直接执行工具；产出 AIMessage 后由既有 call_tools 节点执行。

    Args:
        state: 当前状态（含 messages、search_attempts）
        llm_chat: 聊天模型
        tool_config: ToolConfig 实例（含工具定义，用于生成 tool_calls）
        bandit: EpsilonGreedyBandit 实例（或 None，此时固定 default）
        force_strategy: 固定策略（离线评测用），优先级高于 bandit

    Returns:
        dict: 状态更新（messages + search_attempts + selected_strategy）
    """
    question = get_latest_question(state)
    if not question:
        return {"messages": [AIMessage(content="无法获取问题，请重试")],
                "selected_strategy": "default"}

    # 决策策略：固定 > bandit > default
    if force_strategy:
        strategy = force_strategy
    elif bandit is not None:
        strategy = bandit.choose()
    else:
        strategy = "default"

    search_attempts = state.get("search_attempts", 0) + 1

    # 判断上一步检索用的是哪个工具（从最后一条 ToolMessage 推断）
    last_tool = None
    for msg in reversed(state.get("messages", [])):
        if getattr(msg, "name", None):
            last_tool = msg.name
            break
    target_tool = last_tool if last_tool in RETRIEVAL_TOOLS else "retrieve"

    # 生成 tool_calls（模拟 LLM 输出的工具调用，ID 用 uuid 保证唯一）
    import uuid
    rewritten = _rewrite_query(llm_chat, question)

    if strategy == "multi_query":
        # 原查询 + 变体两个 tool_call（ParallelToolNode 并行执行、结果合并到 messages）
        tool_calls = [
            {"name": target_tool, "args": {"query": question}, "id": f"call_{uuid.uuid4().hex[:8]}"},
            {"name": target_tool, "args": {"query": rewritten}, "id": f"call_{uuid.uuid4().hex[:8]}"},
        ]
    elif strategy == "top_k":
        # 提高检索召回：n_results=10（仅 retrieve 支持该参数，其他工具忽略）
        args = {"query": rewritten, "n_results": 10} if target_tool == "retrieve" else {"query": rewritten}
        tool_calls = [{"name": target_tool, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}]
    else:  # default
        tool_calls = [{"name": target_tool, "args": {"query": rewritten}, "id": f"call_{uuid.uuid4().hex[:8]}"}]

    # 产出带 tool_calls 的 AIMessage（无内容文本，纯工具调用）
    ai_msg = AIMessage(content="", tool_calls=tool_calls)

    return {
        "messages": [ai_msg],
        "search_attempts": search_attempts,
        "selected_strategy": strategy,
    }


def update_bandit(bandit, strategy: str, score: str):
    """grade 节点回调：把评分结果作为奖励回填给 bandit

    Args:
        bandit: EpsilonGreedyBandit 实例
        strategy: 本次实际执行的策略名
        score: grade 的 binary_score（"yes"/"no"）
    """
    if bandit is None or not strategy:
        return
    reward = 1.0 if str(score).lower() == "yes" else 0.0
    bandit.update(strategy, reward)
    from rl.bandit import save_bandit
    save_bandit(bandit, Config.BANDIT_STATE_FILE)
