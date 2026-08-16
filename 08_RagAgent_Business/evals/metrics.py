# evals/metrics.py
# 评测指标：
#   - 检索层：hit@k（工具输出含金标准子串）
#   - Agent 行为层（P0-B）：
#       task_success_rate   任务成功率（LLM-judge 判定回答覆盖金标准）
#       tool_accuracy       工具调用准确率（期望工具是否被正确调用）
#       multi_step_rate     多步完成率（complex 用例是否走了多步规划且完成）
#       invalid_llm_rate    无效 LLM 调用率（未调用工具却产出无意义回答 / 无解用例误答）
#       step_latency_token  步数 / 延迟 / token 成本（可观测层 Trace 数据）
# 纯字符串/数值计算，不调用 LLM（LLM 判定在 judge.py）

import re
from typing import List, Optional


def _strip_ws(text: str) -> str:
    """去除所有空白字符"""
    return re.sub(r"\s+", "", text)


# ---------- 检索层指标 ----------

def check_hit(tool_output: str, golden_contains: List[str]) -> bool:
    """判断工具输出是否命中任一金标准子串"""
    if not tool_output or not golden_contains:
        return False
    normalized = _strip_ws(str(tool_output))
    return any(_strip_ws(g) in normalized for g in golden_contains)


def hit_at_k(samples: List[bool]) -> float:
    """计算命中率"""
    if not samples:
        return 0.0
    return sum(samples) / len(samples)


def accuracy(samples: List[bool]) -> float:
    """通过率（与 hit_at_k 同义，用于 judge 结果）"""
    return hit_at_k(samples)


# ---------- Agent 行为层指标（P0-B） ----------

def task_success(judge_results: List[Optional[bool]]) -> tuple:
    """任务成功率：judge 通过（True）的比例
    返回 (rate, ok_count, total)
    """
    judged = [r for r in judge_results if r is not None]
    total = len(judged)
    if total == 0:
        return 0.0, 0, 0
    ok = sum(1 for r in judged if r)
    return ok / total, ok, total


def tool_accuracy(trace_used_tools: List[List[str]], expected_tools: List[List[str]]) -> tuple:
    """工具调用准确率：每个用例期望工具被正确调用（命中 / 期望）
    返回 (rate, ok_count, total_cases)
    """
    total = len(expected_tools)
    if total == 0:
        return 0.0, 0, 0
    ok = 0
    for used, expected in zip(trace_used_tools, expected_tools):
        used_set = set(used)
        exp_set = set(expected)
        # 期望工具非空：全部命中即准确
        if exp_set and exp_set.issubset(used_set):
            ok += 1
        # 期望工具为空（无解/问候）：未调用任何工具即准确
        elif not exp_set and not used_set:
            ok += 1
        # 其他情况：部分命中或误用 → 不准确
    return ok / total, ok, total


def multi_step_completion(cases: List, traces: List) -> tuple:
    """多步完成率：complex 用例成功完成多步执行的比例
    判定：期望多工具用例 → 实际调用了 >1 个不同工具（跨步）且最终有回答
    """
    total = 0
    ok = 0
    for case, trace in zip(cases, traces):
        if case.complexity_hint != "complex":
            continue
        total += 1
        used_set = set(trace.used_tools)
        has_answer = bool(trace.final_answer)
        if len(used_set) >= 2 and has_answer:
            ok += 1
        # 单工具但需要多步规划的情形（如聚合统计）
        elif len(trace.steps) >= 3 and has_answer:
            ok += 1
    return (ok / total, ok, total) if total else (0.0, 0, 0)


def invalid_llm_calls(cases: List, traces: List) -> tuple:
    """无效 LLM 调用率：
       - 无解用例（no_solution）：期望拒答，若产生大段虚构回答则为无效
       - 所有用例：未调用工具却生成回答（脱靶）的比例
    返回 (rate, invalid_count, total_cases)
    """
    total = len(cases)
    if total == 0:
        return 0.0, 0, 0
    invalid = 0
    for case, trace in zip(cases, traces):
        used = set(trace.used_tools)
        # 无解且期望无工具：若回答了（而非拒答）视为无效
        if case.category == "no_solution" and not case.expected_tools:
            if trace.final_answer and trace.final_answer != "无法生成回复":
                invalid += 1
            continue
        # 期望有工具但未调用任何工具 → 无效 LLM 调用（LLM 空转）
        if case.expected_tools and not used:
            invalid += 1
    return invalid / total, invalid, total


def avg_steps(traces: List) -> float:
    """平均执行步数（节点数）"""
    if not traces:
        return 0.0
    return sum(t.node_count for t in traces) / len(traces)


def avg_latency(traces: List) -> float:
    """平均延迟（毫秒）"""
    if not traces:
        return 0.0
    return sum(t.total_latency_ms for t in traces) / len(traces)


def avg_tokens(traces: List) -> float:
    """平均 token 消耗"""
    if not traces:
        return 0.0
    return sum(t.total_tokens for t in traces) / len(traces)
