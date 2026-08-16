# evals/agent_eval.py
# P0-B 评测基准：Agent 行为层评测 runner
# 在分类评测集上运行完整 graph，计算 Agent 行为层指标，输出基准对比表。
#
# 设计（面试话术钩子）：
#   "在检索层 hit@k 之上，补齐 Agent 行为层指标：任务成功率、工具调用准确率、
#    多步完成率、无效 LLM 调用率、步数/延迟/token 成本，并支持多策略对比
#    （default 快路径 vs plan-execute 规划）产出基准表。"

import json
import os
import sys
import time
import uuid

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from collections import Counter

from evals.dataset.categorized import load_categorized_cases, AgentTestCase
from evals.trace import collect_trace, Trace
from evals.metrics import (
    task_success, tool_accuracy, multi_step_completion,
    invalid_llm_calls, avg_steps, avg_latency, avg_tokens,
)
from evals.judge import judge_answer
from evals.observability import with_observability, summarize_observer


def build_runtime(llm_chat, llm_embedding, graph, tool_config):
    """评测 runtime（复用外部已初始化的组件）"""
    return llm_chat, llm_embedding, graph, tool_config


def _run_single(graph, llm_chat, case: AgentTestCase, config: dict,
                observer, judge_enabled: bool = True, strategy_label: str = "") -> dict:
    """运行单个评测用例，返回结构化结果"""
    cfg = {
        "configurable": {
            "thread_id": f"agent-eval-{case.id}-{uuid.uuid4().hex[:6]}",
            "user_id": "agent-eval",
        }
    }
    cfg["configurable"].update(config.get("configurable", {}))
    trace = collect_trace(graph, case.id, case.query, cfg)

    result = {
        "id": case.id,
        "category": case.category,
        "query": case.query,
        "expected_tools": case.expected_tools,
        "used_tools": trace.used_tools,
        "tool_call_count": trace.tool_call_count(),
        "node_count": trace.node_count,
        "strategy": strategy_label,
        "plan_mode": None,
        "total_latency_ms": trace.total_latency_ms,   # P0-B 成本指标
        "total_tokens": trace.total_tokens,
        "llm_calls": trace.llm_calls,
    }

    # 判定
    result["hit"] = False
    if case.golden_contains:
        # 对最后一条工具输出做 hit 判定（多工具用汇总）
        last_out = trace.last_tool_output(case.expected_tools[-1]) if case.expected_tools else ""
        result["hit"] = any(
            _substr_contains(trace.final_answer or "", g)
            for g in case.golden_contains
        ) or any(
            _substr_contains(last_out, g) for g in case.golden_contains
        )

    # LLM-judge 任务成功率
    result["judge"] = None
    if judge_enabled and trace.final_answer and not trace.error:
        try:
            result["judge"] = judge_answer(llm_chat, case.query, trace.final_answer, case.golden_contains)
        except Exception as e:
            result["judge_error"] = str(e)
    result["final_answer"] = (trace.final_answer or "")[:300]
    result["error"] = trace.error

    return result


def _substr_contains(text: str, sub: str) -> bool:
    """去除空白后的子串包含判定"""
    import re
    return bool(sub) and re.sub(r"\s+", "", sub) in re.sub(r"\s+", "", text or "")


def run_agent_eval(graph, llm_chat, cases=None, judge_enabled=True,
                   observer=None, strategy_label="default", config=None,
                   max_cases: int = None) -> dict:
    """执行 Agent 行为层评测

    Args:
        graph: 编译后的 LangGraph（已按策略构建）
        llm_chat: 聊天模型
        cases: 评测用例（默认加载全部 5 类）
        judge_enabled: 是否启用 LLM-judge
        observer: 可观测器（记录 LLM 调用成本）
        strategy_label: 当前策略标签（default/plan-execute/react）
        config: 额外 runtime config
        max_cases: 评测用例数上限

    Returns:
        dict: 完整评测报告（含指标 + 基准表数据）
    """
    cases = cases or load_categorized_cases()
    if max_cases:
        cases = cases[:max_cases]

    results = []
    for case in cases:
        try:
            r = _run_single(graph, llm_chat, case, config or {}, observer, judge_enabled, strategy_label)
        except Exception as e:
            r = {
                "id": case.id, "category": case.category, "query": case.query,
                "expected_tools": case.expected_tools, "used_tools": [],
                "tool_call_count": 0, "node_count": 0, "strategy": strategy_label,
                "hit": False, "judge": None, "final_answer": "", "error": str(e),
                "total_latency_ms": 0.0, "total_tokens": 0, "llm_calls": 0,
            }
        results.append(r)

    # ---- 聚合指标 ----
    traces = [_result_to_trace(r) for r in results]
    judge_results = [r.get("judge") for r in results]
    task_rate, task_ok, task_total = task_success(judge_results)
    tool_rate, tool_ok, tool_total = tool_accuracy(
        [r["used_tools"] for r in results],
        [r["expected_tools"] for r in results],
    )
    ms_rate, ms_ok, ms_total = multi_step_completion(
        [c for c in cases if c.category in ("multi_tool", "cross_source")],
        [t for c, t in zip(cases, traces) if c.category in ("multi_tool", "cross_source")],
    )
    inval_rate, inval_cnt, inval_total = invalid_llm_calls(cases, traces)

    metrics = {
        "strategy": strategy_label,
        "cases_total": len(results),
        "task_success_rate": round(task_rate, 4),
        "tool_accuracy": round(tool_rate, 4),
        "multi_step_completion_rate": round(ms_rate, 4),
        "invalid_llm_call_rate": round(inval_rate, 4),
        "avg_steps": round(avg_steps(traces), 2),
        "avg_latency_ms": round(avg_latency(traces), 1),
        "avg_tokens": round(avg_tokens(traces), 1),
        "hit_rate": round(sum(r["hit"] for r in results) / len(results), 4) if results else 0.0,
        "raw_counts": {
            "task_ok": task_ok, "task_total": task_total,
            "tool_ok": tool_ok, "tool_total": tool_total,
            "multi_ok": ms_ok, "multi_total": ms_total,
            "invalid": inval_cnt,
        },
    }

    # 可观测层成本
    if observer is not None:
        metrics["cost"] = summarize_observer(observer)

    return {
        "strategy": strategy_label,
        "metrics": metrics,
        "results": results,
        "by_category": _aggregate_by_category(results),
    }


def _result_to_trace(r: dict) -> Trace:
    """将 result 字典转回轻量 Trace（供指标复用）"""
    t = Trace(case_id=r["id"])
    t.used_tools = list(r.get("used_tools", []))
    t.final_answer = r.get("final_answer") or None
    t.error = r.get("error")
    t.total_latency_ms = r.get("total_latency_ms", 0.0)
    t.total_tokens = r.get("total_tokens", 0)
    t.llm_calls = r.get("llm_calls", 0)
    # node_count 需回填（result 里已存）
    for _ in range(r.get("node_count", 0)):
        from evals.trace import StepRecord
        t.steps.append(StepRecord(node=""))
    return t


def _aggregate_by_category(results: list) -> dict:
    """按类别聚合工具调用准确率，形成分类对比表"""
    from collections import defaultdict
    agg = defaultdict(lambda: {"total": 0, "tool_ok": 0, "hit": 0})
    for r in results:
        a = agg[r["category"]]
        a["total"] += 1
        exp = set(r["expected_tools"])
        used = set(r["used_tools"])
        if (exp and exp.issubset(used)) or (not exp and not used):
            a["tool_ok"] += 1
        if r.get("hit"):
            a["hit"] += 1
    return {k: {**v, "tool_accuracy": round(v["tool_ok"] / v["total"], 4) if v["total"] else 0.0,
                 "hit_rate": round(v["hit"] / v["total"], 4) if v["total"] else 0.0}
            for k, v in agg.items()}


def export_benchmark(rows: list, output_path: str) -> str:
    """导出多策略基准对比表（markdown）"""
    if not rows:
        return ""
    header = ["策略", "用例数", "任务成功率", "工具准确率", "多步完成率", "无效调用率", "平均步数", "平均延迟(ms)", "平均Token"]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        m = r["metrics"]
        lines.append(
            f"| {r['strategy']} | {m['cases_total']} | "
            f"{m['task_success_rate']:.1%} | {m['tool_accuracy']:.1%} | "
            f"{m['multi_step_completion_rate']:.1%} | {m['invalid_llm_call_rate']:.1%} | "
            f"{m['avg_steps']} | {m['avg_latency_ms']:.0f} | {m['avg_tokens']:.0f} |"
        )
    md = "\n".join(lines)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Agent 行为层基准对比表\n\n" + md + "\n")
    return md
