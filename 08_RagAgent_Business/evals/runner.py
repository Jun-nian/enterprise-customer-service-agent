# evals/runner.py
# 评测流程编排：初始化 LLM/工具/图 → 逐用例运行 → 采集轨迹 → 判定指标 → 输出报告
# 复刻 main.py lifespan 的初始化顺序，无需启动 FastAPI

import json
import os
import sys
import time
import uuid

# Windows 控制台默认 GBK，重配置 stdout 为 UTF-8 防止中文/符号打印报错
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from collections import Counter

from demoRagAgent import ToolConfig, create_graph
from utils.config import Config
from utils.llms import get_llm
from utils.tools_config import get_tools
from evals.dataset import load_test_cases
from evals.metrics import check_hit, hit_at_k
from evals.trace import collect_trace

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


def build_runtime():
    """复刻 main.py lifespan 初始化顺序，返回 (llm_chat, llm_embedding, graph, tool_config)"""
    llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)
    tools = get_tools(llm_embedding)
    tool_config = ToolConfig(tools)
    graph = create_graph(llm_chat, llm_embedding, tool_config)
    return llm_chat, llm_embedding, graph, tool_config


def _run_case(graph, llm_chat, case, judge_enabled: bool) -> dict:
    """运行单个评测用例，返回结果字典"""
    # 每条用例独立线程（thread_id 唯一），避免检查点污染
    config = {
        "configurable": {
            "thread_id": f"eval-{case.id}-{uuid.uuid4().hex[:8]}",
            "user_id": "eval",
        }
    }
    trace = collect_trace(graph, case.id, case.query, config)

    result = {
        "id": case.id,
        "tool_name": case.tool_name,
        "query": case.query,
        "golden_contains": case.golden_contains,
    }

    # 1) 检索命中判定：最后一个同名工具的输出是否包含金标准子串
    tool_output = trace.last_tool_output(case.tool_name)
    result["hit"] = check_hit(tool_output, case.golden_contains)
    result["tool_output"] = tool_output[:300]  # 截断存报告
    result["used_tools"] = trace.used_tools
    result["relevance_scores"] = trace.relevance_scores
    result["rewrite_count"] = trace.rewrite_count
    result["strategy"] = trace.strategy
    result["error"] = trace.error

    # 2) LLM-judge：回答是否覆盖金标准事实
    result["judge"] = None
    if judge_enabled and trace.final_answer and not trace.error:
        from evals.judge import judge_answer
        try:
            result["judge"] = judge_answer(llm_chat, case.query, trace.final_answer, case.golden_contains)
        except Exception as e:
            result["judge_error"] = str(e)
    result["final_answer"] = (trace.final_answer or "")[:300]

    return result


def _aggregate(results: list) -> dict:
    """按工具聚合指标"""
    agg = {"overall": {"total": 0, "hit": 0, "judge": 0, "judge_total": 0}}
    for r in results:
        key = r["tool_name"]
        if key not in agg:
            agg[key] = {"total": 0, "hit": 0, "judge": 0, "judge_total": 0}
        agg[key]["total"] += 1
        agg["overall"]["total"] += 1
        if r.get("hit"):
            agg[key]["hit"] += 1
            agg["overall"]["hit"] += 1
        if r.get("judge") is not None:
            agg[key]["judge_total"] += 1
            agg["overall"]["judge_total"] += 1
            if r["judge"]:
                agg[key]["judge"] += 1
                agg["overall"]["judge"] += 1

    for stats in agg.values():
        stats["hit_rate"] = (stats["hit"] / stats["total"]) if stats["total"] else 0.0
        stats["judge_rate"] = (stats["judge"] / stats["judge_total"]) if stats["judge_total"] else None
    return agg


def _print_summary(agg: dict, strategy_stats: Counter, judge_enabled: bool):
    """控制台输出摘要表格"""
    print("\n" + "=" * 62)
    print("评测报告摘要")
    print("=" * 62)
    print(f"{'工具':<20} {'用例数':>6} {'命中率':>8} {'LLM-judge通过率':>16}")
    print("-" * 62)
    for key in ["overall"] + [k for k in agg if k != "overall"]:
        s = agg[key]
        judge_str = f"{s['judge_rate']:.1%}" if s["judge_rate"] is not None else "-"
        label = "总体" if key == "overall" else key
        print(f"{label:<20} {s['total']:>6} {s['hit_rate']:.1%} {judge_str:>16}")
    if strategy_stats:
        print("-" * 62)
        print("策略分布:", dict(strategy_stats))
    if not judge_enabled:
        print("\n(LLM-judge 已跳过 --no-judge)")


def run_eval(llm_chat, llm_embedding, graph, tool_config, args) -> dict:
    """执行完整评测流程

    Args:
        llm_chat / llm_embedding / graph / tool_config: build_runtime() 产物
        args: argparse.Namespace（limit / cases / no_judge / strategy）

    Returns:
        dict: 完整报告（写 output/eval_report_<ts>.json）
    """
    cases = load_test_cases()
    if getattr(args, "cases", None):
        cases = [c for c in cases if c.tool_name in args.cases]
    if getattr(args, "limit", None):
        cases = cases[: args.limit]

    print(f"[eval] 加载 {len(cases)} 个用例，judge={'开' if not args.no_judge else '关'}")
    judge_enabled = not args.no_judge

    results = []
    strategy_stats = Counter()
    for i, case in enumerate(cases, 1):
        try:
            result = _run_case(graph, llm_chat, case, judge_enabled)
        except Exception as e:
            result = {
                "id": case.id, "tool_name": case.tool_name, "query": case.query,
                "hit": False, "judge": None, "error": str(e),
                "used_tools": [], "relevance_scores": [], "strategy": None,
            }
        results.append(result)
        if result.get("strategy"):
            strategy_stats[result["strategy"]] += 1
        # 用 ASCII 标记避免 Windows 控制台 GBK 编码报错
        hit_flag = "[OK]" if result["hit"] else "[X ]"
        judge_flag = ("[OK]" if result.get("judge") else "[X ]") if result.get("judge") is not None else "[--]"
        print(f"[{i}/{len(cases)}] {case.id:<16} {hit_flag} hit  {judge_flag} judge  tools={result['used_tools']}")
        if result.get("error"):
            print(f"         error: {result['error'][:200]}")

    agg = _aggregate(results)
    _print_summary(agg, strategy_stats, judge_enabled)

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "llm_type": Config.LLM_TYPE,
            "enable_search_agent": Config.ENABLE_SEARCH_AGENT,
            "strategy": getattr(args, "strategy", None),
            "no_judge": args.no_judge,
            "limit": getattr(args, "limit", None),
            "cases": getattr(args, "cases", None),
        },
        "aggregate": agg,
        "strategy_stats": dict(strategy_stats),
        "results": results,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(OUTPUT_DIR, f"eval_report_{ts}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[eval] 报告已写入: {report_path}")
    return report
