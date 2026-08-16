# eval_rag.py
# 评测体系 CLI 入口：对测试集跑完整 graph 流程，输出 hit@k + LLM-judge + Agent 行为层报告
# 用法示例:
#   python eval_rag.py --limit 20                  # 纯检索评测（默认含 LLM-judge，可加 --no-judge）
#   python eval_rag.py --no-judge                  # 跳过 LLM-judge（更快）
#   python eval_rag.py --strategy default          # RL 固定 default 策略（基线）
#   python eval_rag.py --strategy auto             # RL bandit 在线学习（需 ENABLE_SEARCH_AGENT=true）
#   python eval_rag.py --cases search_faq          # 只评测 FAQ 用例
#   python eval_rag.py --agent-eval                # Agent 行为层评测（P0-B，5 类分类评测集）
#   python eval_rag.py --agent-eval --no-judge     # Agent 行为层评测（跳过 LLM-judge，更快）
#   python eval_rag.py --agent-eval --compare      # 多策略基准对比（default vs plan-execute）

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="RAG Agent 评测体系（hit@k + LLM-judge + Agent 行为层 + 基准对比）")
    parser.add_argument("--strategy", default=None, choices=["auto", "default", "multi_query", "top_k"],
                        help="检索策略：auto=bandit在线学习 / default / multi_query / top_k（默认 None=不覆盖）")
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM-judge 判分（更快，仅检索命中率）")
    parser.add_argument("--limit", type=int, default=None, help="最多评测的用例数")
    parser.add_argument("--cases", nargs="*", default=None,
                        help="限定工具: retrieve / search_faq / search_employees / search_orders / search_tickets")
    # P0-B Agent 行为层
    parser.add_argument("--agent-eval", action="store_true", help="运行 Agent 行为层评测（5 类分类评测集）")
    parser.add_argument("--compare", action="store_true", help="多策略基准对比（default vs plan-execute）")
    parser.add_argument("--benchmark-out", default="output/agent_benchmark.md", help="基准对比表导出路径")
    return parser.parse_args()


def _build_runtime():
    """初始化 LLM + 工具 + 图（与 main.py lifespan 顺序一致）"""
    from evals.runner import build_runtime
    return build_runtime()


def _run_agent_eval(args):
    """Agent 行为层评测（P0-B）"""
    from evals.observability import with_observability
    from evals.agent_eval import run_agent_eval, export_benchmark
    from evals.dataset.categorized import load_categorized_cases

    print(f"[eval] Agent 行为层评测启动（ENABLE_PLANNING={Config.ENABLE_PLANNING}）")
    llm_chat, llm_embedding, graph, tool_config = _build_runtime()

    cases = load_categorized_cases()
    if args.limit:
        cases = cases[: args.limit]
    print(f"[eval] 加载 {len(cases)} 个分类评测用例")

    if args.compare:
        # 多策略对比：default（快路径） vs plan-execute（规划）
        rows = []
        strategies = [
            ("plan-execute", True),
            ("default", False),
        ]
        for label, enable_planning in strategies:
            os.environ["ENABLE_PLANNING"] = "true" if enable_planning else "false"
            # 重新加载配置与图
            import importlib
            import utils.config as cfg_mod
            importlib.reload(cfg_mod)
            from utils.config import Config as C2
            print(f"\n[eval] 构建策略 [{label}] ENABLE_PLANNING={C2.ENABLE_PLANNING}")
            llm, emb, g, tc = _build_runtime()
            llm, obs = with_observability(llm)
            report = run_agent_eval(g, llm, cases=cases, judge_enabled=not args.no_judge,
                                    observer=obs, strategy_label=label)
            rows.append(report)
            _print_agent_report(report)
        md = export_benchmark(rows, args.benchmark_out)
        print(f"\n[eval] 基准对比表已导出: {args.benchmark_out}")
        print(md)
    else:
        llm, emb, g, tc = _build_runtime()
        llm, obs = with_observability(llm)
        report = run_agent_eval(g, llm, cases=cases, judge_enabled=not args.no_judge,
                                observer=obs, strategy_label="current")
        _print_agent_report(report)
        # 导出当前策略基准行
        export_benchmark([report], args.benchmark_out)


def _print_agent_report(report: dict):
    """控制台输出 Agent 行为层报告"""
    m = report["metrics"]
    print("\n" + "=" * 62)
    print(f"Agent 行为层评测报告（策略: {report['strategy']}）")
    print("=" * 62)
    print(f"{'指标':<22}{'数值':>12}")
    print("-" * 62)
    print(f"{'任务成功率 (LLM-judge)':<22}{m['task_success_rate']:>11.1%}")
    print(f"{'工具调用准确率':<22}{m['tool_accuracy']:>11.1%}")
    print(f"{'多步完成率':<22}{m['multi_step_completion_rate']:>11.1%}")
    print(f"{'无效 LLM 调用率':<22}{m['invalid_llm_call_rate']:>11.1%}")
    print(f"{'平均步数':<22}{m['avg_steps']:>11.1f}")
    print(f"{'平均延迟 (ms)':<22}{m['avg_latency_ms']:>11.1f}")
    print(f"{'平均 Token':<22}{m['avg_tokens']:>11.1f}")
    if "cost" in m:
        c = m["cost"]
        print(f"{'LLM 调用次数':<22}{c['llm_calls']:>11d}")
        print(f"{'总 Token':<22}{c['total_tokens']:>11d}")
    # 分类对比
    print("-" * 62)
    print("按类别工具准确率:")
    for cat, agg in report["by_category"].items():
        print(f"  {cat:<16} 准确率={agg['tool_accuracy']:.1%}  命中率={agg['hit_rate']:.1%}  (n={agg['total']})")
    print("=" * 62)


def main():
    args = parse_args()

    # 若指定 --strategy 且非 auto，检查开关是否开启
    if args.strategy and args.strategy != "auto" and not Config.ENABLE_SEARCH_AGENT:
        print("[eval] 提示: ENABLE_SEARCH_AGENT 未开启（默认关），--strategy 固定策略不影响图结构，"
              "仅当开关开启时 search_agent 节点才生效。继续评测（按原链路跑）。")

    print(f"[eval] LLM_TYPE={Config.LLM_TYPE}  ENABLE_SEARCH_AGENT={Config.ENABLE_SEARCH_AGENT}")

    # Agent 行为层评测入口
    if args.agent_eval:
        _run_agent_eval(args)
        return

    # 原检索层评测
    print("[eval] 初始化运行时（LLM → 工具 → 图）...")
    from evals.runner import run_eval
    llm_chat, llm_embedding, graph, tool_config = _build_runtime()
    print("[eval] 运行时初始化完成")

    run_eval(llm_chat, llm_embedding, graph, tool_config, args)


if __name__ == "__main__":
    main()
