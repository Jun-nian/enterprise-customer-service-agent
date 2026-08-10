# eval_rag.py
# 评测体系 CLI 入口：对测试集跑完整 graph 流程，输出 hit@k + LLM-judge 报告
# 用法示例:
#   python eval_rag.py --limit 20                  # 纯检索评测（默认含 LLM-judge，可加 --no-judge）
#   python eval_rag.py --no-judge                  # 跳过 LLM-judge（更快）
#   python eval_rag.py --strategy default          # RL 固定 default 策略（基线）
#   python eval_rag.py --strategy auto             # RL bandit 在线学习（需 ENABLE_SEARCH_AGENT=true）
#   python eval_rag.py --cases search_faq          # 只评测 FAQ 用例

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from evals.runner import build_runtime, run_eval
from utils.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="RAG Agent 评测体系（hit@k + LLM-judge + RL 对比）")
    parser.add_argument("--strategy", default=None, choices=["auto", "default", "multi_query", "top_k"],
                        help="检索策略：auto=bandit在线学习 / default / multi_query / top_k（默认 None=不覆盖）")
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM-judge 判分（更快，仅检索命中率）")
    parser.add_argument("--limit", type=int, default=None, help="最多评测的用例数")
    parser.add_argument("--cases", nargs="*", default=None,
                        help="限定工具: retrieve / search_faq / search_employees / search_orders / search_tickets")
    return parser.parse_args()


def main():
    args = parse_args()

    # 若指定 --strategy 且非 auto，检查开关是否开启
    if args.strategy and args.strategy != "auto" and not Config.ENABLE_SEARCH_AGENT:
        print("[eval] 提示: ENABLE_SEARCH_AGENT 未开启（默认关），--strategy 固定策略不影响图结构，"
              "仅当开关开启时 search_agent 节点才生效。继续评测（按原链路跑）。")

    print(f"[eval] LLM_TYPE={Config.LLM_TYPE}  ENABLE_SEARCH_AGENT={Config.ENABLE_SEARCH_AGENT}")
    print("[eval] 初始化运行时（LLM → 工具 → 图）...")
    llm_chat, llm_embedding, graph, tool_config = build_runtime()
    print("[eval] 运行时初始化完成")

    run_eval(llm_chat, llm_embedding, graph, tool_config, args)


if __name__ == "__main__":
    main()
