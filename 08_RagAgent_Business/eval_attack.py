# eval_attack.py
# P2-B 攻击测试用例评测 CLI：量化输入护栏检测性能
# 用法:
#   python eval_attack.py                # 全量攻击测试集评估
#   python eval_attack.py --verbose      # 显示每个用例的判定
#
# 输出指标（对齐 vivo 项目实验设计）：
#   - 召回率（recall）：攻击用例被正确识别比例
#   - 误报率（FPR）：正常用例被误判为攻击的比例
#   - 准确率（accuracy）：全量正确判定比例

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="输入护栏攻击测试集评测")
    parser.add_argument("--verbose", action="store_true", help="显示每个用例的判定明细")
    return parser.parse_args()


def main():
    args = parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from guardrails.patterns.attack_patterns import detect_attack
    from guardrails.tests.attack_cases import load_attack_cases, evaluate_guard_detection

    cases = load_attack_cases()
    result = evaluate_guard_detection(detect_attack, cases)

    print("=" * 60)
    print("输入护栏攻击测试集评测报告")
    print("=" * 60)
    print(f"{'用例总数':<20}{result['total']}")
    print(f"{'正确拦截 (TP)':<20}{result['tp']}")
    print(f"{'漏检 (FN)':<20}{result['fn']}")
    print(f"{'正常正确 (TN)':<20}{result['tn']}")
    print(f"{'误报 (FP)':<20}{result['fp']}")
    print("-" * 60)
    print(f"{'召回率 (Recall)':<20}{result['recall']:.1%}")
    print(f"{'误报率 (FPR)':<20}{result['false_positive_rate']:.1%}")
    print(f"{'准确率 (Accuracy)':<20}{result['accuracy']:.1%}")
    print("=" * 60)

    if args.verbose:
        print("\n用例明细:")
        for cid, detail in result["per_category"].items():
            flag = "✓" if detail["correct"] else "✗"
            exp = detail["expected"]
            got = detail["got"] or "clean"
            print(f"  {flag} {cid:<8} 期望={exp:<16} 实际={got}")

    # 导出报告
    out = os.path.join(PROJECT_ROOT, "output", "attack_eval_report.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import json
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n报告已导出: {out}")


if __name__ == "__main__":
    main()
