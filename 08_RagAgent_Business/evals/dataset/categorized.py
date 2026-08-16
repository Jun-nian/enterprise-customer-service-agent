# evals/dataset/categorized.py
# P0-B 评测基准：5 类分类评测集
#   1. single_tool  — 单工具精确查询（最简单，验证基线能力）
#   2. multi_tool   — 多工具多步查询（验证 Plan-and-Execute 多步拆解）
#   3. cross_source — 跨数据源对比/聚合（验证跨领域协作）
#   4. no_solution  — 无解/边界查询（验证正确拒答，不产生无效工具调用）
#   5. adversarial  — 对抗/模糊查询（验证鲁棒性与护栏兜底）
#
# 每个用例含：
#   - id / category / query
#   - expected_tools: 期望调用的工具名集合（Agent 行为层"工具调用准确率"判定）
#   - golden_contains: 命中判定的金标准子串（LLM-judge 依据）
#   - complexity_hint: 提示该用例期望的规划模式（simple/complex/na）
#
# 注意：data/*.json 为模拟数据，用例金标准基于真实字段构造，避免歧义。

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AgentTestCase:
    """Agent 行为层评测用例（扩展自原 TestCase）"""
    id: str
    category: str                                   # single_tool / multi_tool / cross_source / no_solution / adversarial
    query: str
    expected_tools: List[str] = field(default_factory=list)   # 期望调用的工具集合
    golden_contains: List[str] = field(default_factory=list)  # 最终回答应包含的金标准子串
    complexity_hint: str = "na"                     # simple / complex / na
    note: str = ""


# 确定性用例（不依赖 data 文件内容，保证可复现）
HANDCRAFTED: List[AgentTestCase] = [
    # ---------- 1. 单工具 ----------
    AgentTestCase("single_emp_dept", "single_tool", "张建国的邮箱是什么？",
                  expected_tools=["search_employees"], golden_contains=["zhangjianguo@xinrui.com"],
                  complexity_hint="simple", note="单工具精确查询"),
    AgentTestCase("single_faq_leave", "single_tool", "公司年假政策是什么？",
                  expected_tools=["search_faq"], golden_contains=["年假"],
                  complexity_hint="simple", note="单工具 FAQ 查询"),

    # ---------- 2. 多工具多步 ----------
    AgentTestCase("multi_emp_ticket", "multi_tool",
                  "查询张建国的部门，同时查一下他名下的IT工单",
                  expected_tools=["search_employees", "search_tickets"],
                  golden_contains=["张建国"],
                  complexity_hint="complex", note="员工+工单双工具多步"),
    AgentTestCase("multi_faq_order", "multi_tool",
                  "分别查询公司的报销制度，以及有哪些销售订单",
                  expected_tools=["search_faq", "search_orders"],
                  golden_contains=["报销"],
                  complexity_hint="complex", note="FAQ+订单双工具"),

    # ---------- 3. 跨源聚合 ----------
    AgentTestCase("cross_order_summary", "cross_source",
                  "统计一下所有订单的数量和总金额，并列出处于已发货状态的订单",
                  expected_tools=["search_orders"],
                  golden_contains=["订单"],
                  complexity_hint="complex", note="跨源聚合统计"),
    AgentTestCase("cross_emp_order", "cross_source",
                  "对比张建国和李娜的职位，以及他们的订单情况",
                  expected_tools=["search_employees", "search_orders"],
                  golden_contains=["张建国"],
                  complexity_hint="complex", note="员工+订单跨源对比"),

    # ---------- 4. 无解/边界 ----------
    AgentTestCase("nosol_nonexist", "no_solution",
                  "查询员工小明（公司没有此人）的工资信息",
                  expected_tools=["search_employees"], golden_contains=["未找到"],
                  complexity_hint="na", note="不存在实体，应正确拒答"),
    AgentTestCase("nosol_greeting", "no_solution",
                  "你好呀", expected_tools=[], golden_contains=[],
                  complexity_hint="simple", note="纯问候，无需工具"),

    # ---------- 5. 对抗/模糊 ----------
    AgentTestCase("adv_ambiguous", "adversarial",
                  "给我看下所有和钱有关的记录",
                  expected_tools=[], golden_contains=[],
                  complexity_hint="na", note="模糊查询，应澄清而非瞎猜"),
    AgentTestCase("adv_false_premise", "adversarial",
                  "为什么我的工单被删除了？（实际上工单号不存在）",
                  expected_tools=["search_tickets"], golden_contains=[],
                  complexity_hint="na", note="虚假前提，应核实后澄清"),
]


def load_categorized_cases() -> List[AgentTestCase]:
    """加载 5 类分类评测集"""
    return list(HANDCRAFTED)


def load_cases_by_category(category: str) -> List[AgentTestCase]:
    """按类别加载用例"""
    return [c for c in HANDCRAFTED if c.category == category]


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    from collections import Counter
    cases = load_categorized_cases()
    print(f"分类评测集总数: {len(cases)}")
    print(Counter(c.category for c in cases))
    for c in cases:
        print(f"  [{c.category}] {c.id}: {c.query} → {c.expected_tools}")
