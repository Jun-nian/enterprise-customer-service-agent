# evals/dataset.py
# 评测测试集构造：从 data/*.json 生成 QA 用例
# ground truth 判定规则：
#   - faq:       query = 原问题，golden = 从 answer 中段截取的唯一短语
#   - employees: query = 姓名，golden = 姓名（字段天然唯一）
#   - orders:    query = 订单号，golden = 订单号
#   - tickets:   query = 工单号，golden = 工单号
#   - retrieve:  默认 0 例（与 Chroma 解耦），若存在 input/retrieve_qa.json 则读取追加

import json
import os
import re
from dataclasses import dataclass, field
from typing import List

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RETRIEVE_QA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input", "retrieve_qa.json"
)


@dataclass
class TestCase:
    """单个评测用例"""
    id: str                # 用例唯一 ID，如 faq_0 / employees_3
    tool_name: str         # 期望调用的工具名: retrieve/search_employees/search_faq/search_orders/search_tickets
    query: str             # 用户问题
    golden_contains: List[str] = field(default_factory=list)  # 命中判定的金标准子串（任一命中即 hit）


def _strip_ws(text: str) -> str:
    """去除所有空白字符，用于子串匹配"""
    return re.sub(r"\s+", "", text)


def _pick_golden_phrase(answer: str, min_len: int = 12, max_len: int = 20) -> str:
    """从答案中段截取唯一短语作为金标准（避开首段高频关键词，避免歧义误判）"""
    text = _strip_ws(answer)
    if len(text) <= max_len:
        return text
    # 从中段 40%~60% 位置截取
    start = int(len(text) * 0.4)
    end = min(start + max_len, len(text))
    phrase = text[start:end]
    # 若截取位置落在半个中文词上，向下取整到字符边界（中文按字截取即可，无需分词）
    return phrase


def _build_faq_cases(faqs: list) -> List[TestCase]:
    cases = []
    for i, faq in enumerate(faqs):
        answer = faq.get("answer", "")
        golden = _pick_golden_phrase(answer)
        if not golden:
            continue
        cases.append(TestCase(
            id=f"faq_{i}",
            tool_name="search_faq",
            query=faq.get("question", ""),
            golden_contains=[golden],
        ))
    return cases


def _build_employees_cases(employees: list) -> List[TestCase]:
    cases = []
    for i, emp in enumerate(employees):
        name = emp.get("name", "")
        if not name:
            continue
        cases.append(TestCase(
            id=f"employees_{i}",
            tool_name="search_employees",
            query=name,
            golden_contains=[name],
        ))
    return cases


def _build_orders_cases(orders: list) -> List[TestCase]:
    cases = []
    for i, order in enumerate(orders):
        oid = order.get("order_id", "")
        if not oid:
            continue
        cases.append(TestCase(
            id=f"orders_{i}",
            tool_name="search_orders",
            query=oid,
            golden_contains=[oid],
        ))
    return cases


def _build_tickets_cases(tickets: list) -> List[TestCase]:
    cases = []
    for i, ticket in enumerate(tickets):
        tid = ticket.get("ticket_id", "")
        if not tid:
            continue
        cases.append(TestCase(
            id=f"tickets_{i}",
            tool_name="search_tickets",
            query=tid,
            golden_contains=[tid],
        ))
    return cases


def _load_retrieve_cases() -> List[TestCase]:
    """读取 input/retrieve_qa.json（可选）追加 retrieve 用例

    格式: [{"query": "...", "golden_contains": ["短语1"]}, ...]
    """
    if not os.path.exists(RETRIEVE_QA_FILE):
        return []
    try:
        with open(RETRIEVE_QA_FILE, "r", encoding="utf-8") as f:
            items = json.load(f)
        return [
            TestCase(
                id=f"retrieve_{i}",
                tool_name="retrieve",
                query=item.get("query", ""),
                golden_contains=item.get("golden_contains", []),
            )
            for i, item in enumerate(items)
        ]
    except Exception as e:
        print(f"[dataset] 读取 {RETRIEVE_QA_FILE} 失败: {e}")
        return []


def load_test_cases(data_dir: str = DATA_DIR) -> List[TestCase]:
    """从 data/*.json 加载全部评测用例"""
    cases: List[TestCase] = []
    for fname, builder in [
        ("faq.json", _build_faq_cases),
        ("employees.json", _build_employees_cases),
        ("orders.json", _build_orders_cases),
        ("tickets.json", _build_tickets_cases),
    ]:
        filepath = os.path.join(data_dir, fname)
        if not os.path.exists(filepath):
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            cases.extend(builder(data))
        except Exception as e:
            print(f"[dataset] 加载 {filepath} 失败: {e}")
    cases.extend(_load_retrieve_cases())
    return cases


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    all_cases = load_test_cases()
    print(f"总用例数: {len(all_cases)}")
    from collections import Counter
    print(Counter(c.tool_name for c in all_cases))
    for c in all_cases[:3]:
        print(c)
