# evals/metrics.py
# 评测指标：检索命中率 hit@k 判定
# 纯字符串判定（不调 LLM）：去除空白后，任一 golden 子串出现在工具输出中即命中

import re
from typing import List


def _strip_ws(text: str) -> str:
    """去除所有空白字符"""
    return re.sub(r"\s+", "", text)


def check_hit(tool_output: str, golden_contains: List[str]) -> bool:
    """判断工具输出是否命中任一金标准子串

    Args:
        tool_output: 工具返回的文本
        golden_contains: 金标准子串列表（任一出现即命中）

    Returns:
        bool: 是否命中
    """
    if not tool_output or not golden_contains:
        return False
    normalized = _strip_ws(str(tool_output))
    return any(_strip_ws(g) in normalized for g in golden_contains)


def hit_at_k(samples: List[bool]) -> float:
    """计算命中率

    Args:
        samples: 每个用例是否命中的布尔列表

    Returns:
        float: 命中率（0.0~1.0），空列表返回 0.0
    """
    if not samples:
        return 0.0
    return sum(samples) / len(samples)


def accuracy(samples: List[bool]) -> float:
    """通过率（与 hit_at_k 同义，用于 judge 结果）"""
    return hit_at_k(samples)
