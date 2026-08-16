# evals/memory_eval.py
# P2-A 记忆检索质量评估：评估跨线程长期记忆的检索质量
# 模拟多个用户记忆存储，评估 InMemoryStore 向量检索的命中与排序质量。
#
# 面试话术钩子：
#   "在检索层/行为层评测之外，补上记忆层评估——验证长期记忆的向量检索
#    能否召回正确用户记忆，为多轮记忆增强对话提供量化依据。"

import os
import sys
import uuid

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from typing import List


def build_memory_store(llm_embedding):
    """构建带向量索引的内存记忆库"""
    from langgraph.store.memory import InMemoryStore
    from utils.llms import get_embedding_dims
    dims = get_embedding_dims(llm_embedding)
    store = InMemoryStore(index={"dims": dims, "embed": llm_embedding})
    return store


def seed_memories(store, user_id: str, memories: List[str]):
    """为指定用户写入多条记忆"""
    namespace = ("memories", user_id)
    for m in memories:
        store.put(namespace, str(uuid.uuid4()), {"data": m})


def eval_memory_retrieval(store, user_id: str, query: str, expected_substr: str,
                          top_k: int = 3) -> dict:
    """评估单条查询的记忆检索命中

    Args:
        store: InMemoryStore
        user_id: 用户
        query: 检索查询（模拟新对话中的语义查询）
        expected_substr: 期望召回的答案中包含的子串
        top_k: 检索数量

    Returns:
        dict: {hit, recall_position, retrieved}
    """
    namespace = ("memories", user_id)
    results = store.search(namespace, query=query, limit=top_k)
    retrieved = [d.value["data"] for d in results]

    hit = False
    recall_position = -1
    for i, r in enumerate(retrieved):
        if expected_substr in r:
            hit = True
            recall_position = i + 1
            break
    return {"hit": hit, "recall_position": recall_position, "retrieved": retrieved}


def run_memory_eval(llm_embedding, verbose: bool = True) -> dict:
    """执行记忆检索质量评估

    Args:
        llm_embedding: 嵌入模型
        verbose: 是否打印明细

    Returns:
        dict: 评估报告
    """
    store = build_memory_store(llm_embedding)

    # 构造评估记忆：两个用户的差异化记忆
    seed_memories(store, "user_a", [
        "我的名字是小明，我是技术部的前端工程师",
        "我偏好简洁直接的回复风格",
        "我的项目是公司官网改版，使用 Vue3",
    ])
    seed_memories(store, "user_b", [
        "我是HR部门的小红，负责员工考勤",
        "我经常需要查询报销流程",
    ])

    # 评估用例：(查询, 期望子串, 说明)
    cases = [
        ("我叫什么名字", "小明", "用户记忆-姓名"),
        ("我的工作内容是什么", "前端工程师", "用户记忆-岗位"),
        ("我偏好什么风格", "简洁直接", "用户记忆-偏好"),
        ("我负责什么项目", "官网改版", "用户记忆-项目"),
        ("我是哪个部门的", "HR", "用户记忆-部门（跨用户）"),
    ]

    results = []
    for query, expected, note in cases:
        r = eval_memory_retrieval(store, "user_a", query, expected)
        r["query"] = query
        r["expected"] = expected
        r["note"] = note
        results.append(r)
        if verbose:
            print(f"  [{'✓' if r['hit'] else '✗'}] {note} | 查询: {query!r} 期望: {expected!r}"
                  f" 召回位: {r['recall_position']}")

    hits = sum(1 for r in results if r["hit"])
    recall_at_k = hits / len(results) if results else 0.0

    report = {
        "cases_total": len(results),
        "hits": hits,
        "recall_at_k": recall_at_k,
        "avg_recall_position": round(
            sum(r["recall_position"] for r in results if r["hit"]) / max(hits, 1), 2
        ),
        "results": [
            {"query": r["query"], "expected": r["expected"], "hit": r["hit"],
             "recall_position": r["recall_position"], "note": r["note"]}
            for r in results
        ],
    }
    if verbose:
        print(f"\n  记忆检索 recall@{3} = {recall_at_k:.1%}")
    return report


if __name__ == "__main__":
    from utils.llms import get_llm
    llm_chat, llm_embedding = get_llm("ollama")
    run_memory_eval(llm_embedding, verbose=True)
